#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const os = require('os');
const https = require('https');
const { spawn } = require('child_process');

const VERSION = '1.2.0-freeze';
const RELEASE_TAG = '1.2.0';
const REPO = 'walkowicz19/sytra-studio';

const USAGE = `
Sytra Studio CLI Installer & Runner (${VERSION})

Usage:
  sytra         - Launch Sytra Studio Desktop application (downloads binaries if missing)
  sytra mcp     - Launch Sytra MCP server (for Claude Code, Cursor, Codex, etc.)
  sytra install - Force download/reinstall of Sytra binaries
  sytra help    - Show this help message
`;

const args = process.argv.slice(2);
const command = args[0] || 'gui';

if (command === 'help' || command === '--help' || command === '-h') {
  console.log(USAGE);
  process.exit(0);
}

const platformMap = {
  win32: 'windows',
  darwin: 'macos',
  linux: 'linux'
};

const platform = platformMap[process.platform];
if (!platform) {
  console.error(`Error: Unsupported platform "${process.platform}". Sytra supports Windows, macOS, and Linux.`);
  process.exit(1);
}

const exeName = platform === 'windows' ? 'sytra-studio.exe' : 'sytra-studio';
const mcpExeName = platform === 'windows' ? 'sytra-mcp.exe' : 'sytra-mcp';

const sytraDir = path.join(os.homedir(), '.sytra');
const binDir = path.join(sytraDir, 'bin');
const runnerDir = path.join(sytraDir, 'runner');
const scriptsDir = path.join(sytraDir, 'scripts');
const versionFile = path.join(binDir, 'VERSION');

const studioPath = path.join(binDir, exeName);
const mcpPath = path.join(binDir, mcpExeName);

function installedVersion() {
  try {
    return fs.readFileSync(versionFile, 'utf8').trim();
  } catch {
    return '';
  }
}

function firstExistingDir(candidates) {
  return candidates.find((candidate) => candidate && fs.existsSync(candidate));
}

function downloadFile(url, destPath) {
  const tmpPath = `${destPath}.tmp`;
  return new Promise((resolve, reject) => {
    function get(requestUrl) {
      https.get(requestUrl, (response) => {
        if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
          get(response.headers.location);
          return;
        }

        if (response.statusCode !== 200) {
          reject(new Error(`Failed to download from ${requestUrl}. Status Code: ${response.statusCode}`));
          return;
        }

        const file = fs.createWriteStream(tmpPath);
        response.pipe(file);

        file.on('finish', () => {
          file.close(() => {
            try {
              if (fs.existsSync(destPath)) fs.unlinkSync(destPath);
              fs.renameSync(tmpPath, destPath);
              resolve();
            } catch (e) {
              reject(e);
            }
          });
        });
      }).on('error', (err) => {
        if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
        reject(err);
      });
    }

    get(url);
  });
}

function copyDirSync(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  const entries = fs.readdirSync(src, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      copyDirSync(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

async function downloadBinary(fileName, destPath) {
  const urls = [
    `https://raw.githubusercontent.com/${REPO}/${RELEASE_TAG}/binaries/${platform}/${fileName}`,
    `https://raw.githubusercontent.com/${REPO}/main/binaries/${platform}/${fileName}`,
  ];
  let lastError;
  for (const url of urls) {
    try {
      await downloadFile(url, destPath);
      return;
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError;
}

async function installBinaries() {
  console.log(`Preparing Sytra ${VERSION} under ${sytraDir}...`);
  fs.mkdirSync(binDir, { recursive: true });
  fs.mkdirSync(runnerDir, { recursive: true });
  fs.mkdirSync(scriptsDir, { recursive: true });

  const packageRunner = firstExistingDir([
    path.join(__dirname, '..', 'runner'),
    path.join(__dirname, '..', '..', 'runner'),
  ]);
  const packageScripts = firstExistingDir([
    path.join(__dirname, '..', 'scripts'),
    path.join(__dirname, '..', '..', 'runner', 'scripts'),
    path.join(__dirname, '..', '..', 'scripts'),
  ]);

  if (packageRunner) {
    console.log('Deploying Python runner scripts...');
    copyDirSync(packageRunner, runnerDir);
  }

  if (packageScripts) {
    console.log('Deploying supporting scripts...');
    copyDirSync(packageScripts, scriptsDir);
  }

  console.log(`Downloading Sytra Studio Desktop (${platform}) from tag ${RELEASE_TAG}...`);
  try {
    await downloadBinary(exeName, studioPath);
    if (platform !== 'windows') {
      fs.chmodSync(studioPath, 0o755);
    }
    console.log('Sytra Studio Desktop download complete.');
  } catch (err) {
    console.error('Failed to download Sytra Studio Desktop binary:', err.message);
    process.exit(1);
  }

  console.log(`Downloading Sytra MCP Server (${platform}) from tag ${RELEASE_TAG}...`);
  try {
    await downloadBinary(mcpExeName, mcpPath);
    if (platform !== 'windows') {
      fs.chmodSync(mcpPath, 0o755);
    }
    console.log('Sytra MCP Server download complete.');
  } catch (err) {
    console.error('Failed to download Sytra MCP Server binary:', err.message);
    process.exit(1);
  }

  fs.writeFileSync(versionFile, VERSION, 'utf8');
  console.log(`\nInstallation completed successfully (${VERSION}).`);
}

async function run() {
  const staleVersion = installedVersion() !== VERSION;
  let needsInstall = command === 'install' || staleVersion || !fs.existsSync(studioPath) || !fs.existsSync(mcpPath);
  if (!needsInstall) {
    try {
      if (fs.statSync(studioPath).size < 1000000 || fs.statSync(mcpPath).size < 1000000) {
        needsInstall = true;
      }
    } catch {
      needsInstall = true;
    }
  }

  if (needsInstall) {
    await installBinaries();
    if (command === 'install') {
      process.exit(0);
    }
  }

  const env = { ...process.env, SYTRA_WORKSPACE: sytraDir };

  if (command === 'mcp') {
    console.error(`Starting Sytra MCP Server ${VERSION} from ${mcpPath}...`);
    const child = spawn(mcpPath, [], { env, stdio: 'inherit' });

    child.on('close', (code) => {
      process.exit(code || 0);
    });
  } else {
    console.log(`Launching Sytra Studio Desktop from ${studioPath}...`);
    const child = spawn(studioPath, [], {
      env,
      detached: true,
      stdio: 'ignore'
    });
    child.unref();
    process.exit(0);
  }
}

run().catch((err) => {
  console.error('An error occurred running Sytra:', err);
  process.exit(1);
});
