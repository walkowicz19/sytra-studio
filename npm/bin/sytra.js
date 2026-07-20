#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const os = require('os');
const https = require('https');
const { spawn } = require('child_process');

const VERSION = '1.0.0';
const REPO = 'walkowicz19/sytra-studio';

const USAGE = `
Sytra Studio CLI Installer & Runner (v${VERSION})

Usage:
  sytra         - Launch Sytra Studio Desktop application (downloads binaries if missing)
  sytra mcp     - Launch Sytra MCP server (for Claude Code, Cursor, Codex, etc.)
  sytra install - Force download/reinstall of Sytra binaries
  sytra help    - Show this help message
`;

// Parse command line arguments
const args = process.argv.slice(2);
const command = args[0] || 'gui';

if (command === 'help' || command === '--help' || command === '-h') {
  console.log(USAGE);
  process.exit(0);
}

// Map process.platform to binary platform folder
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

// Executable names
const exeName = platform === 'windows' ? 'sytra-studio.exe' : 'sytra-studio';
const mcpExeName = platform === 'windows' ? 'sytra-mcp.exe' : 'sytra-mcp';

// Local user directory directories
const sytraDir = path.join(os.homedir(), '.sytra');
const binDir = path.join(sytraDir, 'bin');
const runnerDir = path.join(sytraDir, 'runner');
const scriptsDir = path.join(sytraDir, 'scripts');

const studioPath = path.join(binDir, exeName);
const mcpPath = path.join(binDir, mcpExeName);

// Helper function to download a file with redirect support
function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);
    
    function get(requestUrl) {
      https.get(requestUrl, (response) => {
        // Handle HTTP Redirects
        if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
          get(response.headers.location);
          return;
        }
        
        if (response.statusCode !== 200) {
          reject(new Error(`Failed to download from ${requestUrl}. Status Code: ${response.statusCode}`));
          return;
        }
        
        response.pipe(file);
        
        file.on('finish', () => {
          file.close();
          resolve();
        });
      }).on('error', (err) => {
        fs.unlink(destPath, () => {}); // Delete temp file on error
        reject(err);
      });
    }
    
    get(url);
  });
}

// Synchronously copy directory recursively
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

async function installBinaries() {
  console.log(`Preparing Sytra installation directory under ${sytraDir}...`);
  fs.mkdirSync(binDir, { recursive: true });
  fs.mkdirSync(runnerDir, { recursive: true });
  fs.mkdirSync(scriptsDir, { recursive: true });

  // 1. Copy python runner & scripts from npm package source files
  const packageRunner = path.join(__dirname, '..', 'runner');
  const packageScripts = path.join(__dirname, '..', 'scripts');
  
  if (fs.existsSync(packageRunner)) {
    console.log('Deploying Python runner scripts...');
    copyDirSync(packageRunner, runnerDir);
  }
  
  if (fs.existsSync(packageScripts)) {
    console.log('Deploying supporting scripts...');
    copyDirSync(packageScripts, scriptsDir);
  }

  // 2. Download executables from Git raw files under tag 1.0.0
  const baseUrl = `https://raw.githubusercontent.com/${REPO}/1.0.0/binaries/${platform}`;
  
  console.log(`Downloading Sytra Studio Desktop (${platform})...`);
  try {
    await downloadFile(`${baseUrl}/${exeName}`, studioPath);
    // Mark as executable on macOS/Linux
    if (platform !== 'windows') {
      fs.chmodSync(studioPath, 0o755);
    }
    console.log('Sytra Studio Desktop download complete.');
  } catch (err) {
    console.error('Failed to download Sytra Studio Desktop binary:', err.message);
    process.exit(1);
  }

  console.log(`Downloading Sytra MCP Server (${platform})...`);
  try {
    await downloadFile(`${baseUrl}/${mcpExeName}`, mcpPath);
    // Mark as executable on macOS/Linux
    if (platform !== 'windows') {
      fs.chmodSync(mcpPath, 0o755);
    }
    console.log('Sytra MCP Server download complete.');
  } catch (err) {
    console.error('Failed to download Sytra MCP Server binary:', err.message);
    process.exit(1);
  }

  console.log('\nInstallation completed successfully!');
}

async function run() {
  const needsInstall = !fs.existsSync(studioPath) || !fs.existsSync(mcpPath);
  
  if (command === 'install' || needsInstall) {
    await installBinaries();
    if (command === 'install') {
      process.exit(0);
    }
  }

  // Setup SYTRA_WORKSPACE pointing to the .sytra directory in the user's home
  const env = { ...process.env, SYTRA_WORKSPACE: sytraDir };

  if (command === 'mcp') {
    console.log(`Starting Sytra MCP Server from ${mcpPath}...`);
    const child = spawn(mcpPath, [], { env, stdio: 'inherit' });
    
    child.on('close', (code) => {
      process.exit(code || 0);
    });
  } else {
    console.log(`Launching Sytra Studio Desktop from ${studioPath}...`);
    // For GUI, run detached and exit parent process so terminal remains free
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
