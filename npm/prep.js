const fs = require('fs');
const path = require('path');

function copyDirSync(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  const entries = fs.readdirSync(src, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      if (entry.name !== '__pycache__' && entry.name !== '.pytest_cache' && entry.name !== '.sytra-envs') {
        copyDirSync(srcPath, destPath);
      }
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

const rootRunner = path.join(__dirname, '..', 'runner');
const rootScripts = path.join(__dirname, '..', 'scripts');

const npmRunner = path.join(__dirname, 'runner');
const npmScripts = path.join(__dirname, 'scripts');

console.log('Preparing Sytra NPM package files...');

// Clean existing target dirs if any
if (fs.existsSync(npmRunner)) fs.rmSync(npmRunner, { recursive: true, force: true });
if (fs.existsSync(npmScripts)) fs.rmSync(npmScripts, { recursive: true, force: true });

// Copy runner & scripts from workspace root
if (fs.existsSync(rootRunner)) {
  console.log(`Copying runner from ${rootRunner} to ${npmRunner}...`);
  copyDirSync(rootRunner, npmRunner);
}
if (fs.existsSync(rootScripts)) {
  console.log(`Copying scripts from ${rootScripts} to ${npmScripts}...`);
  copyDirSync(rootScripts, npmScripts);
}

console.log('NPM package directories prepared successfully.');
