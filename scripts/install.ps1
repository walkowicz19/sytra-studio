[CmdletBinding()]
param(
    [string]$Version = $(if ($env:SYTRA_VERSION) { $env:SYTRA_VERSION } else { '1.3.1' }),
    [string]$Repo = $(if ($env:SYTRA_REPO) { $env:SYTRA_REPO } else { 'walkowicz19/sytra-studio' }),
    [string]$InstallDir = $(if ($env:SYTRA_INSTALL_DIR) { $env:SYTRA_INSTALL_DIR } else { Join-Path $env:USERPROFILE '.sytra' }),
    [string]$Clients = $(if ($env:SYTRA_MCP_CLIENTS) { $env:SYTRA_MCP_CLIENTS } else { 'auto' }),
    [string]$LocalDir = $env:SYTRA_LOCAL_DIR,
    [switch]$SkipDesktop,
    [switch]$SkipMcp,
    [switch]$NoPath,
    [switch]$NoShortcut
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Test-PeBinary {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $info = Get-Item -LiteralPath $Path
    if ($info.Length -lt 524288) { return $false }
    $fs = [System.IO.File]::OpenRead($Path)
    try {
        $b0 = $fs.ReadByte()
        $b1 = $fs.ReadByte()
        return ($b0 -eq 0x4D -and $b1 -eq 0x5A)
    } finally {
        $fs.Dispose()
    }
}

function Get-ReleaseBaseUrl {
    param([string]$RepoSlug, [string]$Tag)
    if ($Tag -eq 'latest') {
        return "https://github.com/$RepoSlug/releases/latest/download"
    }
    return "https://github.com/$RepoSlug/releases/download/$Tag"
}

function Get-ChecksumMap {
    param([string]$BaseUrl)
    $map = @{}
    $url = "$BaseUrl/SHA256SUMS.txt"
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 20
        foreach ($line in ($resp.Content -split "`r?`n")) {
            if ($line -match '^\s*([0-9a-fA-F]{64})\s+\*?(.+?)\s*$') {
                $map[$matches[2].Trim()] = $matches[1].ToLowerInvariant()
            }
        }
    } catch {
        # Release may predate SHA256SUMS.txt generation; fall back to size + MZ validation.
    }
    return $map
}

function Install-BinaryAsset {
    param(
        [string]$AssetName,
        [string]$DestPath,
        [string]$BaseUrl,
        [hashtable]$Checksums,
        [string]$SourceDir
    )
    $tmpPath = "$DestPath.tmp"
    if (Test-Path -LiteralPath $tmpPath) { Remove-Item -LiteralPath $tmpPath -Force }

    try {
        if ($SourceDir) {
            $localFile = Join-Path $SourceDir $AssetName
            if (-not (Test-Path -LiteralPath $localFile)) {
                $altName = $AssetName -replace '-windows\.exe$', '.exe'
                $localFile = Join-Path $SourceDir $altName
            }
            if (-not (Test-Path -LiteralPath $localFile)) {
                throw "Local binary not found in '$SourceDir' for '$AssetName'."
            }
            Write-Host "Installing $AssetName from local source..."
            Copy-Item -LiteralPath $localFile -Destination $tmpPath -Force
        } else {
            $url = "$BaseUrl/$AssetName"
            Write-Host "Downloading $url..."
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $url -OutFile $tmpPath -UseBasicParsing
        }

        if (-not (Test-PeBinary -Path $tmpPath)) {
            throw "Downloaded file for $AssetName is corrupted or not a valid Windows executable."
        }

        if ($Checksums -and $Checksums.ContainsKey($AssetName)) {
            $expected = $Checksums[$AssetName]
            $actual = (Get-FileHash -LiteralPath $tmpPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne $expected) {
                throw "SHA-256 mismatch for $AssetName (expected $expected, got $actual)."
            }
            Write-Host "  SHA-256 verified ($AssetName)."
        }

        Move-Item -LiteralPath $tmpPath -Destination $DestPath -Force
    } catch {
        if (Test-Path -LiteralPath $tmpPath) { Remove-Item -LiteralPath $tmpPath -Force }
        throw
    }
}

$binDir = Join-Path $InstallDir 'bin'
$runsDir = Join-Path $InstallDir 'runs'
New-Item -ItemType Directory -Path $binDir -Force | Out-Null
New-Item -ItemType Directory -Path $runsDir -Force | Out-Null

# Auto-detect unpacked portable folder when running install.ps1 directly beside the .exe files
$resolvedLocalDir = $LocalDir
if (-not $resolvedLocalDir -and $PSScriptRoot) {
    $siblingMcp = Join-Path $PSScriptRoot 'sytra-mcp.exe'
    $parentMcp = Join-Path (Split-Path -Parent $PSScriptRoot) 'sytra-mcp.exe'
    if (Test-Path -LiteralPath $siblingMcp) {
        $resolvedLocalDir = $PSScriptRoot
    } elseif (Test-Path -LiteralPath $parentMcp) {
        $resolvedLocalDir = Split-Path -Parent $PSScriptRoot
    }
}

$baseUrl = Get-ReleaseBaseUrl -RepoSlug $Repo -Tag $Version
$checksums = if ($resolvedLocalDir) { @{} } else { Get-ChecksumMap -BaseUrl $baseUrl }

$mcpDest = Join-Path $binDir 'sytra-mcp.exe'
$studioDest = Join-Path $binDir 'sytra-studio.exe'

Install-BinaryAsset -AssetName 'sytra-mcp-windows.exe' -DestPath $mcpDest -BaseUrl $baseUrl -Checksums $checksums -SourceDir $resolvedLocalDir
if (-not $SkipDesktop) {
    Install-BinaryAsset -AssetName 'sytra-studio-windows.exe' -DestPath $studioDest -BaseUrl $baseUrl -Checksums $checksums -SourceDir $resolvedLocalDir
}

# Trigger embedded Python runner deployment cleanly without Node/NPM
Write-Host "Deploying Sytra Python runner to $InstallDir\runner..."
& $mcpDest status --workspace $InstallDir | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "sytra-mcp.exe status failed with exit code $LASTEXITCODE"
}

if (-not $NoPath) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @()
    if ($userPath) { $parts = $userPath -split ';' | Where-Object { $_ -and $_.Trim() -ne '' } }
    if ($parts -notcontains $binDir) {
        $newPath = (($parts + $binDir) -join ';')
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        if (($env:Path -split ';') -notcontains $binDir) {
            $env:Path = "$binDir;$env:Path"
        }
        Write-Host "Added $binDir to User PATH."
    }
}

if (-not $SkipDesktop -and -not $NoShortcut -and (Test-Path -LiteralPath $studioDest)) {
    try {
        $startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
        New-Item -ItemType Directory -Path $startMenu -Force | Out-Null
        $lnkPath = Join-Path $startMenu 'Sytra Studio.lnk'
        $wsh = New-Object -ComObject WScript.Shell
        $shortcut = $wsh.CreateShortcut($lnkPath)
        $shortcut.TargetPath = $studioDest
        $shortcut.WorkingDirectory = $InstallDir
        $shortcut.Description = 'Sytra Studio — Local LLM Fine-Tuning, Merging & Serving'
        $shortcut.Save()
        Write-Host "Created Start Menu shortcut: $lnkPath"
    } catch {
        Write-Warning "Could not create Start Menu shortcut: $_"
    }
}

if (-not $SkipMcp) {
    Write-Host "Configuring MCP clients ($Clients) without NPM..."
    & $mcpDest install --client $Clients --workspace $InstallDir
}

Write-Host ""
Write-Host "Sytra Studio installed under $InstallDir" -ForegroundColor Green
if (-not $SkipDesktop) {
    Write-Host "  Desktop : $studioDest"
}
Write-Host "  MCP     : $mcpDest"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Warning "'uv' was not found on PATH. Training/downloading requires uv (install via: irm https://astral.sh/uv/install.ps1 | iex)."
}
