#!/usr/bin/env sh
set -eu

VERSION="${SYTRA_VERSION:-1.3.0}"
REPO="${SYTRA_REPO:-walkowicz19/sytra-studio}"
INSTALL_DIR="${SYTRA_INSTALL_DIR:-$HOME/.sytra}"
CLIENTS="${SYTRA_MCP_CLIENTS:-auto}"
LOCAL_DIR="${SYTRA_LOCAL_DIR:-}"
SKIP_DESKTOP="${SYTRA_SKIP_DESKTOP:-0}"
SKIP_MCP="${SYTRA_SKIP_MCP:-0}"

OS_RAW="$(uname -s)"
case "$OS_RAW" in
  Linux*) PLATFORM="linux" ;;
  Darwin*) PLATFORM="macos" ;;
  *)
    echo "Error: Unsupported OS '$OS_RAW'. Use scripts/install.ps1 on Windows." >&2
    exit 1
    ;;
esac

if [ "$VERSION" = "latest" ]; then
  BASE_URL="https://github.com/$REPO/releases/latest/download"
else
  BASE_URL="https://github.com/$REPO/releases/download/$VERSION"
fi

BIN_DIR="$INSTALL_DIR/bin"
mkdir -p "$BIN_DIR" "$INSTALL_DIR/runs"

fetch_to() {
  url="$1"
  out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$out" "$url"
  else
    echo "Error: curl or wget is required to download Sytra binaries." >&2
    exit 1
  fi
}

CHECKSUMS_FILE="$BIN_DIR/.SHA256SUMS.tmp"
rm -f "$CHECKSUMS_FILE"
if [ -z "$LOCAL_DIR" ]; then
  fetch_to "$BASE_URL/SHA256SUMS.txt" "$CHECKSUMS_FILE" 2>/dev/null || rm -f "$CHECKSUMS_FILE"
fi

verify_checksum() {
  asset="$1"
  file="$2"
  [ -f "$CHECKSUMS_FILE" ] || return 0
  expected="$(grep -E "[[:space:]]\*?${asset}\$" "$CHECKSUMS_FILE" | awk '{print $1}' | head -n 1 || true)"
  [ -n "$expected" ] || return 0
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$file" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "$file" | awk '{print $1}')"
  else
    return 0
  fi
  if [ "$actual" != "$expected" ]; then
    echo "Error: SHA-256 mismatch for $asset (expected $expected, got $actual)" >&2
    exit 1
  fi
}

install_asset() {
  asset="$1"
  plain_name="$2"
  dest="$BIN_DIR/$plain_name"
  tmp="$dest.tmp"
  rm -f "$tmp"
  if [ -n "$LOCAL_DIR" ]; then
    if [ -f "$LOCAL_DIR/$asset" ]; then
      cp "$LOCAL_DIR/$asset" "$tmp"
    elif [ -f "$LOCAL_DIR/$plain_name" ]; then
      cp "$LOCAL_DIR/$plain_name" "$tmp"
    else
      echo "Error: local binary not found in $LOCAL_DIR for $asset" >&2
      exit 1
    fi
  else
    echo "Downloading $BASE_URL/$asset..."
    fetch_to "$BASE_URL/$asset" "$tmp"
  fi
  size="$(wc -c < "$tmp" | tr -d ' ')"
  if [ "$size" -lt 524288 ]; then
    rm -f "$tmp"
    echo "Error: downloaded $asset is unexpectedly small ($size bytes)" >&2
    exit 1
  fi
  verify_checksum "$asset" "$tmp"
  chmod +x "$tmp"
  mv -f "$tmp" "$dest"
}

install_asset "sytra-mcp-$PLATFORM" "sytra-mcp"
if [ "$SKIP_DESKTOP" != "1" ]; then
  install_asset "sytra-studio-$PLATFORM" "sytra-studio"
fi
rm -f "$CHECKSUMS_FILE"

"$BIN_DIR/sytra-mcp" status --workspace "$INSTALL_DIR" >/dev/null
if [ "$SKIP_MCP" != "1" ]; then
  "$BIN_DIR/sytra-mcp" install --client "$CLIENTS" --workspace "$INSTALL_DIR"
fi

echo ""
echo "Sytra Studio installed under $INSTALL_DIR"
echo "  MCP     : $BIN_DIR/sytra-mcp"
if [ "$SKIP_DESKTOP" != "1" ]; then
  echo "  Desktop : $BIN_DIR/sytra-studio"
fi
case ":${PATH}:" in
  *:"$BIN_DIR":*) ;;
  *)
    echo "Add to PATH: export PATH=\"$BIN_DIR:\$PATH\""
    ;;
esac
