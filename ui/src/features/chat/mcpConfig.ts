export interface ParsedMcpServer {
  server: string
  argv: string[]
  env?: Record<string, string>
  cwd?: string
}

function normalizeServerName(raw: string): string {
  const cleaned = raw.trim().replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 64)
  if (!cleaned || cleaned === 'sandbox') {
    throw new Error(`Invalid MCP server name "${raw}". Use letters, numbers, '-' or '_' (except "sandbox").`)
  }
  return cleaned
}

function parseServerEntry(rawName: string, entry: unknown): ParsedMcpServer {
  const server = normalizeServerName(rawName)
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
    throw new Error(`Server "${server}" must be a JSON object with a "command".`)
  }
  const obj = entry as Record<string, unknown>
  if (typeof obj.command !== 'string' || !obj.command.trim()) {
    throw new Error(`Server "${server}" is missing a string "command" (remote URL-only servers are not supported in stdio Chat MCP).`)
  }
  let args: string[] = []
  if (obj.args !== undefined) {
    if (!Array.isArray(obj.args) || !obj.args.every((item) => typeof item === 'string')) {
      throw new Error(`Server "${server}" has invalid "args" (expected an array of strings).`)
    }
    args = obj.args.map((a) => a)
  }
  const argv = [obj.command.trim(), ...args]
  let env: Record<string, string> | undefined
  if (obj.env !== undefined) {
    if (!obj.env || typeof obj.env !== 'object' || Array.isArray(obj.env)) {
      throw new Error(`Server "${server}" has invalid "env" (expected an object of strings).`)
    }
    env = {}
    for (const [k, v] of Object.entries(obj.env as Record<string, unknown>)) {
      if (typeof v !== 'string') {
        throw new Error(`Server "${server}" env value for "${k}" must be a string.`)
      }
      env[k] = v
    }
  }
  let cwd: string | undefined
  if (obj.cwd !== undefined) {
    if (typeof obj.cwd !== 'string' || !obj.cwd.trim()) {
      throw new Error(`Server "${server}" has invalid "cwd".`)
    }
    cwd = obj.cwd.trim()
  }
  return { server, argv, ...(env ? { env } : {}), ...(cwd ? { cwd } : {}) }
}

/**
 * Parses standard MCP JSON files (`mcp_config.json`, `mcp.json`, Claude Desktop
 * `mcpServers`, VS Code `servers`, or a single server object `{ command, args }`).
 */
export function parseMcpConfigJson(raw: string): ParsedMcpServer[] {
  const trimmed = raw.replace(/^\uFEFF/, '').trim()
  if (!trimmed) throw new Error('Paste or load a valid mcp_config.json payload.')
  const parsed: unknown = JSON.parse(trimmed)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('MCP config JSON must be an object.')
  }
  const root = parsed as Record<string, unknown>
  const container =
    (root.mcpServers && typeof root.mcpServers === 'object' && !Array.isArray(root.mcpServers) ? root.mcpServers : null)
    ?? (root.servers && typeof root.servers === 'object' && !Array.isArray(root.servers) ? root.servers : null)
    ?? (root.mcp_servers && typeof root.mcp_servers === 'object' && !Array.isArray(root.mcp_servers) ? root.mcp_servers : null)

  if (container) {
    const entries = Object.entries(container as Record<string, unknown>)
    if (!entries.length) throw new Error('No servers found inside mcpServers / servers.')
    return entries.map(([name, cfg]) => parseServerEntry(name, cfg))
  }

  if (typeof root.command === 'string') {
    const fallbackName = typeof root.name === 'string' && root.name.trim() ? root.name : 'mcp-server'
    return [parseServerEntry(fallbackName, root)]
  }

  throw new Error('Expected {"mcpServers": {...}}, {"servers": {...}}, or {"command": "...", "args": [...]}.')
}
