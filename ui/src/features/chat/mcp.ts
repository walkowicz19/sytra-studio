import { api } from '../../api'

export interface IndexedTool {
  server: string
  tool: { name: string; description?: string; inputSchema: Record<string, unknown> }
}

export async function mcpRequest<T>(operation: string, payload: object, signal?: AbortSignal): Promise<T> {
  const token = await api.chatMcpToken()
  const response = await fetch(`http://127.0.0.1:8080/sytra/mcp/${operation}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Sytra-Session': token },
    body: JSON.stringify(payload), signal,
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.error || `MCP request failed (${response.status})`)
  return result as T
}
