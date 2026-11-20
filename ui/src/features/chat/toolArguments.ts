/** Repair only trailing commas and missing closing brackets outside strings. */
export function parseToolArguments(raw: string): { value: Record<string, unknown>; repaired: boolean } {
  function object(text: string): Record<string, unknown> {
    const value: unknown = JSON.parse(text)
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Tool arguments must be an object')
    return value as Record<string, unknown>
  }
  try { return { value: object(raw), repaired: false } } catch { /* Try bounded structural repair. */ }
  let text = '', quoted = false, escaped = false
  const stack: string[] = []
  for (let index = 0; index < raw.length; index++) {
    const char = raw[index]
    if (quoted) {
      text += char
      if (escaped) escaped = false
      else if (char === '\\') escaped = true
      else if (char === '"') quoted = false
    } else if (char === '"') { quoted = true; text += char }
    else if (char === '{' || char === '[') { stack.push(char === '{' ? '}' : ']'); text += char }
    else if (char === '}' || char === ']') {
      if (stack.pop() !== char) throw new Error('Mismatched JSON brackets')
      text += char
    } else if (char === ',' && /^\s*[}\]]/.test(raw.slice(index + 1))) { /* remove trailing comma */ }
    else text += char
  }
  if (quoted || !stack.length && text === raw) throw new Error('Cannot safely repair tool arguments')
  if (stack.length) text = text.replace(/,\s*$/, '')
  text += stack.reverse().join('')
  return { value: object(text), repaired: true }
}
