/** Format explicitly selected skill files without altering their contents. */
export function buildSkillPrompt(ids: string[], contents: string[]): string {
  if (ids.length !== contents.length || ids.length > 8) throw new Error('Invalid skill selection')
  return ids.map((id, index) => {
    if (!/^(workspace|user|user-agents|user-codex):[A-Za-z0-9_-]{1,100}$/.test(id)) throw new Error('Invalid skill identifier')
    return `<skill name="${id}">\n${contents[index]}\n</skill>`
  }).join('\n\n')
}
