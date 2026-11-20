<script lang="ts">
  import { t } from '../../i18n.svelte'
  import { api, type ChatSkill } from '../../api'
  import type { AgentAvatarKind, AgentComputerMode, AgentProfile, MemoryEntry } from '../../types'
  import AgentPet from './AgentPet.svelte'
  import { readImageFileAsDataUrl } from './chatCustomizer.svelte'

  interface Props {
    open: boolean
    agents: AgentProfile[]
    activeAgentId?: string
    skills: ChatSkill[]
    memories: MemoryEntry[]
    onClose: () => void
    onSelectAgent: (agentId: string | undefined) => void
    onAgentsChanged: () => Promise<void>
    onMemoriesChanged: () => Promise<void>
    onSkillsChanged: () => Promise<void>
    onRunRoutineNow: (agent: AgentProfile) => void
    onCompactNow: () => void
    contextPct: number
  }

  let {
    open,
    agents,
    activeAgentId,
    skills,
    memories,
    onClose,
    onSelectAgent,
    onAgentsChanged,
    onMemoriesChanged,
    onSkillsChanged,
    onRunRoutineNow,
    onCompactNow,
    contextPct,
  }: Props = $props()

  let tab = $state<'agents' | 'memory'>('agents')
  let editing = $state<AgentProfile>(blankAgent())
  let statusMsg = $state('')
  let memQuery = $state('')
  let newMemText = $state('')
  let newMemTags = $state('')
  let avatarFileInput = $state<HTMLInputElement | undefined>(undefined)

  function blankAgent(): AgentProfile {
    return {
      id: '',
      name: '',
      role: '',
      system_prompt: '',
      avatar_kind: 'pochi',
      avatar_url: null,
      provider: null,
      model_id: null,
      endpoint_url: null,
      skill_ids: [],
      computer_mode: 'sandbox',
      routine_enabled: false,
      routine_interval_mins: 5,
      routine_prompt: '',
      last_routine_at: null,
    }
  }

  function applyProactivePreset(preset: 'monitor' | 'digest' | 'workspace') {
    if (preset === 'monitor') {
      if (!editing.name.trim()) editing.name = 'Lil Sytra'
      editing.role = 'Live PC & Model Watcher'
      editing.system_prompt = 'Monitor GPU/RAM headroom, running models, and surface concise health KPIs and warnings.'
      editing.routine_enabled = true
      editing.routine_interval_mins = 5
      editing.routine_prompt = 'Inspect PC hardware telemetry, loaded local models, and summarize any bottleneck or tip cleanly in a KPI table.'
    } else if (preset === 'digest') {
      if (!editing.name.trim()) editing.name = 'Lil Sytra'
      editing.role = 'Autonomous Web Researcher'
      editing.system_prompt = 'Search the live web for updates on configured interests and present concise visual highlights.'
      editing.routine_enabled = true
      editing.routine_interval_mins = 15
      editing.routine_prompt = 'Check latest notable local AI / GGUF / benchmark highlights and summarize in a concise briefing table.'
    } else {
      if (!editing.name.trim()) editing.name = 'Lil Sytra'
      editing.role = 'Workspace Organizer'
      editing.system_prompt = 'Track sandbox or permitted workspace files and keep persistent cross-session notes organized.'
      editing.routine_enabled = true
      editing.routine_interval_mins = 10
      editing.routine_prompt = 'Inspect workspace files, summarize pending items or recent updates, and save a concise memory note.'
    }
  }

  function loadForEdit(agent: AgentProfile) {
    editing = {
      ...agent,
      skill_ids: [...(agent.skill_ids ?? [])],
    }
    statusMsg = ''
  }

  async function saveCurrentAgent() {
    statusMsg = ''
    const slug = (editing.id || editing.name).trim()
    if (!slug) {
      statusMsg = t('chat.agentNameRequired')
      return
    }
    try {
      const saved = await api.saveChatAgent({
        ...editing,
        id: slug,
        name: editing.name.trim() || slug,
      })
      editing = { ...saved, skill_ids: [...saved.skill_ids] }
      await onAgentsChanged()
      onSelectAgent(saved.id)
      statusMsg = t('chat.savedAgent', { name: saved.name })
    } catch (err) {
      statusMsg = String(err)
    }
  }

  async function removeAgent(id: string) {
    try {
      await api.deleteChatAgent(id)
      if (activeAgentId === id) onSelectAgent(undefined)
      if (editing.id === id) editing = blankAgent()
      await onAgentsChanged()
      statusMsg = t('chat.removedAgent', { name: id })
    } catch (err) {
      statusMsg = String(err)
    }
  }

  async function savePromptAsSkill() {
    const rawName = editing.name.trim() || editing.id.trim() || 'custom-agent-skill'
    const promptBody = editing.system_prompt.trim()
    if (!promptBody) {
      statusMsg = t('chat.promptRequired')
      return
    }
    try {
      const content = `# ${rawName}\n${editing.role ? `> ${editing.role}\n\n` : '\n'}${promptBody}\n`
      const created = await api.installChatSkill(rawName, content)
      if (!editing.skill_ids.includes(created.id)) {
        editing.skill_ids = [...editing.skill_ids, created.id]
      }
      await onSkillsChanged()
      statusMsg = t('chat.installedSkill', { name: created.id })
    } catch (err) {
      statusMsg = String(err)
    }
  }

  function toggleBoundSkill(id: string) {
    if (editing.skill_ids.includes(id)) {
      editing.skill_ids = editing.skill_ids.filter(s => s !== id)
    } else {
      editing.skill_ids = [...editing.skill_ids, id]
    }
  }

  async function pickCustomAvatar(event: Event) {
    const input = event.currentTarget as HTMLInputElement
    const file = input.files?.[0]
    input.value = ''
    if (!file) return
    try {
      const { dataUrl } = await readImageFileAsDataUrl(file, 450_000)
      editing.avatar_kind = 'custom'
      editing.avatar_url = dataUrl
    } catch (err) {
      statusMsg = String(err)
    }
  }

  async function addManualMemory() {
    if (!newMemText.trim()) return
    try {
      await api.saveAgentMemory({
        agentId: activeAgentId || 'default',
        sessionId: 'manual',
        sessionTitle: 'Manual Note',
        kind: 'fact',
        summary: newMemText.trim(),
        tags: newMemTags.split(',').map(s => s.trim()).filter(Boolean),
      })
      newMemText = ''
      newMemTags = ''
      await onMemoriesChanged()
      statusMsg = t('chat.savedMemory')
    } catch (err) {
      statusMsg = String(err)
    }
  }

  async function deleteMem(id: string) {
    await api.deleteAgentMemory(id)
    await onMemoriesChanged()
  }

  let visibleMemories = $derived.by(() => {
    const q = memQuery.trim().toLowerCase()
    if (!q) return memories
    return memories.filter(m =>
      m.summary.toLowerCase().includes(q)
      || m.session_title.toLowerCase().includes(q)
      || m.tags.some(tag => tag.toLowerCase().includes(q)),
    )
  })

  const SPECIES: { id: AgentAvatarKind; label: string }[] = [
    { id: 'pochi', label: 'Lil Sytra Bot' },
    { id: 'kitsune', label: 'Kitsune Fox' },
    { id: 'mochi', label: 'Mochi Slime' },
    { id: 'owl', label: 'Scholar Owl' },
    { id: 'custom', label: 'Custom GIF/IMG' },
  ]
  const MODES: Record<AgentComputerMode, string> = {
    sandbox: 'chat.localSandboxMode',
    pc_ask: 'chat.localAskMode',
    pc_auto: 'chat.localAutoMode',
  }
</script>

{#if open}
  <section class="studio-drawer" aria-label={t('chat.studio')}>
    <div class="studio-head">
      <div class="studio-tabs" role="tablist">
        <button type="button" role="tab" aria-selected={tab === 'agents'} class:active={tab === 'agents'} onclick={() => { tab = 'agents' }}>
          <i class="bi bi-robot" aria-hidden="true"></i> {t('chat.agentTab', { n: agents.length })}
        </button>
        <button type="button" role="tab" aria-selected={tab === 'memory'} class:active={tab === 'memory'} onclick={() => { tab = 'memory' }}>
          <i class="bi bi-database-check" aria-hidden="true"></i> {t('chat.memoryTab', { n: memories.length })}
        </button>
      </div>
      <button type="button" class="btn btn-secondary btn-sm" onclick={onClose}>{t('chat.close')}</button>
    </div>

    {#if statusMsg}
      <p class="studio-status" role="status">{statusMsg}</p>
    {/if}

    {#if tab === 'agents'}
      <div class="studio-body">
        <div class="agent-roster">
          <button
            type="button"
            class="agent-chip"
            class:selected={!activeAgentId}
            onclick={() => { onSelectAgent(undefined); editing = blankAgent() }}
          >
            <AgentPet species="pochi" size={32} interactive={false} />
            <div>
              <strong>{t('chat.defaultAgent')}</strong>
              <small>{t('chat.defaultAgentHint')}</small>
            </div>
          </button>

          {#each agents as ag (ag.id)}
            <div class="agent-chip-wrap" class:selected={activeAgentId === ag.id}>
              <button
                type="button"
                class="agent-chip-select"
                onclick={() => { onSelectAgent(ag.id); loadForEdit(ag) }}
              >
                <AgentPet species={ag.avatar_kind} customUrl={ag.avatar_url} size={32} interactive={false} />
                <div>
                  <strong>{ag.name}</strong>
                  <small>{t(ag.computer_mode === 'sandbox' ? 'chat.sandboxMode' : ag.computer_mode === 'pc_ask' ? 'chat.askMode' : 'chat.autoMode')}{ag.routine_enabled ? ` · ⏰ ${ag.routine_interval_mins}m` : ''}</small>
                </div>
              </button>
              {#if ag.routine_enabled}
                <button type="button" class="btn btn-ghost btn-sm" title={t('chat.runRoutine')} onclick={() => onRunRoutineNow(ag)}>▶</button>
              {/if}
              <button type="button" class="btn btn-ghost btn-sm" aria-label={t('chat.deleteAgent', { name: ag.name })} onclick={() => removeAgent(ag.id)}>×</button>
            </div>
          {/each}
          <button type="button" class="btn btn-secondary btn-sm" onclick={() => { editing = blankAgent(); statusMsg = '' }}>{t('chat.newAgent')}</button>
        </div>

        <div class="agent-editor">
          <div class="editor-top">
            <div class="pet-preview-frame">
              <AgentPet species={editing.avatar_kind} customUrl={editing.avatar_url} size={52} mood="idle" name={editing.name || t('chat.preview')} />
            </div>
            <div class="editor-fields">
              <div class="row-two">
                <input id="agent-name-input" class="input" placeholder={t('chat.agentName')} bind:value={editing.name} />
                <input id="agent-role-input" class="input" placeholder={t('chat.agentRole')} bind:value={editing.role} />
              </div>
              <div class="species-row">
                {#each SPECIES as sp}
                  <button
                    type="button"
                    class="pill-btn"
                    class:active={editing.avatar_kind === sp.id}
                    onclick={() => {
                      if (sp.id === 'custom') avatarFileInput?.click()
                      else editing.avatar_kind = sp.id
                    }}
                  >{sp.id === 'custom' ? t('chat.customAvatar') : sp.label}</button>
                {/each}
                <input bind:this={avatarFileInput} type="file" accept="image/*,.gif" style="display:none" onchange={pickCustomAvatar} />
              </div>
              <div class="species-row preset-row">
                <span class="field-title">{t('chat.quickPresets')}</span>
                <button type="button" class="pill-btn" onclick={() => applyProactivePreset('monitor')}>{t('chat.monitor')}</button>
                <button type="button" class="pill-btn" onclick={() => applyProactivePreset('digest')}>{t('chat.digest')}</button>
                <button type="button" class="pill-btn" onclick={() => applyProactivePreset('workspace')}>{t('chat.workspaceWatch')}</button>
              </div>
            </div>
          </div>

          <textarea
            id="agent-system-prompt"
            class="input"
            rows="3"
            placeholder={t('chat.systemPrompt')}
            bind:value={editing.system_prompt}
          ></textarea>

          <div class="row-two settings-pair">
            <label class="setting-card" for="agent-computer-mode">
              <span class="field-title">{t('chat.computerEnvironment')}</span>
              <select id="agent-computer-mode" class="input" bind:value={editing.computer_mode}>
                {#each Object.entries(MODES) as [mode, label] (mode)}
                  <option value={mode}>{t(label)}</option>
                {/each}
              </select>
            </label>

            <div class="setting-card">
              <span class="field-title">{t('chat.backgroundRoutine')}</span>
              <label class="routine-check">
                <input type="checkbox" bind:checked={editing.routine_enabled} />
                <span>{t('chat.runEvery')}</span>
                <input type="number" class="input mins-input" min="1" max="1440" bind:value={editing.routine_interval_mins} />
                <span>{t('chat.minutes')}</span>
              </label>
              {#if editing.routine_enabled}
                <input class="input routine-input" placeholder={t('chat.routineInstruction')} bind:value={editing.routine_prompt} />
              {/if}
            </div>
          </div>

          {#if skills.length}
            <div class="skill-bind-box">
              <span class="field-title">{t('chat.boundSkills')}</span>
              <div class="skill-bind-row">
                {#each skills as sk (sk.id)}
                  <button
                    type="button"
                    class="pill-btn"
                    class:active={editing.skill_ids.includes(sk.id)}
                    onclick={() => toggleBoundSkill(sk.id)}
                  >@{sk.name}</button>
                {/each}
              </div>
            </div>
          {/if}

          <div class="editor-actions">
            <button type="button" id="btn-save-agent-as-skill" class="btn btn-secondary btn-sm" onclick={savePromptAsSkill}>
              <i class="bi bi-list-check" aria-hidden="true"></i> {t('chat.savePrompt')}
            </button>
            <button type="button" id="btn-save-agent" class="btn btn-primary btn-sm" onclick={saveCurrentAgent}>
              <i class="bi bi-check2" aria-hidden="true"></i> {t('chat.saveAgent')}
            </button>
          </div>
        </div>
      </div>
    {:else}
      <div class="memory-pane">
        <div class="memory-toolbar">
          <input class="input" placeholder={t('chat.searchMemory')} bind:value={memQuery} />
          <button type="button" id="btn-compact-session-now" class="btn btn-secondary btn-sm" onclick={onCompactNow}>
            <i class="bi bi-stars" aria-hidden="true"></i> {t('chat.compactNow', { n: contextPct })}
          </button>
        </div>
        <div class="memory-add-row">
          <input id="new-memory-text" class="input" placeholder={t('chat.pinMemory')} bind:value={newMemText} />
          <input class="input tags-input" placeholder={t('chat.memoryTags')} bind:value={newMemTags} />
          <button type="button" id="btn-save-memory" class="btn btn-primary btn-sm" disabled={!newMemText.trim()} onclick={addManualMemory}>{t('chat.saveMemory')}</button>
        </div>
        <div class="memory-list">
          {#if !visibleMemories.length}
            <p class="mem-empty">{t('chat.emptyMemory')}</p>
          {:else}
            {#each visibleMemories as m (m.id)}
              <div class="mem-item">
                <div class="mem-head">
                  <span class={`mem-kind kind-${m.kind}`}>{m.kind}</span>
                  <strong>{m.session_title || t('chat.session')}</strong>
                  <small>{new Date(m.created_at * 1000).toLocaleString()}</small>
                  <button type="button" class="btn btn-ghost btn-sm" onclick={() => deleteMem(m.id)}>×</button>
                </div>
                <p>{m.summary}</p>
                {#if m.tags.length}
                  <div class="mem-tags">{#each m.tags as tag}<span>#{tag}</span>{/each}</div>
                {/if}
              </div>
            {/each}
          {/if}
        </div>
      </div>
    {/if}
  </section>
{/if}

<style>
  .studio-drawer {
    border-bottom: 1px solid var(--color-border);
    background: var(--color-surface);
    padding: 12px 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-height: 52vh;
    overflow: auto;
  }
  .studio-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
  .studio-tabs { display: flex; gap: 6px; flex-wrap: wrap; }
  .studio-tabs button {
    border: 1px solid var(--color-border);
    background: var(--color-surface-muted);
    color: var(--color-ink-subtle);
    padding: 5px 10px;
    font-size: 12px;
    border-radius: var(--radius-md);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .studio-tabs button.active {
    background: var(--color-brand-light);
    color: var(--color-brand);
    border-color: var(--color-brand);
    font-weight: 600;
  }
  .studio-status { margin: 0; font-size: 12px; color: var(--color-brand); }
  .studio-body { display: grid; grid-template-columns: 240px 1fr; gap: 12px; }
  .agent-roster { display: flex; flex-direction: column; gap: 6px; max-height: 310px; overflow: auto; }
  .agent-chip, .agent-chip-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-surface-muted);
    text-align: left;
    cursor: pointer;
  }
  .agent-chip.selected, .agent-chip-wrap.selected {
    border-color: var(--color-brand);
    background: var(--color-brand-subtle);
  }
  .agent-chip-select { flex: 1; display: flex; align-items: center; gap: 8px; border: 0; background: transparent; text-align: left; cursor: pointer; padding: 0; color: inherit; min-width: 0; }
  .agent-chip strong, .agent-chip-select strong { display: block; font-size: 12px; }
  .agent-chip small, .agent-chip-select small { display: block; font-size: 10.5px; color: var(--color-ink-subtle); }
  .agent-editor { display: flex; flex-direction: column; gap: 10px; }
  .editor-top { display: flex; gap: 12px; align-items: stretch; }
  .pet-preview-frame {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 8px 10px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-surface-muted);
    flex-shrink: 0;
  }
  .editor-fields { flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 8px; }
  .row-two { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; align-items: stretch; }
  .species-row, .skill-bind-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; font-size: 11.5px; }
  .pill-btn {
    border: 1px solid var(--color-border);
    background: var(--color-surface-muted);
    color: var(--color-ink-subtle);
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11px;
    cursor: pointer;
  }
  .pill-btn.active { border-color: var(--color-brand); color: var(--color-brand); background: var(--color-brand-light); font-weight: 600; }
  .setting-card, .skill-bind-box {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 9px 11px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-card);
    background: var(--color-surface-muted);
  }
  .field-title { font-size: 11px; font-weight: 600; color: var(--color-ink-subtle); text-transform: uppercase; letter-spacing: 0.03em; }
  .setting-card select.input { height: 34px; width: 100%; }
  .routine-check { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; min-height: 34px; font-size: 12px; color: var(--color-ink); cursor: pointer; }
  .routine-check input[type="checkbox"] { width: 15px; height: 15px; margin: 0; accent-color: var(--color-brand); flex-shrink: 0; }
  .mins-input { width: 64px; height: 30px; padding: 2px 7px; text-align: center; }
  .routine-input { margin-top: 2px; }
  .editor-actions { display: flex; justify-content: flex-end; gap: 8px; flex-wrap: wrap; }

  .memory-pane { display: flex; flex-direction: column; gap: 8px; }
  .memory-toolbar, .memory-add-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .memory-toolbar .input, #new-memory-text { flex: 1; min-width: 180px; }
  .tags-input { width: 140px; }
  .memory-list { display: flex; flex-direction: column; gap: 6px; max-height: 240px; overflow: auto; }
  .mem-empty { margin: 0; font-size: 12px; color: var(--color-ink-subtle); }
  .mem-item { padding: 8px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-md); background: var(--color-surface-muted); font-size: 12px; }
  .mem-head { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
  .mem-head small { margin-left: auto; color: var(--color-ink-subtle); font-size: 10.5px; }
  .mem-kind { font-size: 10px; padding: 1px 6px; border-radius: 999px; text-transform: uppercase; font-weight: 600; background: var(--color-surface); border: 1px solid var(--color-border); }
  .mem-item p { margin: 0; line-height: 1.4; }
  .mem-tags { display: flex; gap: 6px; margin-top: 4px; font-size: 10.5px; color: var(--color-ink-subtle); }

  @media (max-width: 720px) {
    .studio-body { grid-template-columns: 1fr; }
    .row-two { grid-template-columns: 1fr; }
  }
</style>
