import { api } from '../../api'
import { run, resetRun, pushToast, watchTelemetry } from '../../store.svelte'

export async function startWatchedOp(
  kind: 'train' | 'merge',
  start: () => Promise<string>,
): Promise<void> {
  resetRun()
  const opId = await start()
  run.opId = opId
  run.kind = kind
  run.status = 'running'
  run.startedAt = Date.now()
  if ('__TAURI_INTERNALS__' in window) {
    watchTelemetry(opId)
  }
  pushToast('success', `${kind === 'train' ? 'Training' : 'Merge'} started — ${opId.slice(0, 8)}`)
}
