# Chat & inference model preferences

Global defaults (`default_context_window`, `default_temperature`, `enable_flash_attention`,
`kv_cache_quant`, `vram_limit_mb`, `cpu_kv_cache`) live in `.sytra-settings.json` and can be
configured in **Settings -> Inference Defaults**. Per-model overrides in **Models** override
the global defaults for that checkpoint; empty fields inherit the global setting. Save changes
and restart the model server for updates to take effect.

`temperature` values range from `0` to `2` (`0` requests deterministic sampling where supported
by the backend):

- The Sytra-managed local proxy (`127.0.0.1:8080`) exposes the active `temperature` at
  `GET /sytra/preferences` for the built-in chat.
- For external OpenAI- or Anthropic-compatible API clients calling `/v1/chat/completions`,
  `/v1/completions`, or `/v1/messages` with `application/json`, the proxy automatically fills
  in the active `temperature` when `"temperature"` is omitted from the request body, while
  preserving any explicit `"temperature"` sent by the client.
- If the built-in chat connects to an unmanaged external server without `/sytra/preferences`,
  it leaves sampling temperature to that server's default.
