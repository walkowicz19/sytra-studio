# Sytra Cloud Agent

Deploy this companion project separately from Sytra Studio. Vercel runs the
agent, model calls, and isolated Chromium sessions independently of the
desktop. Each navigation uses a new browser with a clean profile. Manage the
cloud agent through the authenticated Eve client.

## Browser access

Set `SYTRA_CLOUD_BROWSER_ORIGINS` in the project environment to a comma-separated
list of exact HTTPS origins, such as `https://demo.playwright.dev`. The browser
restricts network requests to those origins using GET or HEAD. The agent can
read a page and, after per-call approval, fill inputs or click controls.

Use Node 24 for the deployed project. With the project already created, run:

```sh
npm ci
npm run typecheck
npm exec -- eve link --non-interactive --project sytra-cloud-agent
npm exec -- eve deploy --non-interactive --yes
npm exec -- eve remote connect --url https://YOUR-AGENT.vercel.app
```

Vercel OIDC authenticates the client and managed compute. The scaffold uses AI
Gateway for inference. Enable Gateway on the linked project and complete any
billing setup it requests. The tested Vercel team required a payment card on
file before Gateway would accept a model request. See the
[Eve deployment guide](https://eve.dev/docs/guides/deployment/vercel).

## Test the browser

After linking, use the project's development environment to run
`scripts/browser-smoke.ts` with Node. For this test set
`SYTRA_CLOUD_BROWSER_ORIGINS=https://demo.playwright.dev`. The test adds and checks
an item in the public Playwright TodoMVC demonstration with Chromium running in
the Vercel Sandbox.

```powershell
$env:SYTRA_CLOUD_BROWSER_ORIGINS = 'https://demo.playwright.dev'
node --env-file=.env.local scripts/browser-smoke.ts
```

## Enable a recurring check

Copy `examples/site-check.ts.example` to `agent/schedules/site-check.ts`, set
`SYTRA_CLOUD_ROUTINE_URL` to a page on an allowed origin, and deploy the cloud
project. The example reads the page at 09:00 in São Paulo and records the page
title and URL in the cloud logs. The schedule lives on Vercel. See
[Eve schedules](https://eve.dev/docs/schedules).
