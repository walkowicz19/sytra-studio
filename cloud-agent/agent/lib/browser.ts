import { z } from "zod";
import { browserProgram } from "./browser-program.ts";

const selector = z.string().trim().min(1).max(240);
const action = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("click"), selector }),
  z.object({ kind: z.literal("fill"), selector, text: z.string().max(1024) }),
  z.object({ kind: z.literal("enter"), selector }),
]);

export const browserReadInput = z.object({
  url: z.url().max(1024),
  selector: selector.optional(),
});

export const browserActionInput = browserReadInput.extend({
  actions: z.array(action).min(1).max(6),
});

type BrowserTask = z.infer<typeof browserReadInput> & {
  actions: z.infer<typeof action>[];
};

function httpsUrl(raw: string): URL {
  const url = new URL(raw);
  if (
    url.protocol !== "https:" || url.port || url.username || url.password ||
    !/^(?:[a-z0-9-]+\.)+[a-z][a-z0-9-]+$/i.test(url.hostname) ||
    url.hostname.endsWith(".local")
  ) {
    throw new Error("Browser access requires a public HTTPS hostname.");
  }
  return url;
}

export function browserOrigins(config = process.env.SYTRA_CLOUD_BROWSER_ORIGINS ?? ""): string[] {
  return [...new Set(config.split(",").map(value => value.trim()).filter(Boolean).map(value => {
    const url = httpsUrl(value);
    if (url.pathname !== "/" || url.search || url.hash) throw new Error("Configure HTTPS origins without paths.");
    return url.origin;
  }))];
}

export async function runBrowser(task: BrowserTask, abortSignal?: AbortSignal) {
  const url = httpsUrl(task.url);
  const origins = browserOrigins();
  if (!origins.includes(url.origin)) throw new Error("This site is outside the configured browser origins.");
  abortSignal?.throwIfAborted();

  const { Sandbox } = await import("@vercel/sandbox");
  const sandbox = await Sandbox.create({
    persistent: false,
    resources: { vcpus: 2 },
    timeout: 180000,
    signal: abortSignal,
  });
  try {
    const workdir = "/tmp/sytra-browser";
    const browserEnv = { PLAYWRIGHT_BROWSERS_PATH: `${workdir}/browsers` };
    await sandbox.mkDir(workdir);
    const install = await sandbox.runCommand({
      cmd: "npm",
      args: ["install", "--no-audit", "--no-fund", "--ignore-scripts", "playwright-core@1.65.1"],
      cwd: workdir,
      signal: abortSignal,
    });
    if (install.exitCode !== 0) throw new Error("Playwright could not be installed in the remote sandbox.");
    const browserInstall = await sandbox.runCommand({
      cmd: "node",
      args: ["node_modules/playwright-core/cli.js", "install", "--with-deps", "--only-shell", "chromium"],
      cwd: workdir, env: browserEnv, sudo: true, signal: abortSignal,
    });
    if (browserInstall.exitCode !== 0) {
      const details = (await browserInstall.stderr()).trim().slice(-1000);
      throw new Error(`Chromium installation failed: ${details}`);
    }
    await sandbox.update({ networkPolicy: { allow: origins.map(origin => new URL(origin).hostname) } });
    await sandbox.writeFiles([
      { path: `${workdir}/browser.mjs`, content: Buffer.from(browserProgram) },
      { path: `${workdir}/task.json`, content: Buffer.from(JSON.stringify({ ...task, url: url.href, origins })) },
    ]);
    const result = await sandbox.runCommand({
      cmd: "node", args: ["browser.mjs"], cwd: workdir, env: browserEnv, signal: abortSignal,
    });
    if (result.exitCode !== 0) {
      const details = (await result.stderr()).trim().slice(-1000);
      throw new Error(`Remote browser failed: ${details || "Chromium exited without a result."}`);
    }
    return JSON.parse(await result.stdout()) as {
      url: string; title: string; text: string; actionsCompleted: number;
    };
  } finally {
    await sandbox.stop();
  }
}
