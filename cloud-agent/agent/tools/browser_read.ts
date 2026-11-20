import { defineTool } from "eve/tools";
import { browserReadInput, runBrowser } from "../lib/browser.ts";

export default defineTool({
  description: "Read the title and visible text of an allowed HTTPS page in a remote Chromium browser.",
  inputSchema: browserReadInput,
  execute: (input, ctx) => runBrowser({ ...input, actions: [] }, ctx.abortSignal),
});
