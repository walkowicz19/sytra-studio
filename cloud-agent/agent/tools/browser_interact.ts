import { defineTool } from "eve/tools";
import { always } from "eve/tools/approval";
import { browserActionInput, runBrowser } from "../lib/browser.ts";

export default defineTool({
  description: "After approval, open an allowed HTTPS page and click or type into elements in one remote Chromium session. Network writes are blocked.",
  inputSchema: browserActionInput,
  approval: always(),
  execute: (input, ctx) => runBrowser(input, ctx.abortSignal),
});
