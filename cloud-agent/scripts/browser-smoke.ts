import assert from "node:assert/strict";
import { browserActionInput, runBrowser } from "../agent/lib/browser.ts";

const task = browserActionInput.parse({
  url: "https://demo.playwright.dev/todomvc/",
  selector: ".todoapp",
  actions: [
    { kind: "fill", selector: ".new-todo", text: "Sytra remote smoke" },
    { kind: "enter", selector: ".new-todo" },
    { kind: "click", selector: ".todo-list .toggle" },
  ],
});

const result = await runBrowser(task);
assert.equal(result.actionsCompleted, 3);
assert.match(result.text, /Sytra remote smoke/);
assert.match(result.text, /0 items left/);
console.log(result);
