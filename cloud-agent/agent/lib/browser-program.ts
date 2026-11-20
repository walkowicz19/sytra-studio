// This fixed program runs inside the remote sandbox. The tool passes data in a JSON file.
export const browserProgram = String.raw`
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';

const task = JSON.parse(await readFile('task.json', 'utf8'));
const allowed = new Set(task.origins);
const browser = await chromium.launch({ headless: true });

try {
  const context = await browser.newContext({
    acceptDownloads: false,
    serviceWorkers: 'block',
  });
  await context.route('**/*', async route => {
    const request = route.request();
    const target = new URL(request.url());
    if (!allowed.has(target.origin) || !['GET', 'HEAD'].includes(request.method())) {
      await route.abort();
      return;
    }
    await route.continue();
  });
  const page = await context.newPage();
  page.setDefaultTimeout(5000);
  const response = await page.goto(task.url, { waitUntil: 'domcontentloaded', timeout: 12000 });
  if (!response?.ok()) throw new Error('The page returned HTTP ' + (response?.status() ?? 0));

  const challenge = () => page.locator(
    'iframe[src*="recaptcha"], iframe[src*="hcaptcha"], iframe[src*="challenges.cloudflare.com"], #challenge-form',
  ).count();
  const checkPage = async () => {
    if (!allowed.has(new URL(page.url()).origin)) throw new Error('Page left the configured origins');
    if (await challenge()) throw new Error('The site presented a verification challenge');
  };
  await checkPage();

  for (const action of task.actions) {
    const element = page.locator(action.selector);
    if (action.kind === 'fill') await element.fill(action.text);
    if (action.kind === 'click') await element.click();
    if (action.kind === 'enter') await element.press('Enter');
    await checkPage();
  }

  const content = page.locator(task.selector || 'body');
  console.log(JSON.stringify({
    url: page.url(),
    title: (await page.title()).slice(0, 240),
    text: (await content.innerText()).trim().slice(0, 4000),
    actionsCompleted: task.actions.length,
  }));
} finally {
  await browser.close();
}
`;
