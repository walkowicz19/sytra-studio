import { expect, test } from '@playwright/test'

test('renders v1 UI and shared resource controls without browser errors', async ({ page }) => {
  const errors: string[] = []
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text())
  })
  page.on('pageerror', error => errors.push(error.message))

  await page.goto('/')

  await expect(page.getByText('STUDIO', { exact: true })).toBeVisible()
  await expect(page.getByText('STUDIO — 01', { exact: true })).toHaveCount(0)
  await expect(page.getByText('v1.0.0', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Choose where models are downloaded' })).toBeVisible()

  const memory = page.getByLabel('Main memory limit')
  await expect(memory).toBeVisible()
  await expect(memory).toHaveValue('auto')
  const seventyFiveValue = await memory.locator('option').filter({ hasText: '75%' }).getAttribute('value')
  expect(seventyFiveValue).not.toBeNull()
  await memory.selectOption(seventyFiveValue!)
  await expect(page.getByText('Main memory limit saved')).toBeVisible()

  await page.getByRole('button', { name: /Combine/ }).click()
  await expect(page.getByRole('heading', { name: 'Combine models' })).toBeVisible()
  await page.getByRole('button', { name: /History/ }).click()
  await expect(page.getByRole('heading', { name: /history/i })).toBeVisible()

  expect(errors).toEqual([])
})
