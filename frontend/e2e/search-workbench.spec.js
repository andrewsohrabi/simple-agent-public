import { expect, test } from '@playwright/test'

function watchBrowserErrors(page) {
  const errors = []
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text())
  })
  page.on('pageerror', error => {
    errors.push(error.message)
  })
  return errors
}

async function expectNoHorizontalOverflow(page) {
  const overflow = await page.evaluate(() => {
    const clientWidth = document.documentElement.clientWidth
    const offenders = Array.from(document.querySelectorAll('body *'))
      .map(element => {
        const rect = element.getBoundingClientRect()
        return {
          tag: element.tagName.toLowerCase(),
          className: typeof element.className === 'string' ? element.className : '',
          text: (element.textContent ?? '').replace(/\s+/g, ' ').trim().slice(0, 90),
          left: Math.round(rect.left),
          right: Math.round(rect.right),
          width: Math.round(rect.width),
        }
      })
      .filter(item => item.right > clientWidth + 1 || item.width > clientWidth + 1)
      .slice(0, 8)
    return {
      clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
      offenders,
    }
  })
  expect(overflow.scrollWidth, JSON.stringify(overflow.offenders, null, 2)).toBeLessThanOrEqual(
    overflow.clientWidth + 1,
  )
}

test('loads desktop index status and model configuration', async ({ page }) => {
  const errors = watchBrowserErrors(page)

  await page.goto('/?mode=local&limit=8')

  await expect(page.getByRole('heading', { name: 'Internal Search Workbench' })).toBeVisible()
  await expect(page.getByText('Index Ready').first()).toBeVisible()
  await page.getByRole('button', { name: /Debug \/ Dev Panel/i }).click()
  await expect(page.getByText('text-embedding-3-large')).toBeVisible()
  await expect(page.getByText('3072')).toBeVisible()
  await expect(page.getByText('IndexFlatIP')).toBeVisible()
  await expect(page.getByText('189')).toBeVisible()
  await expect(page.getByText('7778')).toBeVisible()
  await expectNoHorizontalOverflow(page)
  expect(errors).toEqual([])
})

test('runs a real search and renders citations from the backend', async ({ page }) => {
  const errors = watchBrowserErrors(page)

  await page.goto('/?mode=local&limit=8')
  await page.getByRole('textbox', { name: /prompt/i }).fill('Find BOM-055 Rev G')
  await page.getByRole('button', { name: 'Search' }).click()

  await expect(page.getByRole('heading', { name: 'Findings' })).toBeVisible()
  await expect(page.getByText('BOM-055 Rev G').first()).toBeVisible()
  await page.getByRole('tab', { name: 'Sources' }).click()
  await expect(page.locator('.source-item').first()).toContainText('BOM-055')
  await page.getByRole('button', { name: /Debug \/ Dev Panel/i }).click()
  await expect(page.locator('.debug-pre')).toContainText('"category": "known_item"')
  await expectNoHorizontalOverflow(page)
  expect(errors).toEqual([])
})

test('runs an enumeration example in the desktop workbench', async ({ page }) => {
  const errors = watchBrowserErrors(page)

  await page.goto('/?mode=local&limit=6')
  await page.getByRole('button', { name: 'How many engineering change requests are in the system?' }).click()
  await expect(page.getByRole('textbox', { name: /prompt/i })).toHaveValue(
    'How many engineering change requests are in the system?',
  )
  await page.getByRole('button', { name: 'Search' }).click()

  await expect(page.getByText('Count:')).toBeVisible()
  await page.getByRole('tab', { name: 'Sources' }).click()
  await expect(page.locator('.source-item').first()).toContainText('ECR')
  await expectNoHorizontalOverflow(page)
  expect(errors).toEqual([])
})
