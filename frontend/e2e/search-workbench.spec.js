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

const corsHeaders = {
  'Access-Control-Allow-Headers': 'content-type',
  'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  'Access-Control-Allow-Origin': '*',
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

test('separates selected mode from reported retrieval backend', async ({ page }) => {
  const errors = watchBrowserErrors(page)

  await page.route('**/stats', async route => {
    await route.fulfill({
      headers: corsHeaders,
      json: {
        corpus_zip_exists: true,
        sqlite: { documents: 2, chunks: 4, references: 1 },
        vector_index: { vectors: 4 },
      },
    })
  })
  await page.route('**/search', async route => {
    if (route.request().method() === 'OPTIONS') {
      await route.fulfill({ status: 204, headers: corsHeaders })
      return
    }
    expect(route.request().postDataJSON().mode).toBe('hybrid')
    await route.fulfill({
      headers: corsHeaders,
      json: {
        answer: 'Mock hybrid answer from source-backed evidence.',
        citations: [
          {
            doc_id: 'BOM-055',
            revision: 'G',
            title: 'Bill of Materials',
            section: 'metadata_inventory',
            filename: 'BOM-055.docx',
            markdown_path: null,
            chunk_id: null,
          },
        ],
        retrieved_documents: [
          {
            doc_id: 'BOM-055',
            revision: 'G',
            title: 'Bill of Materials',
            section: 'metadata_inventory',
            score: 0.98,
            source: 'sqlite',
            metadata: { filename: 'BOM-055.docx', is_latest: true, is_obsolete: false },
          },
        ],
        query_plan: { strategy: 'hybrid', category: 'known_item' },
        mode: 'hybrid',
        retrieval_backend: 'local_hybrid',
        warnings: ['reranker_backend:deterministic_fallback'],
      },
    })
  })

  await page.goto('/?mode=hybrid&limit=8')
  await expect(page.getByText(/Strict local hybrid \(SQLite FTS \+ FAISS \+ reranker\)/)).toBeVisible()

  await page.getByRole('textbox', { name: /prompt/i }).fill('Find BOM-055 Rev G')
  await page.getByRole('button', { name: 'Search' }).click()

  const runDetails = page.getByRole('group', { name: 'Retrieval run details' })
  await expect(runDetails).toContainText('Selected Mode')
  await expect(runDetails).toContainText('Hybrid')
  await expect(runDetails).toContainText('Retrieval Backend')
  await expect(runDetails).toContainText('local_hybrid')
  await expect(page.getByText('reranker_backend:deterministic_fallback')).toBeVisible()

  await page.getByRole('tab', { name: 'Trace' }).click()
  await expect(page.getByRole('group', { name: 'Retrieval debug contract' })).toContainText('local_hybrid')

  await page.getByRole('button', { name: /Debug \/ Dev Panel/i }).click()
  await expect(page.locator('.dev-panel-body')).toContainText('Selected Mode')
  await expect(page.locator('.dev-panel-body')).toContainText('Retrieval Backend')
  await expect(page.locator('.dev-panel-body')).toContainText('local_hybrid')

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
