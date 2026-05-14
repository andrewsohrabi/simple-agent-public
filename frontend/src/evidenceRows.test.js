import assert from 'node:assert/strict'
import test from 'node:test'

import { buildEvidenceRows } from './evidenceRows.js'

test('buildEvidenceRows deduplicates citation and retrieved rows for the same evidence', () => {
  const rows = buildEvidenceRows({
    citations: [
      {
        doc_id: 'BOM-055',
        revision: 'G',
        section: 'Document Metadata',
        chunk_id: 'BOM-055:G:metadata:0',
        title: 'MX1 Bill of Materials',
      },
    ],
    retrievedDocuments: [
      {
        doc_id: 'BOM-055',
        revision: 'G',
        section: 'Document Metadata',
        chunk_id: 'BOM-055:G:metadata:0',
        title: 'MX1 Bill of Materials',
        metadata: { is_latest: true },
      },
    ],
  })

  assert.equal(rows.length, 1)
  assert.equal(rows[0].key, 'BOM-055-G-Document Metadata-BOM-055:G:metadata:0')
  assert.equal(rows[0].index, 1)
})
