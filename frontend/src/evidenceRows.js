export function buildEvidenceRows(result = {}) {
  const citations = Array.isArray(result.citations) ? result.citations : []
  const retrieved = Array.isArray(result.retrievedDocuments) ? result.retrievedDocuments : []
  const rows = citations.map(citation => {
    const evidence = retrieved.find(item =>
      item.doc_id === citation.doc_id
      && item.revision === citation.revision
      && (!citation.chunk_id || item.chunk_id === citation.chunk_id),
    )
    return toEvidenceRow(citation, evidence)
  })
  const citationKeys = new Set(rows.map(row => row.key))
  retrieved.forEach(item => {
    const row = toEvidenceRow(null, item)
    if (!citationKeys.has(row.key)) {
      rows.push(row)
      citationKeys.add(row.key)
    }
  })
  return rows.map((row, index) => ({ ...row, index: index + 1 }))
}

function toEvidenceRow(citation, evidence) {
  const metadata = evidence?.metadata ?? {}
  const docId = citation?.doc_id ?? evidence?.doc_id ?? 'Unknown'
  const revision = citation?.revision ?? evidence?.revision ?? 'Unknown'
  const section = citation?.section ?? evidence?.section ?? 'Unknown section'
  const chunkId = citation?.chunk_id ?? evidence?.chunk_id ?? null
  const key = `${docId}-${revision}-${section}-${chunkId ?? 'metadata'}`
  const flags = []
  const evidenceType = citation?.evidence_type ?? evidence?.evidence_type ?? metadata.evidence_type ?? 'metadata'
  const supportLevel = citation?.support_level ?? evidence?.support_level ?? metadata.support_level ?? 'document'
  const tableIndex = citation?.table_index ?? evidence?.table_index ?? metadata.table_index ?? null
  const rowStart = citation?.row_start ?? evidence?.row_start ?? metadata.row_start ?? null
  const rowEnd = citation?.row_end ?? evidence?.row_end ?? metadata.row_end ?? null
  const columns = arrayValue(citation?.columns ?? evidence?.columns ?? metadata.columns)
  const rowCells = objectValue(citation?.row_cells ?? evidence?.row_cells ?? metadata.row_cells)
  if (metadata.is_latest) flags.push('latest')
  if (metadata.is_obsolete) flags.push('obsolete')
  if (section === 'metadata_inventory') flags.push('metadata')
  if (evidenceType === 'table_row') flags.push('table row')
  if (!citation) flags.push('uncited')
  if (evidence?.source === 'reference_follow') flags.push('reference follow')
  return {
    key,
    docId,
    revision,
    title: citation?.title ?? evidence?.title ?? 'Untitled source',
    section,
    filename: citation?.filename ?? metadata.filename ?? 'Not available',
    markdownPath: citation?.markdown_path ?? metadata.markdown_path ?? 'Not available',
    chunkId: chunkId ?? 'Metadata inventory',
    score: evidence?.score,
    source: evidence?.source ?? 'citation',
    evidenceType,
    supportLevel,
    tableIndex,
    rowStart,
    rowEnd,
    columns,
    rowCells,
    evidenceLabel: evidenceLabel(evidenceType, supportLevel, tableIndex, rowStart, rowEnd),
    flags,
    referenceText: metadata.reference_text,
  }
}

function evidenceLabel(type, support, tableIndex, rowStart, rowEnd) {
  const parts = [`${type} / ${support}`]
  if (tableIndex !== null && tableIndex !== undefined && tableIndex !== '') parts.push(`Table ${tableIndex}`)
  if (rowStart !== null && rowStart !== undefined && rowStart !== '') {
    let row = `Row ${rowStart}`
    if (rowEnd !== null && rowEnd !== undefined && rowEnd !== '' && rowEnd !== rowStart) row += `-${rowEnd}`
    parts.push(row)
  }
  return parts.join(' -> ')
}

function arrayValue(value) {
  return Array.isArray(value) ? value.map(item => String(item)) : []
}

function objectValue(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {}
}
