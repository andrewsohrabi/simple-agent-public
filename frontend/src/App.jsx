import { useEffect, useMemo, useRef, useState } from 'react'
import { PromptInputBox } from '@/components/ui/ai-prompt-box'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const MODES = [
  { id: 'auto', label: 'Auto', detail: 'Hosted search when available, local fallback otherwise.' },
  { id: 'hosted', label: 'Hosted', detail: 'Prefer OpenAI File Search over normalized Markdown.' },
  { id: 'local', label: 'Local', detail: 'Use SQLite FTS and FAISS local retrieval.' },
  { id: 'hybrid', label: 'Hybrid', detail: 'Blend lexical, dense, and reranked candidates.' },
]

const TABS = [
  { id: 'results', label: 'Results' },
  { id: 'sources', label: 'Sources' },
  { id: 'trace', label: 'Trace' },
  { id: 'inventory', label: 'Inventory' },
]

const EXAMPLES = [
  'Find the Bill of Materials for the MX1 system',
  'Where is the 510(k) summary for the device?',
  'Which verification protocols trace back to the risk analysis?',
  'What changed between Rev C and Rev D of the risk analysis?',
  'Trace the requirement for electrical leakage testing from the risk file through to the verification report',
  'How many engineering change requests are in the system?',
]

const INITIAL_RESULT = {
  status: 'idle',
  answer: '',
  citations: [],
  retrievedDocuments: [],
  queryPlan: null,
  warnings: [],
  lastQuery: '',
  latencyMs: null,
  mode: 'auto',
}

export default function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState(() => getInitialValue('mode', 'auto', value => MODES.some(item => item.id === value)))
  const [limit, setLimit] = useState(() => getInitialLimit())
  const [activeTab, setActiveTab] = useState(() => getInitialValue('tab', 'results', value => TABS.some(item => item.id === value)))
  const [inventoryFilter, setInventoryFilter] = useState(() => getInitialValue('filter', 'all'))
  const [selectedSourceKey, setSelectedSourceKey] = useState(null)
  const [stats, setStats] = useState(null)
  const [statsStatus, setStatsStatus] = useState('loading')
  const [statsError, setStatsError] = useState('')
  const [result, setResult] = useState(INITIAL_RESULT)
  const mainRef = useRef(null)

  const isLoading = result.status === 'loading'
  const selectedMode = useMemo(() => MODES.find(item => item.id === mode) ?? MODES[0], [mode])
  const evidenceRows = useMemo(() => buildEvidenceRows(result), [result])
  const selectedSource = useMemo(
    () => evidenceRows.find(row => row.key === selectedSourceKey) ?? null,
    [evidenceRows, selectedSourceKey],
  )
  const health = useMemo(() => getIndexHealth(stats, statsStatus, statsError), [stats, statsStatus, statsError])

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    params.set('mode', mode)
    params.set('limit', String(limit))
    params.set('tab', activeTab)
    params.set('filter', inventoryFilter)
    window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}`)
  }, [activeTab, inventoryFilter, limit, mode])

  useEffect(() => {
    let cancelled = false
    setStatsStatus('loading')
    fetch(`${API_URL}/stats`)
      .then(async res => {
        const data = await readResponse(res)
        if (!res.ok) throw new Error(errorMessage(data, res))
        return data
      })
      .then(data => {
        if (cancelled) return
        setStats(Array.isArray(data) ? null : data)
        setStatsStatus('ready')
      })
      .catch(error => {
        if (cancelled) return
        setStats(null)
        setStatsStatus('error')
        setStatsError(error instanceof Error ? error.message : 'Stats unavailable')
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (selectedSourceKey && !evidenceRows.some(row => row.key === selectedSourceKey)) {
      setSelectedSourceKey(null)
    }
  }, [evidenceRows, selectedSourceKey])

  async function runSearch(message) {
    const trimmed = message.trim()
    if (!trimmed || isLoading) return

    const startedAt = performance.now()
    setActiveTab('results')
    setSelectedSourceKey(null)
    setResult({ ...INITIAL_RESULT, status: 'loading', lastQuery: trimmed, mode })

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmed, mode, limit }),
      })
      const data = await readResponse(response)
      if (!response.ok) throw new Error(errorMessage(data, response))
      setResult(normalizeResult(data, trimmed, performance.now() - startedAt))
    } catch (error) {
      setResult({
        ...INITIAL_RESULT,
        status: 'error',
        lastQuery: trimmed,
        mode,
        warnings: [error instanceof Error ? error.message : 'Search failed'],
      })
    }
  }

  function openSource(row) {
    setSelectedSourceKey(row.key)
    setActiveTab('sources')
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workbench-main">Skip to search</a>
      <header className="workbench-header">
        <div>
          <p className="eyebrow">MedAI QMS</p>
          <h1>Internal Search Workbench</h1>
        </div>
        <div className="header-actions" aria-label="Workbench status">
          <StatusBadge status={result.status} />
          <span className={`health-badge health-badge-${health.tone}`}>{health.label}</span>
        </div>
      </header>

      <main id="workbench-main" ref={mainRef} className="workbench-main" tabIndex="-1">
        <section className="search-console" aria-labelledby="composer-heading">
          <div className="console-topline">
            <div>
              <p className="eyebrow">Search Prompt</p>
              <h2 id="composer-heading">Ask the controlled corpus</h2>
            </div>
            <div className="console-metrics" aria-live="polite">
              <Metric label="Mode" value={selectedMode.label} />
              <Metric label="Depth" value={String(limit)} />
              <Metric label="Latency" value={formatLatency(result.latencyMs)} />
            </div>
          </div>

          <PromptInputBox
            value={query}
            onValueChange={setQuery}
            onSend={runSearch}
            isLoading={isLoading}
            placeholder="Find the current BOM for the MX1 system..."
            submitLabel="Search"
            leftActions={
              <>
                <span className="composer-pill">{selectedMode.detail}</span>
                <span className="composer-pill">{health.shortLabel}</span>
              </>
            }
          />

          <div className="control-bar" aria-label="Search controls">
            <fieldset className="segmented-field">
              <legend>Retrieval Mode</legend>
              <div className="segmented-control">
                {MODES.map(item => (
                  <label key={item.id}>
                    <input
                      type="radio"
                      name="search-mode"
                      value={item.id}
                      checked={mode === item.id}
                      onChange={() => setMode(item.id)}
                    />
                    <span>{item.label}</span>
                  </label>
                ))}
              </div>
            </fieldset>

            <label className="range-control" htmlFor="retrieval-depth">
              <span>Retrieval Depth</span>
              <input
                id="retrieval-depth"
                type="range"
                min="3"
                max="20"
                step="1"
                value={limit}
                onChange={event => setLimit(Number(event.target.value))}
                aria-valuetext={`${limit} sources`}
              />
              <strong>{limit}</strong>
            </label>
          </div>

          <div className="example-strip" aria-label="Example searches">
            {EXAMPLES.map(example => (
              <button key={example} type="button" onClick={() => setQuery(example)}>
                {example}
              </button>
            ))}
          </div>
        </section>

        <IndexStateBanner health={health} />

        <section className="workbench-surface" aria-label="Search workbench">
          <div className="tabs-row" role="tablist" aria-label="Workbench views">
            {TABS.map(tab => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                id={`tab-${tab.id}`}
                aria-selected={activeTab === tab.id}
                aria-controls={`panel-${tab.id}`}
                className={activeTab === tab.id ? 'active' : ''}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="tab-layout">
            <div className="tab-panel-shell">
              {activeTab === 'results' && (
                <ResultsTab result={result} evidenceRows={evidenceRows} onRetry={() => runSearch(result.lastQuery)} onOpenSource={openSource} />
              )}
              {activeTab === 'sources' && (
                <SourcesTab result={result} evidenceRows={evidenceRows} selectedSourceKey={selectedSourceKey} onSelectSource={setSelectedSourceKey} />
              )}
              {activeTab === 'trace' && <TraceTab result={result} />}
              {activeTab === 'inventory' && (
                <InventoryTab
                  stats={stats}
                  result={result}
                  rows={evidenceRows}
                  filter={inventoryFilter}
                  onFilterChange={setInventoryFilter}
                />
              )}
            </div>

            <SourceDrawer source={selectedSource} result={result} onClose={() => setSelectedSourceKey(null)} />
          </div>
        </section>

        <DevPanel stats={stats} statsStatus={statsStatus} statsError={statsError} result={result} apiUrl={API_URL} />
      </main>
    </div>
  )
}

function ResultsTab({ result, evidenceRows, onRetry, onOpenSource }) {
  return (
    <section id="panel-results" role="tabpanel" aria-labelledby="tab-results" className="tab-panel" tabIndex="0">
      <PanelHeading eyebrow="Results" title="Findings" status={result.status} />
      {result.status === 'idle' && <EmptyState title="No Search Run Yet" copy="Select an example or enter a QMS question to start." />}
      {result.status === 'loading' && <LoadingState />}
      {result.status === 'error' && <ErrorState warnings={result.warnings} onRetry={onRetry} />}
      {result.status === 'success' && (
        <div className="answer-content">
          <div className="query-recap">
            <span>Query</span>
            <p>{result.lastQuery}</p>
          </div>
          <EvidenceQuality result={result} evidenceRows={evidenceRows} />
          <FormattedAnswer text={result.answer} />
          {evidenceRows.length > 0 && (
            <div className="citation-chips" aria-label="Citations">
              {evidenceRows.slice(0, 8).map(row => (
                <button key={row.key} type="button" onClick={() => onOpenSource(row)}>
                  [{row.index}] {row.docId} Rev {row.revision}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

function SourcesTab({ result, evidenceRows, selectedSourceKey, onSelectSource }) {
  return (
    <section id="panel-sources" role="tabpanel" aria-labelledby="tab-sources" className="tab-panel" tabIndex="0">
      <PanelHeading eyebrow="Sources" title="Evidence Drawer" status={evidenceRows.length ? `${evidenceRows.length} items` : result.status} />
      {result.status === 'idle' && <CompactState copy="Sources appear after a search." />}
      {result.status === 'loading' && <CompactState copy="Preparing citations..." loading />}
      {result.status === 'success' && evidenceRows.length === 0 && (
        <WeakEvidenceState copy="No citations or retrieved documents were returned. Treat this answer as unsupported." />
      )}
      {evidenceRows.length > 0 && (
        <ol className="source-list">
          {evidenceRows.map(row => (
            <li key={row.key} className={selectedSourceKey === row.key ? 'source-item selected' : 'source-item'}>
              <button type="button" className="source-card-button" onClick={() => onSelectSource(row.key)}>
                <span className="source-index">{row.index}</span>
                <span>
                  <strong>{row.docId} Rev {row.revision}</strong>
                  <small>{row.title}</small>
                </span>
                <span className="source-score">{formatScore(row.score)}</span>
              </button>
              <div className="source-meta">
                <span>{row.section}</span>
                <span>{row.filename}</span>
                <span>{row.source}</span>
              </div>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}

function TraceTab({ result }) {
  const planEntries = objectEntries(result.queryPlan)
  const warningRows = Array.isArray(result.warnings) ? result.warnings : []
  return (
    <section id="panel-trace" role="tabpanel" aria-labelledby="tab-trace" className="tab-panel" tabIndex="0">
      <PanelHeading eyebrow="Trace" title="Routing & Fallbacks" status={result.queryPlan?.strategy ?? result.status} />
      {result.status === 'idle' && <CompactState copy="Query planning and fallback traces appear after a search." />}
      {result.status === 'loading' && <CompactState copy="Building query plan..." loading />}
      {result.status !== 'idle' && result.status !== 'loading' && (
        <div className="trace-grid">
          <div className="trace-card">
            <h3>Query Plan</h3>
            {planEntries.length > 0 ? (
              <dl className="key-value-list">
                {planEntries.map(([key, value]) => <Detail key={key} label={key} value={formatValue(value)} />)}
              </dl>
            ) : (
              <CompactState copy="No query plan returned by the backend." />
            )}
          </div>
          <div className="trace-card">
            <h3>Warnings & Fallbacks</h3>
            {warningRows.length > 0 ? (
              <ul className="warning-list">
                {warningRows.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}
              </ul>
            ) : (
              <CompactState copy="No fallback or citation warnings were returned." />
            )}
          </div>
        </div>
      )}
    </section>
  )
}

function InventoryTab({ stats, result, rows, filter, onFilterChange }) {
  const filteredRows = filterRows(rows, filter)
  const sqlite = stats?.sqlite ?? {}
  const vector = stats?.vector_index ?? {}
  return (
    <section id="panel-inventory" role="tabpanel" aria-labelledby="tab-inventory" className="tab-panel" tabIndex="0">
      <PanelHeading eyebrow="Inventory" title="Index & Retrieved Set" status={`${filteredRows.length}/${rows.length} shown`} />
      <div className="inventory-summary" aria-label="Index inventory summary">
        <Metric label="Docs" value={formatCount(sqlite.documents)} />
        <Metric label="Chunks" value={formatCount(sqlite.chunks)} />
        <Metric label="References" value={formatCount(sqlite.references)} />
        <Metric label="Vectors" value={formatCount(vector.vectors)} />
      </div>

      <label className="filter-control" htmlFor="inventory-filter">
        <span>Inventory Filter</span>
        <select id="inventory-filter" value={filter} onChange={event => onFilterChange(event.target.value)}>
          <option value="all">All retrieved records</option>
          <option value="latest">Latest only</option>
          <option value="obsolete">Obsolete</option>
          <option value="metadata">Metadata inventory</option>
          <option value="weak">Weak evidence</option>
        </select>
      </label>

      {result.status === 'idle' && <CompactState copy="Run a search to inspect the retrieved inventory slice." />}
      {result.status === 'loading' && <CompactState copy="Loading inventory slice..." loading />}
      {result.status === 'success' && rows.length === 0 && <WeakEvidenceState copy="The backend returned no retrieved inventory rows." />}
      {rows.length > 0 && filteredRows.length === 0 && <CompactState copy="No retrieved records match this filter." />}
      {filteredRows.length > 0 && (
        <div className="inventory-table-wrap">
          <table className="inventory-table">
            <thead>
              <tr>
                <th scope="col">Doc ID</th>
                <th scope="col">Rev</th>
                <th scope="col">Section</th>
                <th scope="col">Source</th>
                <th scope="col">Flags</th>
                <th scope="col">Score</th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.map(row => (
                <tr key={row.key}>
                  <td>{row.docId}</td>
                  <td>{row.revision}</td>
                  <td>{row.section}</td>
                  <td>{row.source}</td>
                  <td>{row.flags.join(', ') || 'none'}</td>
                  <td>{formatScore(row.score)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function SourceDrawer({ source, result, onClose }) {
  return (
    <aside className="source-drawer" aria-label="Source details">
      <div className="drawer-heading">
        <div>
          <p className="eyebrow">Evidence</p>
          <h2>Source Details</h2>
        </div>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close source details">x</button>
      </div>
      {!source && <CompactState copy={result.status === 'success' ? 'Select a source to inspect its citation metadata.' : 'Run a search to inspect source evidence.'} />}
      {source && (
        <div className="drawer-body">
          <div className="drawer-title">
            <span className="source-index">{source.index}</span>
            <div>
              <h3>{source.docId} Rev {source.revision}</h3>
              <p>{source.title}</p>
            </div>
          </div>
          <dl className="key-value-list">
            <Detail label="Section" value={source.section} />
            <Detail label="Filename" value={source.filename} />
            <Detail label="Markdown" value={source.markdownPath} />
            <Detail label="Chunk" value={source.chunkId} />
            <Detail label="Source" value={source.source} />
            <Detail label="Score" value={formatScore(source.score)} />
            <Detail label="Flags" value={source.flags.join(', ') || 'none'} />
          </dl>
          {source.referenceText && (
            <blockquote>
              <strong>Reference Followed</strong>
              <p>{source.referenceText}</p>
            </blockquote>
          )}
        </div>
      )}
    </aside>
  )
}

function DevPanel({ stats, statsStatus, statsError, result, apiUrl }) {
  const [open, setOpen] = useState(false)
  return (
    <section className="dev-panel" aria-labelledby="dev-panel-heading">
      <button type="button" className="dev-panel-toggle" onClick={() => setOpen(value => !value)} aria-expanded={open} aria-controls="dev-panel-body">
        <span>Debug / Dev Panel</span>
        <strong>{open ? 'Hide' : 'Show'}</strong>
      </button>
      {open && (
        <div id="dev-panel-body" className="dev-panel-body">
          <h2 id="dev-panel-heading">Runtime Contract</h2>
          <dl className="key-value-list">
            <Detail label="API URL" value={apiUrl} />
            <Detail label="Stats" value={statsStatus === 'error' ? statsError : statsStatus} />
            <Detail label="Result" value={result.status} />
            <Detail label="Mode" value={result.mode} />
          </dl>
          <pre className="debug-pre">{JSON.stringify({ stats, result }, null, 2)}</pre>
        </div>
      )}
    </section>
  )
}

function PanelHeading({ eyebrow, title, status }) {
  return (
    <div className="panel-heading">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h2>{title}</h2>
      </div>
      <StatusBadge status={status} />
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function StatusBadge({ status }) {
  const normalized = String(status ?? 'idle')
  return <span className={`status-badge status-badge-${statusTone(normalized)}`}>{toTitleCase(normalized)}</span>
}

function IndexStateBanner({ health }) {
  if (health.tone === 'success') return null
  return (
    <section className={`state-banner state-banner-${health.tone}`} aria-live="polite">
      <strong>{health.label}</strong>
      <span>{health.detail}</span>
    </section>
  )
}

function EmptyState({ title, copy }) {
  return (
    <div className="empty-state">
      <h3>{title}</h3>
      <p>{copy}</p>
    </div>
  )
}

function LoadingState() {
  return (
    <div className="loading-state">
      <div className="loading-copy">
        <span className="spinner" aria-hidden="true" />
        <span>Retrieving source-backed evidence...</span>
      </div>
      <AnswerSkeleton />
    </div>
  )
}

function ErrorState({ warnings, onRetry }) {
  return (
    <div className="error-state" role="alert">
      <h3>Backend Error</h3>
      <p>{warnings.join('; ') || 'Search failed.'}</p>
      <button type="button" className="secondary-button" onClick={onRetry}>Retry Search</button>
    </div>
  )
}

function CompactState({ copy, loading = false }) {
  return (
    <div className="compact-state">
      {loading && <span className="spinner" aria-hidden="true" />}
      <span>{copy}</span>
    </div>
  )
}

function WeakEvidenceState({ copy }) {
  return (
    <div className="weak-state" role="status">
      <strong>Weak Evidence</strong>
      <span>{copy}</span>
    </div>
  )
}

function EvidenceQuality({ result, evidenceRows }) {
  const warnings = Array.isArray(result.warnings) ? result.warnings : []
  if (warnings.includes('no_retrieval_hits')) {
    return <WeakEvidenceState copy="No retrieval hits were available, so the answer should not be used as factual evidence." />
  }
  if (evidenceRows.length === 0) {
    return <WeakEvidenceState copy="No citations were returned with this answer." />
  }
  if (warnings.length > 0) {
    return (
      <div className="inline-warning" role="status">
        {warnings.join('; ')}
      </div>
    )
  }
  return (
    <div className="inline-success" role="status">
      {evidenceRows.length} citation-backed source{evidenceRows.length === 1 ? '' : 's'} returned.
    </div>
  )
}

function AnswerSkeleton() {
  return (
    <div className="answer-skeleton" aria-hidden="true">
      <div className="skeleton-line wide" />
      <div className="skeleton-line wide" />
      <div className="skeleton-line" />
      <div className="skeleton-gap" />
      <div className="skeleton-line wide" />
      <div className="skeleton-line short" />
    </div>
  )
}

function FormattedAnswer({ text }) {
  const lines = String(text || '').split('\n').filter(Boolean)
  if (lines.length === 0) return <p className="answer-paragraph">The server returned an empty answer.</p>
  return (
    <div className="formatted-answer">
      {lines.map((line, index) =>
        line.trim().startsWith('- ') ? (
          <p key={index} className="answer-bullet">{line}</p>
        ) : (
          <p key={index} className="answer-paragraph">{line}</p>
        ),
      )}
    </div>
  )
}

function Detail({ label, value }) {
  return (
    <div>
      <dt>{toTitleCase(label)}</dt>
      <dd>{value === null || value === undefined || value === '' ? 'Not available' : String(value)}</dd>
    </div>
  )
}

async function readResponse(response) {
  const contentType = response.headers.get('content-type') ?? ''
  if (contentType.includes('application/json')) return response.json()
  return { answer: await response.text() }
}

function normalizeResult(data, query, latencyMs) {
  return {
    status: 'success',
    answer: String(data.answer ?? data.reply ?? ''),
    citations: Array.isArray(data.citations) ? data.citations : [],
    retrievedDocuments: Array.isArray(data.retrieved_documents) ? data.retrieved_documents : [],
    queryPlan: data.query_plan && !data.query_plan.error ? data.query_plan : data.query_plan ?? null,
    warnings: Array.isArray(data.warnings) ? data.warnings : [],
    mode: data.mode ?? 'auto',
    lastQuery: query,
    latencyMs,
  }
}

function buildEvidenceRows(result) {
  const citations = Array.isArray(result.citations) ? result.citations : []
  const retrieved = Array.isArray(result.retrievedDocuments) ? result.retrievedDocuments : []
  const rows = citations.map((citation, index) => {
    const evidence = retrieved.find(item =>
      item.doc_id === citation.doc_id
      && item.revision === citation.revision
      && (!citation.chunk_id || item.chunk_id === citation.chunk_id),
    )
    return toEvidenceRow(citation, evidence, index)
  })
  const citationKeys = new Set(rows.map(row => row.key))
  retrieved.forEach((item, index) => {
    const row = toEvidenceRow(null, item, rows.length + index)
    if (!citationKeys.has(row.key)) rows.push(row)
  })
  return rows.map((row, index) => ({ ...row, index: index + 1 }))
}

function toEvidenceRow(citation, evidence, index) {
  const metadata = evidence?.metadata ?? {}
  const docId = citation?.doc_id ?? evidence?.doc_id ?? 'Unknown'
  const revision = citation?.revision ?? evidence?.revision ?? 'Unknown'
  const section = citation?.section ?? evidence?.section ?? 'Unknown section'
  const chunkId = citation?.chunk_id ?? evidence?.chunk_id ?? null
  const key = `${docId}-${revision}-${section}-${chunkId ?? 'metadata'}-${index}`
  const flags = []
  if (metadata.is_latest) flags.push('latest')
  if (metadata.is_obsolete) flags.push('obsolete')
  if (section === 'metadata_inventory') flags.push('metadata')
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
    flags,
    referenceText: metadata.reference_text,
  }
}

function filterRows(rows, filter) {
  if (filter === 'all') return rows
  if (filter === 'latest') return rows.filter(row => row.flags.includes('latest'))
  if (filter === 'obsolete') return rows.filter(row => row.flags.includes('obsolete'))
  if (filter === 'metadata') return rows.filter(row => row.flags.includes('metadata'))
  if (filter === 'weak') return rows.filter(row => row.flags.includes('uncited') || typeof row.score !== 'number')
  return rows
}

function getIndexHealth(stats, status, error) {
  if (status === 'loading') {
    return { tone: 'loading', label: 'Loading Index Status', shortLabel: 'Index loading', detail: 'Checking local index and hosted state.' }
  }
  if (status === 'error') {
    return { tone: 'error', label: 'Stats Backend Error', shortLabel: 'Stats error', detail: error || 'The stats endpoint did not respond.' }
  }
  const sqliteDocs = Number(stats?.sqlite?.documents ?? 0)
  const sqliteChunks = Number(stats?.sqlite?.chunks ?? 0)
  const corpusExists = stats?.corpus_zip_exists !== false
  if (!corpusExists || sqliteDocs === 0 || sqliteChunks === 0) {
    return { tone: 'warning', label: 'Missing or Sparse Index', shortLabel: 'Index incomplete', detail: 'The local corpus/index appears missing or empty.' }
  }
  return { tone: 'success', label: 'Index Ready', shortLabel: 'Index ready', detail: `${sqliteDocs} documents and ${sqliteChunks} chunks are available.` }
}

function errorMessage(data, response) {
  if (typeof data.detail === 'string') return data.detail
  if (data.error) return String(data.error)
  return `${response.status} ${response.statusText || 'Request failed'}`
}

function getInitialValue(name, fallback, isValid = value => Boolean(value)) {
  const value = new URLSearchParams(window.location.search).get(name)
  return value && isValid(value) ? value : fallback
}

function getInitialLimit() {
  const value = Number(new URLSearchParams(window.location.search).get('limit'))
  return Number.isFinite(value) ? Math.min(20, Math.max(3, value)) : 16
}

function formatLatency(latencyMs) {
  if (latencyMs === null || latencyMs === undefined) return 'Not run'
  return `${Math.round(latencyMs).toLocaleString()} ms`
}

function formatScore(score) {
  if (typeof score !== 'number' || Number.isNaN(score)) return 'Not scored'
  return score.toFixed(4)
}

function formatCount(value) {
  return typeof value === 'number' ? value.toLocaleString() : 'Not indexed'
}

function formatValue(value) {
  if (Array.isArray(value)) return value.length ? value.join(', ') : 'none'
  if (value && typeof value === 'object') return JSON.stringify(value)
  return value ?? 'Not available'
}

function objectEntries(value) {
  return value && typeof value === 'object' ? Object.entries(value) : []
}

function statusTone(status) {
  if (status.includes('error')) return 'error'
  if (status.includes('loading')) return 'loading'
  if (status.includes('success') || /\d/.test(status)) return 'success'
  return 'idle'
}

function toTitleCase(value) {
  return String(value).replace(/[-_]/g, ' ').replace(/\b\w/g, letter => letter.toUpperCase())
}
