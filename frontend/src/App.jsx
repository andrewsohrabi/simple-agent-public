import { useEffect, useMemo, useState } from 'react'
import { PromptInputBox } from '@/components/ui/ai-prompt-box'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const MODES = [
  { id: 'auto', label: 'Auto', detail: 'Use hosted search when available, with local fallback.' },
  { id: 'hosted', label: 'Hosted', detail: 'Prefer OpenAI File Search over normalized Markdown.' },
  { id: 'local', label: 'Local', detail: 'Use SQLite FTS and FAISS local retrieval.' },
  { id: 'hybrid', label: 'Hybrid', detail: 'Blend lexical, dense, and reranked candidates.' },
]

const EXAMPLE_GROUPS = [
  {
    category: 'Known Item',
    examples: [
      'Find the Bill of Materials for the MX1 system',
      'Where is the 510(k) summary for the device?',
    ],
  },
  {
    category: 'Exploratory',
    examples: [
      'What verification test protocols do we have for the MX1?',
      'Show me all risk-related documents',
    ],
  },
  {
    category: 'Compliance',
    examples: [
      'Does our Design History File include everything required by FDA 21 CFR 820.30?',
      'Which verification protocols trace back to the risk analysis?',
    ],
  },
  {
    category: 'Extraction',
    examples: [
      'What are the acceptance criteria for the electrical safety verification test?',
      'Summarize all design review action items that are still open',
    ],
  },
  {
    category: 'Revisions',
    examples: [
      'What changed between Rev C and Rev D of the risk analysis?',
      'Show me all ECRs filed in the last year and their status',
    ],
  },
  {
    category: 'Cross Document',
    examples: [
      'Trace the requirement for electrical leakage testing from the risk file through to the verification report',
      'Map all third-party test reports to the regulatory requirements they satisfy',
    ],
  },
  {
    category: 'Counts',
    examples: [
      'How many engineering change requests are in the system?',
      'How many verification protocols have we completed vs. planned?',
    ],
  },
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
}

export default function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState(() => getInitialMode())
  const [limit, setLimit] = useState(() => getInitialLimit())
  const [stats, setStats] = useState(null)
  const [result, setResult] = useState(INITIAL_RESULT)
  const isLoading = result.status === 'loading'

  const selectedMode = useMemo(
    () => MODES.find(item => item.id === mode) ?? MODES[0],
    [mode],
  )

  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    params.set('mode', mode)
    params.set('limit', String(limit))
    window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}`)
  }, [mode, limit])

  useEffect(() => {
    fetch(`${API_URL}/stats`)
      .then(res => (res.ok ? res.json() : null))
      .then(data => setStats(data))
      .catch(() => setStats(null))
  }, [])

  async function runSearch(message) {
    const trimmed = message.trim()
    if (!trimmed || isLoading) return

    const startedAt = performance.now()
    setResult({ ...INITIAL_RESULT, status: 'loading', lastQuery: trimmed })

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
        warnings: [error instanceof Error ? error.message : 'Search failed'],
      })
    }
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workbench-main">Skip to search</a>
      <header className="workbench-header">
        <div>
          <p className="eyebrow">MedAI QMS</p>
          <h1>Internal Search Workbench</h1>
        </div>
        <div className="header-status" aria-live="polite">
          <span className="status-dot" aria-hidden="true" />
          <span>{stats ? 'Index status loaded' : `API ${API_URL}`}</span>
        </div>
      </header>

      <div className="workbench-grid">
        <aside className="left-rail" aria-label="Search setup">
          <ModeSelector mode={mode} onModeChange={setMode} />
          <ExampleGroups onUseExample={setQuery} />
        </aside>

        <main id="workbench-main" className="workbench-main" tabIndex="-1">
          <section className="composer-section" aria-labelledby="composer-heading">
            <div className="section-heading-row">
              <div>
                <p className="eyebrow">Search Prompt</p>
                <h2 id="composer-heading">Ask the controlled corpus</h2>
              </div>
              <span className="mode-chip">{selectedMode.label}</span>
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
                  <span className="composer-pill">{limit} sources</span>
                </>
              }
            />
          </section>

          <AnswerPanel result={result} onRetry={() => runSearch(result.lastQuery)} />
        </main>

        <aside className="right-rail" aria-label="Evidence and configuration">
          <SourcePanel result={result} />
          <StatsConfigPanel
            stats={stats}
            result={result}
            limit={limit}
            onLimitChange={setLimit}
          />
        </aside>
      </div>
    </div>
  )
}

function ModeSelector({ mode, onModeChange }) {
  return (
    <section className="panel" aria-labelledby="mode-heading">
      <div className="panel-heading">
        <p className="eyebrow">Mode</p>
        <h2 id="mode-heading">Retrieval</h2>
      </div>
      <div className="mode-list" role="radiogroup" aria-labelledby="mode-heading">
        {MODES.map(item => (
          <label key={item.id} className="mode-option">
            <input
              type="radio"
              name="search-mode"
              value={item.id}
              checked={mode === item.id}
              onChange={() => onModeChange(item.id)}
            />
            <span>
              <span className="mode-option-top">
                <strong>{item.label}</strong>
                <em>{item.id}</em>
              </span>
              <span className="mode-option-copy">{item.detail}</span>
            </span>
          </label>
        ))}
      </div>
    </section>
  )
}

function ExampleGroups({ onUseExample }) {
  return (
    <section className="panel examples-panel" aria-labelledby="examples-heading">
      <div className="panel-heading">
        <p className="eyebrow">Examples</p>
        <h2 id="examples-heading">Eval Patterns</h2>
      </div>
      <div className="example-groups">
        {EXAMPLE_GROUPS.map(group => (
          <section key={group.category} className="example-group">
            <h3>{group.category}</h3>
            <div className="example-list">
              {group.examples.map(example => (
                <button
                  key={example}
                  type="button"
                  className="example-button"
                  onClick={() => onUseExample(example)}
                >
                  <span>{example}</span>
                </button>
              ))}
            </div>
          </section>
        ))}
      </div>
    </section>
  )
}

function AnswerPanel({ result, onRetry }) {
  return (
    <section className="answer-panel panel" aria-labelledby="answer-heading" aria-live="polite">
      <div className="section-heading-row">
        <div>
          <p className="eyebrow">Answer</p>
          <h2 id="answer-heading">Findings</h2>
        </div>
        <StatusBadge status={result.status} />
      </div>

      {result.status === 'idle' && (
        <div className="empty-state">
          <h3>No search run yet</h3>
          <p>Select an example or enter a QMS question to start.</p>
        </div>
      )}
      {result.status === 'loading' && (
        <div className="loading-state">
          <div className="loading-copy">
            <span className="spinner" aria-hidden="true" />
            <span>Retrieving source-backed evidence</span>
          </div>
          <AnswerSkeleton />
        </div>
      )}
      {result.status === 'error' && (
        <div className="error-state" role="alert">
          <h3>Search failed</h3>
          <p>{result.warnings.join('; ')}</p>
          <button type="button" className="secondary-button" onClick={onRetry}>
            Retry Search
          </button>
        </div>
      )}
      {result.status === 'success' && (
        <div className="answer-content">
          <div className="query-recap">
            <span>Query</span>
            <p>{result.lastQuery}</p>
          </div>
          {result.warnings.length > 0 && (
            <div className="inline-warning" role="status">
              {result.warnings.join('; ')}
            </div>
          )}
          <FormattedAnswer text={result.answer} />
          {result.queryPlan && (
            <pre className="debug-pre">{JSON.stringify(result.queryPlan, null, 2)}</pre>
          )}
        </div>
      )}
    </section>
  )
}

function SourcePanel({ result }) {
  const citations = Array.isArray(result.citations) ? result.citations : []
  return (
    <section className="panel source-panel" aria-labelledby="sources-heading">
      <div className="panel-heading">
        <p className="eyebrow">Sources</p>
        <h2 id="sources-heading">Citations</h2>
      </div>
      {result.status === 'idle' && <div className="compact-empty">Sources appear after search.</div>}
      {result.status === 'loading' && <div className="source-loading">Preparing citations...</div>}
      {result.status === 'success' && citations.length === 0 && (
        <div className="partial-state">
          <strong>No citations returned</strong>
          <span>The answer is not acceptable for production without source metadata.</span>
        </div>
      )}
      {citations.length > 0 && (
        <ol className="source-list">
          {citations.map((citation, index) => (
            <li key={`${citation.doc_id}-${citation.revision}-${citation.section}-${index}`} className="source-item">
              <div className="source-title-row">
                <span className="source-index">{index + 1}</span>
                <div>
                  <h3>{citation.doc_id} Rev {citation.revision}</h3>
                  <p>{citation.title}</p>
                </div>
              </div>
              <div className="source-meta">
                <span>{citation.section}</span>
                <span>{citation.filename}</span>
              </div>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}

function StatsConfigPanel({ stats, result, limit, onLimitChange }) {
  const modelConfig = stats?.model_config ?? {}
  return (
    <section className="panel stats-panel" aria-labelledby="config-heading">
      <div className="panel-heading">
        <p className="eyebrow">Config</p>
        <h2 id="config-heading">Models & Index</h2>
      </div>
      <label className="field-label" htmlFor="retrieval-depth">
        Retrieval Depth
        <input
          id="retrieval-depth"
          type="range"
          min="3"
          max="20"
          step="1"
          value={limit}
          onChange={event => onLimitChange(Number(event.target.value))}
          aria-valuetext={`${limit} sources`}
        />
        <span className="range-value">{limit} sources</span>
      </label>
      <dl className="stats-list">
        <Stat label="Status" value={toTitleCase(result.status)} />
        <Stat label="Latency" value={formatLatency(result.latencyMs)} />
        <Stat label="Embedding" value={modelConfig.embedding_model ?? 'Unknown'} />
        <Stat label="Dimensions" value={modelConfig.embedding_dimensions ?? 'Unknown'} />
        <Stat label="FAISS" value={modelConfig.faiss_index_type ?? 'Unknown'} />
        <Stat label="Chat" value={modelConfig.chat_model ?? 'Unknown'} />
        <Stat label="Agent" value={modelConfig.agent_model ?? 'Unknown'} />
        <Stat label="Reranker" value={modelConfig.reranker_model ?? 'Unknown'} />
        <Stat label="Docs" value={stats?.sqlite?.documents ?? 'Not indexed'} />
        <Stat label="Chunks" value={stats?.sqlite?.chunks ?? 'Not indexed'} />
      </dl>
    </section>
  )
}

function Stat({ label, value }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

function StatusBadge({ status }) {
  return <span className={`status-badge status-badge-${status}`}>{toTitleCase(status)}</span>
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
    queryPlan: data.query_plan ?? null,
    warnings: Array.isArray(data.warnings) ? data.warnings : [],
    mode: data.mode ?? 'auto',
    lastQuery: query,
    latencyMs,
  }
}

function errorMessage(data, response) {
  if (typeof data.detail === 'string') return data.detail
  if (data.error) return String(data.error)
  return `${response.status} ${response.statusText || 'Request failed'}`
}

function getInitialMode() {
  const value = new URLSearchParams(window.location.search).get('mode')
  return MODES.some(mode => mode.id === value) ? value : 'auto'
}

function getInitialLimit() {
  const rawValue = new URLSearchParams(window.location.search).get('limit')
  const value = Number(rawValue)
  return Number.isFinite(value) ? Math.min(20, Math.max(3, value)) : 16
}

function formatLatency(latencyMs) {
  if (latencyMs === null || latencyMs === undefined) return 'Not run'
  return `${Math.round(latencyMs).toLocaleString()} ms`
}

function toTitleCase(value) {
  return String(value).replace(/[-_]/g, ' ').replace(/\b\w/g, letter => letter.toUpperCase())
}
