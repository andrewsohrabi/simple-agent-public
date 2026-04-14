import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  buildApiMessages,
  createInitialPanels,
  createPanelState,
  restartPanelState,
  shouldFinalizeSessionBeforeRestart,
  updatePanelState,
} from './panelState.js'
import { buildMemoryViews } from './memoryViews.js'
import { SCENARIOS } from './scenarios.js'
import {
  FALLBACK_DEMO_USERS,
  getDemoUserByKey,
  seedDemoUserFallback,
} from './demoUsers.js'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const MODEL_OPTIONS = [
  {
    value: 'anthropic:claude-haiku-4-5',
    label: 'Claude Haiku 4.5',
  },
  {
    value: 'anthropic:claude-sonnet-4-6',
    label: 'Claude Sonnet 4.6',
  },
  {
    value: 'anthropic:claude-opus-4-6',
    label: 'Claude Opus 4.6',
  },
]

const DEFAULT_UI_MODEL = 'anthropic:claude-opus-4-6'
const SIMPLE_DEMO_USER_KEY = 'blank_demo'

const MODE_CONFIG = [
  {
    key: 'none',
    label: 'None',
    subtitle: 'No long-term memory',
    memoryLabel: 'No persistent memory for this mode.',
  },
  {
    key: 'facts',
    label: 'Facts',
    subtitle: 'Structured on-disk facts',
    memoryLabel: 'facts.json',
  },
  {
    key: 'summary',
    label: 'Summary',
    subtitle: 'Latest coherent cross-session summary',
    memoryLabel: 'summary.json',
  },
]

const JSON_HEADERS = { 'Content-Type': 'application/json' }
const MODE_KEYS = MODE_CONFIG.map(({ key }) => key)
const PERSISTENT_MODE_KEYS = MODE_KEYS.filter(key => key !== 'none')
const DEFAULT_PROMPT_NOTICE =
  'No injected system prompt in this demo. The model only sees the live session transcript.'
const FACTS_EXTRACTOR = 'hybrid'

function createInitialSessionMessages() {
  return Object.fromEntries(MODE_KEYS.map(key => [key, []]))
}

function createInitialPanelDrafts() {
  return Object.fromEntries(MODE_KEYS.map(key => [key, '']))
}

function getMessageTone(role) {
  if (role === 'user') return 'user'
  if (role === 'assistant') return 'assistant'
  return 'meta'
}

function getMessageLabel(role) {
  if (role === 'user') return 'You'
  if (role === 'assistant') return 'Assistant'
  return 'Session'
}

function getHealthStatus(health) {
  if (!health) {
    return {
      tone: 'checking',
      label: 'Checking server',
    }
  }

  if (health.serverReachable) {
    return {
      tone: 'ready',
      label: 'Server reachable',
    }
  }

  return {
    tone: 'offline',
    label: 'Server offline',
  }
}

function normalizeHealthPayload(payload) {
  if (payload?.serverReachable !== undefined) {
    return payload
  }

  if (payload?.status === 'ok') {
    return {
      serverReachable: true,
      note: 'The server responded, but detailed health checks are unavailable until the backend is restarted on the latest code.',
      memoryStore: { writable: false, path: 'memory_store' },
      providers: {},
      chatModel: { configured: false, provider: 'unknown' },
      extractorModel: { configured: false, provider: 'unknown' },
      defaultChatModel: 'unknown',
      defaultExtractorModel: 'unknown',
    }
  }

  return {
    serverReachable: false,
    note: 'The frontend could not reach the FastAPI server.',
    memoryStore: { writable: false, path: 'memory_store' },
    providers: {},
    chatModel: { configured: false, provider: 'unknown' },
    extractorModel: { configured: false, provider: 'unknown' },
    defaultChatModel: 'unknown',
    defaultExtractorModel: 'unknown',
  }
}

function defaultPromptPreview(modeKey) {
  if (modeKey === 'none') {
    return DEFAULT_PROMPT_NOTICE
  }
  return 'No persisted memory is currently being injected.'
}

function getModelSelectionNote(isLocked, selectedModelOption) {
  if (isLocked) {
    return 'Current live sessions are already using a model. Restart all sessions before switching models.'
  }

  return `New sessions will use ${selectedModelOption.label}.`
}

function getDemoUserFieldNote(isDemoUserApiAvailable) {
  if (isDemoUserApiAvailable) {
    return 'Selecting a demo path immediately seeds the stored facts and summary for that user. Scenario preset chooses the scripted prompts; demo user path chooses the starting on-disk memory.'
  }

  return 'This backend does not expose /demo-users, so the frontend seeds the selected path through the normal chat and memory APIs instead. Scenario preset still chooses the scripted prompts; demo user path still chooses the starting on-disk memory.'
}

function getScenarioFieldNote(showDeveloperMode) {
  if (showDeveloperMode) {
    return 'Runs the selected scripted setup, then restarts the session in the UI and asks the recall question on top of the selected starting saved memory.'
  }

  return 'Runs the selected scripted setup from a blank saved-memory start, restarts the session in the UI, and then asks the recall question.'
}

function getSeedNotice(seedSource, demoUser, nextUserId) {
  if (seedSource === 'backend') {
    return `Loaded ${demoUser.label}. Facts and summary memory were seeded on disk for ${nextUserId}.`
  }

  return `Loaded ${demoUser.label}. Facts and summary were seeded through the normal chat + memory APIs for ${nextUserId}.`
}

function getSeedingPathLabel(seedSource) {
  if (seedSource === 'backend') {
    return 'backend demo-user API'
  }

  return 'client fallback via chat + memory APIs'
}

function getRecommendedDemoUserLabels(recommendedUsers, demoUsers) {
  return recommendedUsers
    .map(key => demoUsers.find(user => user.key === key)?.label ?? key)
    .join(', ')
}

function createSeededPanels(data) {
  return {
    none: {
      ...createPanelState(),
      promptPreview: defaultPromptPreview('none'),
    },
    facts: {
      ...createPanelState(),
      memory: data.facts,
      factsEvents: data.factsEvents ?? [],
      factsEventsPath: data.factsEventsPath ?? null,
      promptPreview: data.promptPreviews?.facts ?? defaultPromptPreview('facts'),
    },
    summary: {
      ...createPanelState(),
      memory: data.summary,
      promptPreview: data.promptPreviews?.summary ?? defaultPromptPreview('summary'),
    },
  }
}

function scrollTranscriptToBottom(element) {
  if (!element) {
    return
  }

  element.scrollTo({
    top: element.scrollHeight,
    behavior: 'smooth',
  })
}

function ModePanel({
  mode,
  panel,
  userId,
  draft,
  disabled,
  onRestart,
  onClearMemory,
  onRefreshMemory,
  onDraftChange,
  onSend,
  setTranscriptRef,
}) {
  const memoryViews = buildMemoryViews(mode.key, panel, mode.memoryLabel)

  return (
    <section className="panel">
      <div className="panel-header">
        <div className="panel-copy">
          <p className="panel-eyebrow">{mode.label}</p>
          <h2 className="panel-title">{mode.subtitle}</h2>
          <p className="panel-meta">User: {userId}</p>
          {panel.modelUsed && <p className="panel-meta">Model: {panel.modelUsed}</p>}
        </div>
        <div className="panel-actions">
          <button
            type="button"
            className="button button-secondary"
            onClick={onRestart}
          >
            Restart session
          </button>
          {mode.key !== 'none' && (
            <button
              type="button"
              className="button button-secondary"
              onClick={onClearMemory}
            >
              Clear memory
            </button>
          )}
        </div>
      </div>

      <div className="transcript" ref={setTranscriptRef}>
        {panel.displayMessages.length === 0 && (
          <p className="empty-state">
            No turns yet. Run a preset scenario or continue this session here.
          </p>
        )}

        {panel.displayMessages.map((message, index) => (
          <div
            key={`${mode.key}-${index}`}
            className={`message-card message-${getMessageTone(message.role)}`}
          >
            <span className={`message-role role-${getMessageTone(message.role)}`}>
              {getMessageLabel(message.role)}
            </span>
            <p className="message-content">{message.content}</p>
          </div>
        ))}

        {panel.loading && (
          <div className="message-card message-assistant">
            <span className="message-role role-assistant">Assistant</span>
            <p className="message-content">Thinking…</p>
          </div>
        )}
      </div>

      <div className="panel-composer">
        <span className="field-label">Continue this session</span>
        <div className="panel-composer-row">
          <input
            value={draft}
            onChange={event => onDraftChange(event.target.value)}
            className="control-input panel-input"
            placeholder={`Ask ${mode.label.toLowerCase()} a follow-up question after the preset or start a fresh live session.`}
            disabled={disabled}
          />
          <button
            type="button"
            className="button button-secondary"
            onClick={onSend}
            disabled={disabled || !draft.trim()}
          >
            Send
          </button>
        </div>
      </div>

      <div className="memory-section">
        {mode.key === 'none' ? (
          <p className="empty-state">No persistent memory for this mode.</p>
        ) : (
          memoryViews.map((view, index) => (
            <div className="memory-view" key={`${mode.key}-memory-view-${index}`}>
              <div className="memory-header">
                <span className="field-label">{view.label}</span>
                {index === 0 && (
                  <button
                    type="button"
                    className="button button-text"
                    onClick={onRefreshMemory}
                  >
                    Refresh
                  </button>
                )}
              </div>
              <pre className="memory-code">{view.content}</pre>
            </div>
          ))
        )}
        <div className="prompt-section">
          <span className="field-label">Effective system prompt</span>
          <pre className="prompt-code">
            {panel.promptPreview ?? defaultPromptPreview(mode.key)}
          </pre>
        </div>
      </div>
    </section>
  )
}

export default function App() {
  const [userId, setUserId] = useState('demo_user')
  const [input, setInput] = useState('')
  const [scenarioKey, setScenarioKey] = useState('identity_recall')
  const [selectedModel, setSelectedModel] = useState(DEFAULT_UI_MODEL)
  const [panels, setPanels] = useState(() => createInitialPanels(MODE_CONFIG))
  const [panelDrafts, setPanelDrafts] = useState(() => createInitialPanelDrafts())
  const [health, setHealth] = useState(null)
  const [runningScenario, setRunningScenario] = useState(false)
  const [demoUsers, setDemoUsers] = useState([])
  const [selectedDemoUserKey, setSelectedDemoUserKey] = useState('blank_demo')
  const [activeDemoUserKey, setActiveDemoUserKey] = useState('blank_demo')
  const [demoUserLoading, setDemoUserLoading] = useState(false)
  const [demoUserApiAvailable, setDemoUserApiAvailable] = useState(true)
  const [demoUserSeedSource, setDemoUserSeedSource] = useState('backend')
  const [showDeveloperMode, setShowDeveloperMode] = useState(false)
  const [uiNotice, setUiNotice] = useState('')
  const transcriptRefs = useRef({})
  const panelsRef = useRef(panels)
  const sessionMessagesRef = useRef(createInitialSessionMessages())
  const activeUserIdRef = useRef('demo_user')

  useEffect(() => {
    void initializeDemo()
  }, [])

  useEffect(() => {
    if (showDeveloperMode || demoUserLoading || runningScenario) {
      return
    }
    if (activeDemoUserKey === SIMPLE_DEMO_USER_KEY) {
      return
    }
    void applyDemoUser(SIMPLE_DEMO_USER_KEY, { showNotice: false })
  }, [showDeveloperMode, activeDemoUserKey, demoUserLoading, runningScenario])

  useEffect(() => {
    MODE_CONFIG.forEach(({ key }) => {
      scrollTranscriptToBottom(transcriptRefs.current[key])
    })
  }, [panels])

  const scenarioOptions = useMemo(
    () => Object.entries(SCENARIOS).map(([key, value]) => ({ key, ...value })),
    []
  )

  const selectedScenario = SCENARIOS[scenarioKey]
  const resolvedDemoUsers = Array.isArray(demoUsers) && demoUsers.length ? demoUsers : FALLBACK_DEMO_USERS
  const activeDemoUser = getDemoUserByKey(resolvedDemoUsers, activeDemoUserKey)
  const selectedModelOption =
    MODEL_OPTIONS.find(option => option.value === selectedModel) ?? MODEL_OPTIONS[0]
  const modelSelectionLocked = MODE_CONFIG.some(
    ({ key }) => panels[key].loading || (panels[key].apiMessages?.length ?? 0) > 0
  )
  const healthStatus = getHealthStatus(health)
  const recommendedDemoUsersText = getRecommendedDemoUserLabels(
    selectedScenario.recommendedUsers,
    resolvedDemoUsers
  )
  const modelSelectionNote = getModelSelectionNote(modelSelectionLocked, selectedModelOption)
  const demoUserFieldNote = getDemoUserFieldNote(demoUserApiAvailable)
  const scenarioFieldNote = getScenarioFieldNote(showDeveloperMode)

  async function initializeDemo() {
    await checkHealth()
    await loadDemoUsers()
  }

  function commitPanels(updater) {
    const nextPanels =
      typeof updater === 'function' ? updater(panelsRef.current) : updater
    panelsRef.current = nextPanels
    setPanels(nextPanels)
    return nextPanels
  }

  function updatePanel(mode, updater) {
    return commitPanels(current => updatePanelState(current, mode, updater))
  }

  function getActiveUserId() {
    return activeUserIdRef.current || userId
  }

  function updatePanelDraft(mode, value) {
    setPanelDrafts(current => ({
      ...current,
      [mode]: value,
    }))
  }

  function resetPanelDrafts() {
    setPanelDrafts(createInitialPanelDrafts())
  }

  function appendPanelMetaMessage(mode, content) {
    updatePanel(mode, panel => ({
      ...panel,
      displayMessages: [...panel.displayMessages, { role: 'meta', content }],
    }))
  }

  async function checkHealth() {
    try {
      const response = await fetch(`${API_URL}/health`)
      const data = await response.json()
      setHealth(normalizeHealthPayload(data))
    } catch {
      setHealth(normalizeHealthPayload(null))
    }
  }

  async function loadDemoUsers() {
    try {
      const response = await fetch(`${API_URL}/demo-users`)
      if (!response.ok) {
        throw new Error('backend does not yet expose /demo-users')
      }
      const data = await response.json()
      const nextDemoUsers = Array.isArray(data.demoUsers) ? data.demoUsers : FALLBACK_DEMO_USERS
      const nextDefault = data.defaultDemoUser || 'blank_demo'
      setDemoUserApiAvailable(true)
      setDemoUserSeedSource('backend')
      setDemoUsers(nextDemoUsers)
      setSelectedDemoUserKey(nextDefault)
      await applyDemoUser(nextDefault, { showNotice: false })
    } catch (error) {
      setDemoUserApiAvailable(false)
      setDemoUserSeedSource('fallback')
      setDemoUsers(FALLBACK_DEMO_USERS)
      setSelectedDemoUserKey('blank_demo')
      await applyDemoUser('blank_demo', {
        showNotice: false,
        forceFallback: true,
      })
      setUiNotice(
        `The backend does not expose /demo-users. The demo-user dropdown still works by seeding memory through the normal chat and memory APIs. (${error.message})`
      )
    }
  }

  async function refreshMemory(mode, nextUserId = activeUserIdRef.current || userId) {
    if (mode === 'none') {
      updatePanel(mode, panel => ({ ...panel, memory: null }))
      return
    }

    try {
      const params = new URLSearchParams({ memoryType: mode, userId: nextUserId })
      const response = await fetch(`${API_URL}/memory?${params.toString()}`)
      const data = await response.json()
      updatePanel(mode, panel => ({
        ...panel,
        memory: data.memory,
        factsEvents: data.factsEvents,
        factsEventsPath: data.factsEventsPath,
        promptPreview: data.promptPreview ?? defaultPromptPreview(mode),
      }))
    } catch (error) {
      appendPanelMetaMessage(mode, `Failed to load memory: ${error.message}`)
    }
  }

  async function clearMemory(mode) {
    if (mode === 'none') return
    const nextUserId = getActiveUserId()
    try {
      const data = await deleteModeMemory(mode, nextUserId)
      updatePanel(mode, panel => ({
        ...panel,
        memory: data.memory,
        factsEvents: data.factsEvents,
        factsEventsPath: data.factsEventsPath,
        promptPreview: data.promptPreview ?? defaultPromptPreview(mode),
        displayMessages: [...panel.displayMessages],
      }))
      appendPanelMetaMessage(mode, `${mode} memory cleared for ${nextUserId}.`)
    } catch (error) {
      appendPanelMetaMessage(mode, `Failed to clear memory: ${error.message}`)
    }
  }

  async function deleteModeMemory(mode, nextUserId) {
    const params = new URLSearchParams({ memoryType: mode, userId: nextUserId })
    const response = await fetch(`${API_URL}/memory?${params.toString()}`, {
      method: 'DELETE',
    })
    return response.json()
  }

  async function finalizeSessionBeforeRestart(mode) {
    const panel = panelsRef.current[mode]
    if (
      !shouldFinalizeSessionBeforeRestart({
        mode,
        panel,
        factsExtractor: FACTS_EXTRACTOR,
      })
    ) {
      return
    }

    try {
      const response = await fetch(`${API_URL}/session/finalize`, {
        method: 'POST',
        headers: JSON_HEADERS,
        body: JSON.stringify({
          messages: panel.apiMessages,
          memoryType: mode,
          userId: getActiveUserId(),
          factsExtractor: FACTS_EXTRACTOR,
        }),
      })
      if (!response.ok) {
        throw new Error(`session finalize failed (${response.status})`)
      }

      const data = await response.json()
      updatePanel(mode, currentPanel => ({
        ...currentPanel,
        memory: data.memory,
        factsEvents: data.factsEvents,
        factsEventsPath: data.factsEventsPath,
        promptPreview: data.promptPreview ?? defaultPromptPreview(mode),
      }))
    } catch (error) {
      appendPanelMetaMessage(mode, `Failed to finalize session memory: ${error.message}`)
    }
  }

  function restartSession(mode, { preserveDisplay = false, note } = {}) {
    sessionMessagesRef.current = {
      ...sessionMessagesRef.current,
      [mode]: [],
    }
    updatePanelDraft(mode, '')
    return commitPanels(current =>
      restartPanelState(current, mode, { preserveDisplay, note })
    )
  }

  async function prepareSessionRestart(
    mode,
    { preserveDisplay = false, note, skipFinalize = false } = {}
  ) {
    if (!skipFinalize) {
      await finalizeSessionBeforeRestart(mode)
    }
    restartSession(mode, { preserveDisplay, note })
  }

  async function restartAllSessions({ announce = true, skipFinalize = false } = {}) {
    resetPanelDrafts()
    await Promise.allSettled(
      MODE_CONFIG.map(({ key }) =>
        prepareSessionRestart(key, { skipFinalize })
      )
    )
    if (announce) {
      setUiNotice(
        'Restarted all sessions. Live conversation history was cleared; saved facts and summary memory were preserved.'
      )
    }
  }

  async function clearAllPersistentMemory({ restartSessions = false, notice } = {}) {
    const nextUserId = getActiveUserId()
    const results = await Promise.allSettled(
      PERSISTENT_MODE_KEYS.map(async mode => ({
        mode,
        data: await deleteModeMemory(mode, nextUserId),
      }))
    )

    results.forEach(result => {
      if (result.status === 'fulfilled') {
        const { mode, data } = result.value
        updatePanel(mode, panel => ({
          ...panel,
          memory: data.memory,
          factsEvents: data.factsEvents,
          factsEventsPath: data.factsEventsPath,
          promptPreview: data.promptPreview ?? defaultPromptPreview(mode),
          displayMessages: [...panel.displayMessages],
        }))
        if (!restartSessions) {
          appendPanelMetaMessage(mode, `${mode} memory cleared for ${nextUserId}.`)
        }
        return
      }

      PERSISTENT_MODE_KEYS.forEach(mode => {
        appendPanelMetaMessage(mode, `Failed to clear memory: ${result.reason.message}`)
      })
    })

    if (restartSessions) {
      await restartAllSessions({ announce: false, skipFinalize: true })
    }

    if (notice) {
      setUiNotice(notice)
      return
    }

    if (restartSessions) {
      setUiNotice(`Cleared saved facts and summary memory for ${nextUserId} and restarted all sessions.`)
      return
    }

    setUiNotice(`Cleared saved facts and summary memory for ${nextUserId}. Live conversation history was preserved.`)
  }

  async function resetSimpleMode() {
    await applyDemoUser(SIMPLE_DEMO_USER_KEY, { showNotice: false })
    setUiNotice('Cleared saved memory and restarted all sessions from a blank start.')
  }

  async function sendToMode(mode, content, { freshSession = false } = {}) {
    const currentApiMessages = sessionMessagesRef.current[mode] || []
    const apiMessages = buildApiMessages(currentApiMessages, content, { freshSession })
    const nextUserId = getActiveUserId()
    sessionMessagesRef.current = {
      ...sessionMessagesRef.current,
      [mode]: apiMessages,
    }

    updatePanel(mode, panel => ({
      ...panel,
      apiMessages,
      loading: true,
      displayMessages: [...panel.displayMessages, { role: 'user', content }],
    }))

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: JSON_HEADERS,
        body: JSON.stringify({
          messages: apiMessages,
          memoryType: mode,
          userId: nextUserId,
          model: selectedModel,
          factsExtractor: mode === 'facts' ? FACTS_EXTRACTOR : undefined,
        }),
      })
      const data = await response.json()
      const assistantMessage = { role: 'assistant', content: data.reply }

      updatePanel(mode, panel => ({
        ...panel,
        apiMessages: [...apiMessages, assistantMessage],
        displayMessages: [...panel.displayMessages, assistantMessage],
        loading: false,
        memory: data.memory,
        factsEvents: data.factsEvents,
        factsEventsPath: data.factsEventsPath,
        promptPreview: data.promptPreview ?? defaultPromptPreview(mode),
        modelUsed: data.modelUsed,
      }))
      sessionMessagesRef.current = {
        ...sessionMessagesRef.current,
        [mode]: [...apiMessages, assistantMessage],
      }
    } catch (error) {
      updatePanel(mode, panel => ({
        ...panel,
        apiMessages,
        displayMessages: [
          ...panel.displayMessages,
          { role: 'assistant', content: `Error: ${error.message}` },
        ],
        loading: false,
      }))
    }
  }

  async function sendBroadcast() {
    if (!input.trim() || runningScenario || demoUserLoading) return
    const message = input.trim()
    setInput('')
    setUiNotice(
      'Sent the shared prompt to all three modes without restarting the current session.'
    )
    await Promise.allSettled(MODE_CONFIG.map(({ key }) => sendToMode(key, message)))
  }

  async function sendPanelFollowUp(mode) {
    const draft = panelDrafts[mode]?.trim()
    if (!draft || runningScenario || demoUserLoading) return

    updatePanelDraft(mode, '')
    await sendToMode(mode, draft)
  }

  async function runScenario() {
    const scenario = SCENARIOS[scenarioKey]
    if (!scenario || demoUserLoading) return

    setRunningScenario(true)
    resetPanelDrafts()
    setUiNotice(
      `Running ${scenario.label}. After it finishes, continue in any panel with its own follow-up input.`
    )
    try {
      const scenarioDemoUserKey = showDeveloperMode
        ? selectedDemoUserKey
        : SIMPLE_DEMO_USER_KEY

      if (scenarioDemoUserKey !== activeDemoUserKey) {
        await applyDemoUser(scenarioDemoUserKey, { showNotice: false })
      }

      for (const prompt of scenario.setup) {
        await Promise.allSettled(MODE_CONFIG.map(({ key }) => sendToMode(key, prompt)))
      }

      await Promise.allSettled(
        MODE_CONFIG.map(({ key }) =>
          prepareSessionRestart(key, {
            preserveDisplay: true,
            note: `Session restarted to test cross-session memory for ${scenario.label}.`,
          })
        )
      )

      let isFirstRecallTurn = true
      for (const prompt of scenario.recall) {
        await Promise.allSettled(
          MODE_CONFIG.map(({ key }) =>
            sendToMode(key, prompt, { freshSession: isFirstRecallTurn })
          )
        )
        isFirstRecallTurn = false
      }
    } finally {
      setRunningScenario(false)
    }
  }

  async function applyDemoUser(
    nextDemoUserKey = selectedDemoUserKey,
    { showNotice = true, forceFallback = false } = {}
  ) {
    setDemoUserLoading(true)
    try {
      const { data, seedSource } = await loadDemoUserData(nextDemoUserKey, forceFallback)

      setSelectedDemoUserKey(nextDemoUserKey)
      setActiveDemoUserKey(nextDemoUserKey)
      setUserId(data.userId)
      activeUserIdRef.current = data.userId
      sessionMessagesRef.current = createInitialSessionMessages()
      resetPanelDrafts()
      setDemoUserSeedSource(seedSource)
      commitPanels(createSeededPanels(data))
      if (showNotice) {
        setUiNotice(getSeedNotice(seedSource, data.demoUser, data.userId))
      }
    } catch (error) {
      setSelectedDemoUserKey(activeDemoUserKey)
      setUiNotice(`Failed to load demo user: ${error.message}`)
    } finally {
      setDemoUserLoading(false)
    }
  }

  async function loadDemoUserData(nextDemoUserKey, forceFallback) {
    if (demoUserApiAvailable && !forceFallback) {
      try {
        const response = await fetch(`${API_URL}/demo-users/load`, {
          method: 'POST',
          headers: JSON_HEADERS,
          body: JSON.stringify({ demoUser: nextDemoUserKey }),
        })
        if (!response.ok) {
          throw new Error('backend does not yet expose demo-user loading')
        }
        return {
          data: await response.json(),
          seedSource: 'backend',
        }
      } catch {
        return {
          data: await seedDemoUserFallback({
            apiUrl: API_URL,
            demoUser: getDemoUserByKey(resolvedDemoUsers, nextDemoUserKey),
            model: selectedModel,
          }),
          seedSource: 'fallback',
        }
      }
    }

    const demoUserPool = forceFallback ? FALLBACK_DEMO_USERS : resolvedDemoUsers
    return {
      data: await seedDemoUserFallback({
        apiUrl: API_URL,
        demoUser: getDemoUserByKey(demoUserPool, nextDemoUserKey),
        model: selectedModel,
      }),
      seedSource: 'fallback',
    }
  }

  return (
    <div className="app-shell">
      <div className="app-page">
        <header className="hero">
          <div className="hero-copy">
            <p className="eyebrow">Memory Demo</p>
            <h1 className="hero-title">Compare same-chat context against cross-session memory</h1>
            <p className="hero-lead">
              Run a preset from a blank start, then continue the restarted session inside any
              panel while inspecting the raw JSON memory for each mode.
            </p>
          </div>
          <div className={`health-badge health-${healthStatus.tone}`}>{healthStatus.label}</div>
        </header>

        <section className="control-band">
          <div className="control-band-header">
            <div>
              <p className="info-title">Preset runner</p>
              <p className="helper-note">
                Simple mode always starts from blank saved memory. Dev mode exposes preseeded
                paths and broadcast tools.
              </p>
            </div>
            <label className="mode-toggle" htmlFor="developer-mode-toggle">
              <span className="mode-toggle-copy">
                <span className="field-label">Dev mode</span>
              </span>
              <span className="mode-toggle-control">
                <input
                  id="developer-mode-toggle"
                  type="checkbox"
                  checked={showDeveloperMode}
                  onChange={event => setShowDeveloperMode(event.target.checked)}
                  disabled={runningScenario || demoUserLoading}
                />
                <span className="mode-toggle-slider" aria-hidden="true" />
              </span>
            </label>
          </div>

          <div className="control-layout">
            <div className="control-stack">
              <div className={`stack-grid ${showDeveloperMode ? 'stack-grid-dev' : 'stack-grid-simple'}`}>
                <label className="field">
                  <span className="field-label">Scenario preset</span>
                  <select
                    value={scenarioKey}
                    onChange={event => setScenarioKey(event.target.value)}
                    className="control-input"
                    disabled={runningScenario}
                  >
                    {scenarioOptions.map(option => (
                      <option key={option.key} value={option.key}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <p className="field-note">{scenarioFieldNote}</p>
                </label>

                {showDeveloperMode && (
                  <label className="field">
                    <span className="field-label">Demo user path</span>
                    <select
                      value={selectedDemoUserKey}
                      onChange={event => {
                        const nextKey = event.target.value
                        setSelectedDemoUserKey(nextKey)
                        void applyDemoUser(nextKey)
                      }}
                      className="control-input"
                      disabled={runningScenario || demoUserLoading}
                    >
                      {resolvedDemoUsers.map(user => (
                        <option key={user.key} value={user.key}>
                          {user.label}
                        </option>
                      ))}
                    </select>
                    <p className="field-note">{demoUserFieldNote}</p>
                  </label>
                )}

                <label className="field">
                  <span className="field-label">Session model</span>
                  <select
                    value={selectedModel}
                    onChange={event => setSelectedModel(event.target.value)}
                    className="control-input"
                    disabled={runningScenario || demoUserLoading || modelSelectionLocked}
                  >
                    {MODEL_OPTIONS.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <p className="field-note">{modelSelectionNote}</p>
                </label>
              </div>

              <div className="action-row action-row-primary">
                <button
                  type="button"
                  className="button button-primary"
                  onClick={runScenario}
                  disabled={runningScenario || demoUserLoading}
                >
                  Run preset
                </button>
                <button
                  type="button"
                  className="button button-secondary"
                  onClick={() => void restartAllSessions()}
                  disabled={demoUserLoading}
                >
                  Restart all sessions
                </button>
                {showDeveloperMode ? (
                  <>
                    <button
                      type="button"
                      className="button button-secondary"
                      onClick={() => void clearAllPersistentMemory()}
                      disabled={runningScenario || demoUserLoading}
                    >
                      Clear all memory
                    </button>
                    <button
                      type="button"
                      className="button button-secondary"
                      onClick={() => void clearAllPersistentMemory({ restartSessions: true })}
                      disabled={runningScenario || demoUserLoading}
                    >
                      Clear memory + restart
                    </button>
                  </>
                ) : (
                  <button
                    type="button"
                    className="button button-secondary"
                    onClick={() => void resetSimpleMode()}
                    disabled={runningScenario || demoUserLoading}
                  >
                    Clear memory + restart
                  </button>
                )}
              </div>
            </div>
          </div>

          {showDeveloperMode && (
            <div className="developer-tools">
              <label className="field field-prompt">
                <span className="field-label">Shared prompt</span>
                <textarea
                  value={input}
                  onChange={event => setInput(event.target.value)}
                  placeholder="Developer mode: send one message to none, facts, and summary."
                  rows={3}
                  className="control-input control-textarea"
                  disabled={runningScenario || demoUserLoading}
                />
              </label>
              <div className="action-row action-row-dev">
                <button
                  type="button"
                  className="button button-secondary"
                  onClick={sendBroadcast}
                  disabled={runningScenario || demoUserLoading}
                >
                  Send to all
                </button>
              </div>
            </div>
          )}

          <div className="explainers">
            <article className="info-card">
              <p className="info-title">Selected scenario</p>
              <h2 className="info-heading">{selectedScenario.label}</h2>
              <p className="info-body">{selectedScenario.description}</p>
              <p className="info-detail">
                <strong>What it tests:</strong> {selectedScenario.tests}
              </p>
              <div className="flow-list">
                <div>
                  <span className="flow-label">Setup</span>
                  <ul className="compact-list">
                    {selectedScenario.setup.map(prompt => (
                      <li key={prompt}>{prompt}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <span className="flow-label">After restart</span>
                  <ul className="compact-list">
                    {selectedScenario.recall.map(prompt => (
                      <li key={prompt}>{prompt}</li>
                    ))}
                  </ul>
                </div>
              </div>
              {showDeveloperMode ? (
                <p className="helper-note">
                  Recommended demo users: {recommendedDemoUsersText}
                </p>
              ) : (
                <p className="helper-note">
                  Simple mode always runs this scenario from blank saved memory.
                </p>
              )}
            </article>

            <article className="info-card">
              <p className="info-title">{showDeveloperMode ? 'Control meanings' : 'How this run works'}</p>
              {showDeveloperMode ? (
                <ul className="compact-list">
                  <li>
                    <strong>Run preset</strong> executes the scripted setup prompts, restarts the
                    session in the UI, and then runs the two recall prompts.
                  </li>
                  <li>
                    <strong>Demo user path</strong> selects the starting stored memory. Changing it
                    immediately seeds backend memory and switches the active user id for all panels.
                  </li>
                  <li>
                    <strong>Continue this session</strong> appears in each panel. Use it after the
                    preset if you want to keep talking to that specific mode.
                  </li>
                  <li>
                    <strong>Restart all sessions</strong> clears only the live session transcript in
                    each panel. For `facts` in hybrid mode, it first finalizes the current facts
                    session, then clears the transcript. It does not delete stored `facts.json` or
                    `summary.json`.
                  </li>
                  <li>
                    <strong>Clear all memory</strong> deletes saved `facts.json` and `summary.json`
                    for the active user but leaves the current live transcript visible.
                  </li>
                  <li>
                    <strong>Clear memory + restart</strong> deletes saved memory and then clears the
                    live transcript for all three panels.
                  </li>
                  <li>
                    <strong>Session model</strong> applies to new turns only. Once a live session
                    has started, restart all sessions before switching models.
                  </li>
                </ul>
              ) : (
                <ul className="compact-list">
                  <li>
                    <strong>Run preset</strong> starts from blank saved memory, runs the scripted
                    setup turns, restarts the session, and then runs the two recall turns.
                  </li>
                  <li>
                    <strong>Continue this session</strong> in any panel keeps talking to that mode
                    after the restarted recall.
                  </li>
                  <li>
                    <strong>Restart all sessions</strong> clears the live transcript only. It does
                    not delete the saved `facts.json` or `summary.json` created during the run.
                    For `facts` in hybrid mode, it first finalizes the current facts session.
                  </li>
                  <li>
                    <strong>Clear memory + restart</strong> deletes saved `facts.json` and
                    `summary.json`, then restarts all three panels from a blank state.
                  </li>
                </ul>
              )}
              {uiNotice && <p className="helper-note helper-notice">{uiNotice}</p>}
            </article>

            {showDeveloperMode && (
              <article className="info-card">
                <p className="info-title">Backend checks</p>
                <dl className="status-list">
                  <div>
                    <dt>API server</dt>
                    <dd>{health?.serverReachable ? 'reachable' : 'offline'}</dd>
                  </div>
                  <div>
                    <dt>Memory store</dt>
                    <dd>{health?.memoryStore?.writable ? 'writable' : 'not writable'}</dd>
                  </div>
                  <div>
                    <dt>Default chat model</dt>
                    <dd>
                      {health?.defaultChatModel ?? 'unknown'} ·{' '}
                      {health?.chatModel?.configured ? 'provider env present' : 'provider env missing'}
                    </dd>
                  </div>
                  <div>
                    <dt>Extractor model</dt>
                    <dd>
                      {health?.defaultExtractorModel ?? 'unknown'} ·{' '}
                      {health?.extractorModel?.configured
                        ? 'provider env present'
                        : 'provider env missing'}
                    </dd>
                  </div>
                  <div>
                    <dt>Scope</dt>
                    <dd>{health?.note ?? 'No health details available.'}</dd>
                  </div>
                </dl>
              </article>
            )}
          </div>
        </section>

        {showDeveloperMode && (
        <section className="active-user-card">
          <p className="info-title">Active seeded demo path</p>
          <div className="active-user-grid">
            <div>
              <h2 className="info-heading">
                {activeDemoUser?.label ?? 'Demo user'} · <span className="inline-code">{userId}</span>
              </h2>
              <p className="info-body">
                {activeDemoUser?.description ??
                  'This user id is active for all three comparison panels.'}
              </p>
            </div>
            <p className="helper-note">
              <strong>Best for:</strong> {activeDemoUser?.bestFor ?? 'General manual testing.'}
            </p>
          </div>
          <p className="helper-note">
            <strong>How this combines with the scenario preset:</strong> the scenario controls the
            prompts that run; the active demo path controls the persisted facts and summary already
            stored on disk before those prompts begin.
          </p>
          <p className="helper-note">
            <strong>Current session model:</strong> {selectedModelOption.label}
          </p>
          {!demoUserApiAvailable && (
            <p className="helper-note helper-notice">
              The backend demo-user API is unavailable, so this page is using client-side seeding
              through `/chat` and `/memory`. The demo paths still work; they just use a more
              portable fallback path.
            </p>
          )}
          <p className="helper-note">
            <strong>Seeding path:</strong>{' '}
            {getSeedingPathLabel(demoUserSeedSource)}
          </p>
          <p className="helper-note mode-note">
            <strong>Mode expectations:</strong> `none` should only use the current live session
            transcript. `facts` stores discrete fields and is strongest for precise recall,
            updates, and contradiction handling. `summary` stores one latest-state narrative by
            default and can pull a small targeted temporal block only for clearly temporal
            questions.
          </p>
          <p className="helper-note">
            <strong>Why `facts` and `summary` can look similar on some presets:</strong> both can
            answer simple recall questions after a restart. The difference becomes clearer on
            update and contradiction flows, where `facts` exposes raw structured state plus visible
            event history, while `summary` keeps a compact latest-state narrative and only consults
            targeted history for explicit before/after questions.
          </p>
        </section>
        )}

        <main className="panel-grid">
          {MODE_CONFIG.map(mode => (
            <ModePanel
              key={mode.key}
              mode={mode}
              panel={panels[mode.key]}
              userId={userId}
              draft={panelDrafts[mode.key]}
              disabled={runningScenario || demoUserLoading || panels[mode.key].loading}
              onRestart={() => void prepareSessionRestart(mode.key)}
              onClearMemory={() => clearMemory(mode.key)}
              onRefreshMemory={() => refreshMemory(mode.key)}
              onDraftChange={value => updatePanelDraft(mode.key, value)}
              onSend={() => void sendPanelFollowUp(mode.key)}
              setTranscriptRef={element => {
                transcriptRefs.current[mode.key] = element
              }}
            />
          ))}
        </main>
      </div>
    </div>
  )
}
