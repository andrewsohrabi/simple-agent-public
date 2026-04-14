export function createPanelState() {
  return {
    apiMessages: [],
    displayMessages: [],
    loading: false,
    memory: null,
    factsEvents: null,
    factsEventsPath: null,
    promptPreview: null,
    modelUsed: null,
  }
}

export function createInitialPanels(modeConfig) {
  return Object.fromEntries(modeConfig.map(mode => [mode.key, createPanelState()]))
}

export function updatePanelState(currentPanels, mode, updater) {
  const currentPanel = currentPanels[mode]
  const nextPanel =
    typeof updater === 'function' ? updater(currentPanel) : { ...currentPanel, ...updater }
  return {
    ...currentPanels,
    [mode]: nextPanel,
  }
}

export function restartPanelState(currentPanels, mode, { preserveDisplay = false, note } = {}) {
  return updatePanelState(currentPanels, mode, panel => ({
    ...panel,
    apiMessages: [],
    loading: false,
    modelUsed: null,
    displayMessages: preserveDisplay
      ? [...panel.displayMessages, ...(note ? [{ role: 'meta', content: note }] : [])]
      : [],
  }))
}

export function buildApiMessages(currentMessages, content, { freshSession = false } = {}) {
  const baseMessages = freshSession ? [] : currentMessages ?? []
  return [...baseMessages, { role: 'user', content }]
}

export function shouldFinalizeSessionBeforeRestart({
  mode,
  panel,
  factsExtractor,
}) {
  if (mode !== 'facts') {
    return false
  }

  if (!['hybrid', 'llm'].includes(factsExtractor)) {
    return false
  }

  return Array.isArray(panel?.apiMessages) && panel.apiMessages.length > 0
}
