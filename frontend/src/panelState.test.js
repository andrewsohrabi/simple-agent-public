import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildApiMessages,
  createInitialPanels,
  shouldFinalizeSessionBeforeRestart,
  restartPanelState,
  updatePanelState,
} from './panelState.js'

const MODE_CONFIG = [{ key: 'none' }, { key: 'facts' }, { key: 'summary' }]

test('restartPanelState clears apiMessages while preserving display transcript when requested', () => {
  let panels = createInitialPanels(MODE_CONFIG)

  panels = updatePanelState(panels, 'none', panel => ({
    ...panel,
    apiMessages: [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello' },
    ],
    displayMessages: [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello' },
    ],
    modelUsed: 'anthropic:claude-haiku-4-5',
    loading: true,
  }))

  const restarted = restartPanelState(panels, 'none', {
    preserveDisplay: true,
    note: 'Session restarted.',
  })

  assert.equal(restarted.none.apiMessages.length, 0)
  assert.equal(restarted.none.loading, false)
  assert.equal(restarted.none.modelUsed, null)
  assert.deepEqual(restarted.none.displayMessages, [
    { role: 'user', content: 'Hi' },
    { role: 'assistant', content: 'Hello' },
    { role: 'meta', content: 'Session restarted.' },
  ])
})

test('restartPanelState can fully clear the visible transcript', () => {
  let panels = createInitialPanels(MODE_CONFIG)
  panels = updatePanelState(panels, 'facts', panel => ({
    ...panel,
    apiMessages: [{ role: 'user', content: 'Remember this' }],
    displayMessages: [{ role: 'user', content: 'Remember this' }],
  }))

  const restarted = restartPanelState(panels, 'facts')

  assert.deepEqual(restarted.facts.apiMessages, [])
  assert.deepEqual(restarted.facts.displayMessages, [])
})

test('buildApiMessages preserves the current session transcript by default', () => {
  const nextMessages = buildApiMessages(
    [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello' },
    ],
    'What do you remember?'
  )

  assert.deepEqual(nextMessages, [
    { role: 'user', content: 'Hi' },
    { role: 'assistant', content: 'Hello' },
    { role: 'user', content: 'What do you remember?' },
  ])
})

test('buildApiMessages can force a fresh session for restart-based recalls', () => {
  const nextMessages = buildApiMessages(
    [
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello' },
    ],
    'What do you remember?',
    { freshSession: true }
  )

  assert.deepEqual(nextMessages, [{ role: 'user', content: 'What do you remember?' }])
})

test('post-preset follow-up continues from the restarted session, not the setup turns', () => {
  const setupTurn = [
    { role: 'user', content: 'My name is Andrew. My preferred fruit is mango.' },
    { role: 'assistant', content: 'Got it, Andrew.' },
  ]

  const recallTurn = buildApiMessages(setupTurn, 'What is my name and preferred fruit?', {
    freshSession: true,
  })
  const restartedSession = [
    ...recallTurn,
    { role: 'assistant', content: 'Your name is Andrew and your preferred fruit is mango.' },
  ]

  const followUp = buildApiMessages(restartedSession, 'What else do you know about me?')

  assert.deepEqual(followUp, [
    { role: 'user', content: 'What is my name and preferred fruit?' },
    { role: 'assistant', content: 'Your name is Andrew and your preferred fruit is mango.' },
    { role: 'user', content: 'What else do you know about me?' },
  ])
})

test('shouldFinalizeSessionBeforeRestart only finalizes facts sessions with live transcript', () => {
  assert.equal(
    shouldFinalizeSessionBeforeRestart({
      mode: 'facts',
      panel: {
        apiMessages: [{ role: 'user', content: 'Now I focus on 510(k) submissions for brain implants.' }],
      },
      factsExtractor: 'hybrid',
    }),
    true
  )

  assert.equal(
    shouldFinalizeSessionBeforeRestart({
      mode: 'facts',
      panel: { apiMessages: [] },
      factsExtractor: 'hybrid',
    }),
    false
  )

  assert.equal(
    shouldFinalizeSessionBeforeRestart({
      mode: 'summary',
      panel: {
        apiMessages: [{ role: 'user', content: 'Now I focus on 510(k) submissions for brain implants.' }],
      },
      factsExtractor: 'hybrid',
    }),
    false
  )

  assert.equal(
    shouldFinalizeSessionBeforeRestart({
      mode: 'facts',
      panel: {
        apiMessages: [{ role: 'user', content: 'Now I focus on 510(k) submissions for brain implants.' }],
      },
      factsExtractor: 'deterministic',
    }),
    false
  )
})
