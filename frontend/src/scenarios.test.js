import test from 'node:test'
import assert from 'node:assert/strict'

import { SCENARIOS } from './scenarios.js'

test('all presets include overwrite setup turns and a prior-state recall turn', () => {
  for (const scenario of Object.values(SCENARIOS)) {
    assert.ok(scenario.setup.length >= 2)
    assert.equal(scenario.recall.length, 2)
    assert.match(
      scenario.recall[1].toLowerCase(),
      /(before|prior|previous)/
    )
  }
})

test('preset scripts define the expected overwrite and temporal recall flows', () => {
  assert.deepEqual(SCENARIOS.identity_recall.setup, [
    "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a medical device company, focusing on 510(k) submissions for cardiac devices.",
    'I got married and my last name is now Doe.',
    'Now I focus on 510(k) submissions for brain implants.',
  ])
  assert.deepEqual(SCENARIOS.identity_recall.recall, [
    'What do you remember about me and my work now?',
    'What did I focus on before my current focus area?',
  ])

  assert.deepEqual(SCENARIOS.preference_application.setup, [
    'Going forward, always give me three-line haiku answers.',
    'Update that: start every answer with an ALL-CAPS summary line, then give me concise bullet-point answers. No haikus.',
  ])
  assert.deepEqual(SCENARIOS.preference_application.recall, [
    'Explain the key considerations for predicate device selection in a 510(k).',
    'What response style did I ask for before my current one?',
  ])

  assert.deepEqual(SCENARIOS.project_context_recall.setup, [
    "I'm working on a 510(k) for a new catheter. The main challenge is choosing between two predicate devices.",
    "Now I'm working on a 510(k) for a brain implant. The main challenge is building the clinical evidence plan.",
  ])
  assert.deepEqual(SCENARIOS.project_context_recall.recall, [
    'Can you help me think through next steps for my current project?',
    'What project was I working on before my current one?',
  ])

  assert.deepEqual(SCENARIOS.contradiction_update.setup, [
    'I work in Regulatory Affairs.',
    'Actually, I just transferred to Quality Assurance.',
  ])
  assert.deepEqual(SCENARIOS.contradiction_update.recall, [
    'What department am I in now?',
    'What was my job before?',
  ])

  assert.deepEqual(SCENARIOS.personal_preference_recall.setup, [
    'My name is Andrew. My preferred fruit is mango.',
    'Actually, my preferred fruit is pear now.',
  ])
  assert.deepEqual(SCENARIOS.personal_preference_recall.recall, [
    'What is my name and preferred fruit now?',
    'What fruit did I prefer before my current one?',
  ])
})
