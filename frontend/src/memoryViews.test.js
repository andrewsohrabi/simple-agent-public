import test from 'node:test'
import assert from 'node:assert/strict'

import { buildMemoryViews } from './memoryViews.js'

test('facts panel exposes both facts.json and parsed facts_events.jsonl views', () => {
  const views = buildMemoryViews(
    'facts',
    {
      memory: { profile: { department: { value: 'Quality Assurance' } } },
      factsEvents: [
        {
          timestamp: '2026-04-10T12:34:56Z',
          category: 'profile',
          key: 'department',
          old_value: 'Regulatory Affairs',
          new_value: 'Quality Assurance',
        },
      ],
    },
    'facts.json'
  )

  assert.equal(views.length, 2)
  assert.equal(views[0].label, 'facts.json')
  assert.match(views[0].content, /Quality Assurance/)
  assert.equal(views[1].label, 'facts_events.jsonl')
  assert.match(views[1].content, /Regulatory Affairs/)
})

test('summary panel does not expose a facts event history view', () => {
  const views = buildMemoryViews(
    'summary',
    {
      memory: { summary: 'User is Andrew.' },
      factsEvents: [{ key: 'department' }],
    },
    'summary.json'
  )

  assert.deepEqual(views, [
    {
      label: 'summary.json',
      content: JSON.stringify({ summary: 'User is Andrew.' }, null, 2),
    },
  ])
})
