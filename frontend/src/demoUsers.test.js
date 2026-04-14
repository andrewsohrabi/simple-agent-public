import test from 'node:test'
import assert from 'node:assert/strict'

import {
  FALLBACK_DEMO_USERS,
  getDemoUserByKey,
  seedDemoUserFallback,
} from './demoUsers.js'

test('fallback demo users include seed prompts for non-blank demo paths', () => {
  const blank = getDemoUserByKey(FALLBACK_DEMO_USERS, 'blank_demo')
  const regulatory = getDemoUserByKey(FALLBACK_DEMO_USERS, 'regulatory_lead')
  const personal = getDemoUserByKey(FALLBACK_DEMO_USERS, 'personal_preferences')

  assert.deepEqual(blank.seedPrompts, [])
  assert.ok(regulatory.seedPrompts.length >= 2)
  assert.ok(personal.seedPrompts.some(prompt => prompt.includes('preferred fruit')))
})

test('seedDemoUserFallback clears memory, seeds facts and summary, then reads snapshots', async () => {
  const calls = []
  const demoUser = getDemoUserByKey(FALLBACK_DEMO_USERS, 'personal_preferences')

  async function fetchImpl(url, options = {}) {
    calls.push({ url, options })

    if (url.includes('/memory?') && options.method === 'DELETE') {
      return { ok: true, json: async () => ({ memory: null, promptPreview: null }) }
    }

    if (url.endsWith('/chat')) {
      const body = JSON.parse(options.body)
      return {
        ok: true,
        json: async () => ({
          reply: `stored: ${body.messages.at(-1).content}`,
          memory: {},
          promptPreview: null,
          modelUsed: 'anthropic:test-model',
        }),
      }
    }

    if (url.includes('memoryType=facts')) {
      return {
        ok: true,
        json: async () => ({
          memory: { profile: { name: { value: 'Andrew' } } },
          factsEvents: [
            {
              key: 'name',
              old_value: null,
              new_value: 'Andrew',
            },
          ],
          factsEventsPath: 'memory_store/demo_personal_preferences/facts_events.jsonl',
          promptPreview: 'Long-Term Memory:\n- name: Andrew',
        }),
      }
    }

    if (url.includes('memoryType=summary')) {
      return {
        ok: true,
        json: async () => ({
          memory: { summary: 'User is Andrew. Their preferred fruit is mango.' },
          promptPreview: 'Long-Term Memory:\nUser is Andrew. Their preferred fruit is mango.',
        }),
      }
    }

    throw new Error(`Unexpected fetch call: ${url}`)
  }

  const result = await seedDemoUserFallback({
    apiUrl: 'http://localhost:8000',
    demoUser,
    model: 'anthropic:claude-haiku-4-5',
    fetchImpl,
  })

  assert.equal(result.userId, 'demo_personal_preferences')
  assert.equal(result.facts.profile.name.value, 'Andrew')
  assert.equal(result.factsEvents[0].new_value, 'Andrew')
  assert.equal(result.factsEventsPath, 'memory_store/demo_personal_preferences/facts_events.jsonl')
  assert.match(result.summary.summary, /mango/i)

  const deleteCalls = calls.filter(call => call.options.method === 'DELETE')
  const chatCalls = calls.filter(call => call.url.endsWith('/chat'))
  const getMemoryCalls = calls.filter(
    call => call.url.includes('/memory?') && (!call.options.method || call.options.method === 'GET')
  )

  assert.equal(deleteCalls.length, 2)
  assert.equal(chatCalls.length, demoUser.seedPrompts.length * 2)
  assert.equal(getMemoryCalls.length, 2)
  assert.ok(chatCalls.every(call => JSON.parse(call.options.body).model === 'anthropic:claude-haiku-4-5'))
})
