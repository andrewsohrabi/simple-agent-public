import { buildApiMessages } from './panelState.js'

const PERSISTENT_MEMORY_TYPES = ['facts', 'summary']

export const FALLBACK_DEMO_USERS = [
  {
    key: 'blank_demo',
    label: 'Blank demo user',
    userId: 'demo_user',
    description: 'Fresh start with empty facts and summary memory.',
    bestFor: 'Manual freeform testing from a clean state.',
    seedPrompts: [],
  },
  {
    key: 'regulatory_lead',
    label: 'Regulatory lead',
    userId: 'demo_regulatory_lead',
    description: 'Preloaded regulatory affairs profile with current project context.',
    bestFor: 'Identity recall and project context recall.',
    seedPrompts: [
      "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a medical device company, focusing on 510(k) submissions for cardiac devices.",
      'Going forward, always give me concise bullet-point answers. No long paragraphs.',
      "I'm working on a 510(k) for a new catheter. The main challenge is choosing between two predicate devices.",
    ],
  },
  {
    key: 'personal_preferences',
    label: 'Personal preferences',
    userId: 'demo_personal_preferences',
    description: 'Simple personal identity and preference memory for plain-language demos.',
    bestFor: 'Personal preference recall and same-chat vs restart walkthroughs.',
    seedPrompts: [
      'My name is Andrew. My preferred fruit is mango.',
      'I prefer chocolate over peanut butter.',
    ],
  },
  {
    key: 'style_constrained',
    label: 'Style constrained',
    userId: 'demo_style_constrained',
    description: 'Quality Assurance profile with strong answer-format constraints.',
    bestFor: 'Preference application and contradiction-update demos.',
    seedPrompts: [
      'My name is Jordan Lee. I work in Quality Assurance.',
      'Going forward, always give me concise bullet-point answers. No long paragraphs.',
      "I'm working on CAPA audit preparation.",
    ],
  },
]

export function getDemoUserByKey(demoUsers, key) {
  return (demoUsers || []).find(user => user.key === key) ?? null
}

async function fetchJson(fetchImpl, url, options) {
  const response = await fetchImpl(url, options)
  if (!response.ok) {
    throw new Error(`${options?.method || 'GET'} ${url} failed with ${response.status}`)
  }
  return response.json()
}

async function clearModeMemory(fetchImpl, apiUrl, memoryType, userId) {
  const params = new URLSearchParams({ memoryType, userId })
  return fetchJson(fetchImpl, `${apiUrl}/memory?${params.toString()}`, {
    method: 'DELETE',
  })
}

async function readModeMemory(fetchImpl, apiUrl, memoryType, userId) {
  const params = new URLSearchParams({ memoryType, userId })
  return fetchJson(fetchImpl, `${apiUrl}/memory?${params.toString()}`)
}

async function seedModeWithPrompts(fetchImpl, apiUrl, memoryType, userId, seedPrompts, model) {
  let apiMessages = []

  for (const prompt of seedPrompts) {
    apiMessages = buildApiMessages(apiMessages, prompt)
    const data = await fetchJson(fetchImpl, `${apiUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildChatRequest(apiMessages, memoryType, userId, model)),
    })
    apiMessages = [...apiMessages, { role: 'assistant', content: data.reply }]
  }
}

function buildChatRequest(messages, memoryType, userId, model) {
  return {
    messages,
    memoryType,
    userId,
    model,
    factsExtractor: memoryType === 'facts' ? 'hybrid' : undefined,
  }
}

export async function seedDemoUserFallback({
  apiUrl,
  demoUser,
  model,
  fetchImpl = fetch,
}) {
  if (!demoUser) {
    throw new Error('Demo user configuration is missing')
  }

  await Promise.all(
    PERSISTENT_MEMORY_TYPES.map(memoryType =>
      clearModeMemory(fetchImpl, apiUrl, memoryType, demoUser.userId)
    )
  )

  if (Array.isArray(demoUser.seedPrompts) && demoUser.seedPrompts.length) {
    for (const memoryType of PERSISTENT_MEMORY_TYPES) {
      await seedModeWithPrompts(
        fetchImpl,
        apiUrl,
        memoryType,
        demoUser.userId,
        demoUser.seedPrompts,
        model
      )
    }
  }

  const [factsSnapshot, summarySnapshot] = await Promise.all(
    PERSISTENT_MEMORY_TYPES.map(memoryType =>
      readModeMemory(fetchImpl, apiUrl, memoryType, demoUser.userId)
    )
  )

  return {
    userId: demoUser.userId,
    demoUser: {
      key: demoUser.key,
      label: demoUser.label,
    },
    facts: factsSnapshot.memory,
    factsEvents: factsSnapshot.factsEvents,
    factsEventsPath: factsSnapshot.factsEventsPath,
    summary: summarySnapshot.memory,
    promptPreviews: {
      facts: factsSnapshot.promptPreview,
      summary: summarySnapshot.promptPreview,
    },
  }
}
