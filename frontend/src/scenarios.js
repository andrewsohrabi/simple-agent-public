export const SCENARIOS = {
  identity_recall: {
    label: 'Identity recall',
    description: 'Stores identity and work context, overwrites name and focus in later turns, then restarts and asks for both the current state and the prior focus.',
    tests: 'Whether the system updates current identity/work facts correctly and can recover prior focus from temporal facts history after restart.',
    setup: [
      "Hi, I'm Dr. Sarah Chen. I work in Regulatory Affairs at a medical device company, focusing on 510(k) submissions for cardiac devices.",
      'I got married and my last name is now Doe.',
      'Now I focus on 510(k) submissions for brain implants.',
    ],
    recall: [
      'What do you remember about me and my work now?',
      'What did I focus on before my current focus area?',
    ],
    recommendedUsers: ['blank_demo', 'regulatory_lead'],
  },
  preference_application: {
    label: 'Preference application',
    description: 'Stores one response style, overwrites it with a visibly different one, then restarts and checks both the current behavior and the prior style preference.',
    tests: 'Whether the latest response-style preference is applied after restart and whether the earlier style remains available through targeted temporal retrieval.',
    setup: [
      'Going forward, always give me three-line haiku answers.',
      'Update that: start every answer with an ALL-CAPS summary line, then give me concise bullet-point answers. No haikus.',
    ],
    recall: [
      'Explain the key considerations for predicate device selection in a 510(k).',
      'What response style did I ask for before my current one?',
    ],
    recommendedUsers: ['style_constrained', 'blank_demo'],
  },
  project_context_recall: {
    label: 'Project context recall',
    description: 'Stores one project and challenge, overwrites both with a new workstream, then restarts and checks the current project plus the prior one.',
    tests: 'Whether the latest project context is preserved across sessions and whether the earlier project remains available through temporal facts history.',
    setup: [
      "I'm working on a 510(k) for a new catheter. The main challenge is choosing between two predicate devices.",
      "Now I'm working on a 510(k) for a brain implant. The main challenge is building the clinical evidence plan.",
    ],
    recall: [
      'Can you help me think through next steps for my current project?',
      'What project was I working on before my current one?',
    ],
    recommendedUsers: ['regulatory_lead', 'blank_demo'],
  },
  contradiction_update: {
    label: 'Contradiction update',
    description: 'Stores one department, updates it in a later turn, then restarts and checks both the latest value and the prior one.',
    tests: 'Whether latest-wins updating is handled correctly and whether prior-state temporal memory remains available after restart.',
    setup: ['I work in Regulatory Affairs.', 'Actually, I just transferred to Quality Assurance.'],
    recall: ['What department am I in now?', 'What was my job before?'],
    recommendedUsers: ['style_constrained', 'blank_demo'],
  },
  personal_preference_recall: {
    label: 'Personal preference recall',
    description: 'Stores a simple identity and preference, overwrites the preference later in the session, then restarts and checks both the current and prior preference.',
    tests: 'Whether ordinary non-work facts update cleanly across sessions and whether the prior preference remains available through temporal facts history.',
    setup: [
      'My name is Andrew. My preferred fruit is mango.',
      'Actually, my preferred fruit is pear now.',
    ],
    recall: [
      'What is my name and preferred fruit now?',
      'What fruit did I prefer before my current one?',
    ],
    recommendedUsers: ['personal_preferences', 'blank_demo'],
  },
}
