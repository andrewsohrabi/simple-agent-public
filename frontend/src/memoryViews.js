export function buildMemoryViews(modeKey, panel, memoryLabel) {
  if (modeKey === 'none') {
    return []
  }

  const views = [
    {
      label: memoryLabel,
      content: JSON.stringify(panel.memory ?? {}, null, 2),
    },
  ]

  if (modeKey === 'facts') {
    views.push({
      label: 'facts_events.jsonl',
      content: JSON.stringify(panel.factsEvents ?? [], null, 2),
    })
  }

  return views
}
