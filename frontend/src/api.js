// Thin wrappers around the FastAPI backend.

async function post(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

export const api = {
  parametric: (params) => post('/api/geometry/parametric', params),
  flow: (params) => post('/api/flow', params),
  mesh: (params) => post('/api/mesh', params),

  async step(file, nSpan, nStream) {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('n_span', nSpan)
    fd.append('n_stream', nStream)
    const res = await fetch('/api/geometry/step', { method: 'POST', body: fd })
    if (!res.ok) {
      const d = await res.json().catch(() => ({}))
      throw new Error(d.detail || `STEP import failed (${res.status})`)
    }
    return res.json()
  },

  async export(sessionId, format) {
    const res = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, format }),
    })
    if (!res.ok) throw new Error(`Export failed (${res.status})`)
    const blob = await res.blob()
    const ext = format === 'plot3d' ? 'xyz' : format
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `waverider_mesh.${ext}`
    a.click()
    URL.revokeObjectURL(url)
  },
}
