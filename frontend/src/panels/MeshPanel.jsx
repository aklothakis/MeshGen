import React, { useState } from 'react'
import { api } from '../api.js'
import { Slider, Field, Btn, ErrorMsg } from './widgets.jsx'

export function MeshPanel({ session, flow, onDone, goNext }) {
  const [d, setD] = useState({
    n_normal: 48, farfield_radius_factor: 6, y_plus: 1,
    n_stream_blocks: 2, n_wrap_blocks: 2, n_wrap_mult: 1, smoothing_iters: 0,
  })
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [meshed, setMeshed] = useState(false)

  const set = (k) => (v) => setD({ ...d, [k]: v })

  async function build() {
    if (!session) { setErr('Generate geometry first.'); return }
    setBusy(true); setErr(null)
    try {
      const res = await api.mesh({
        session_id: session, ...d,
        use_flow: !!flow,
        mach: flow?.mach ?? undefined,
        altitude_m: flow?.altitude_m ?? 30000,
      })
      setMeshed(true)
      onDone(res)
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <div className="panelbody">
      <h2>Domain &amp; mesh</h2>
      <p className="desc">
        A body-fitted O-grid wraps each lens cross-section and extrudes to a
        circular farfield with hyperbolic-tangent wall clustering, then splits
        into connected structured blocks.
      </p>
      <Slider label="Wall-normal layers" min={12} max={128} step={1}
        value={d.n_normal} onChange={(v) => set('n_normal')(Math.round(v))} />
      <Slider label="Farfield radius ×" min={2} max={20} step={0.5}
        value={d.farfield_radius_factor} onChange={set('farfield_radius_factor')} />
      <Slider label="Target y⁺" min={0.1} max={30} step={0.1}
        value={d.y_plus} onChange={set('y_plus')} />
      <div className="row2">
        <Field label="Stream blocks" type="number" value={d.n_stream_blocks}
          onChange={(v) => set('n_stream_blocks')(Math.round(v))} />
        <Field label="Wrap blocks" type="number" value={d.n_wrap_blocks}
          onChange={(v) => set('n_wrap_blocks')(Math.round(v))} />
      </div>
      <Slider label="Wrap refinement ×" min={1} max={4} step={1}
        value={d.n_wrap_mult} onChange={(v) => set('n_wrap_mult')(Math.round(v))} />
      <Slider label="Elliptic (Winslow) smoothing" min={0} max={400} step={20}
        value={d.smoothing_iters} onChange={(v) => set('smoothing_iters')(Math.round(v))} />

      <Btn onClick={build} busy={busy}>Generate mesh</Btn>
      <ErrorMsg msg={err} />
      {!flow && <div className="notemsg">No flow set — wall spacing falls back to a length fraction.</div>}
      {meshed && <Btn ghost onClick={goNext}>Continue to export →</Btn>}
    </div>
  )
}
