import React, { useState } from 'react'
import { api } from '../api.js'
import { Field, Slider, Btn, ErrorMsg, InfoRows } from './widgets.jsx'

export function GeometryPanel({ onDone, goNext }) {
  const [tab, setTab] = useState('parametric')
  const [p, setP] = useState({
    mach: 6, shock_angle_deg: 14, length: 2, height_ratio: 0.35,
    width_trim: 0.98, n_span: 61, n_stream: 41, gamma: 1.4,
  })
  const [nSpan, setNSpan] = useState(61)
  const [nStream, setNStream] = useState(41)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [info, setInfo] = useState(null)

  const set = (k) => (v) => setP({ ...p, [k]: v })

  async function generate() {
    setBusy(true); setErr(null)
    try {
      const res = await api.parametric(p)
      setInfo(res.info)
      onDone(res)
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  async function upload(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true); setErr(null)
    try {
      const res = await api.step(file, nSpan, nStream)
      setInfo(res.info)
      onDone(res)
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <div className="panelbody">
      <h2>Geometry</h2>
      <div className="tabs">
        <button className={tab === 'parametric' ? 'on' : ''} onClick={() => setTab('parametric')}>
          Parametric waverider
        </button>
        <button className={tab === 'step' ? 'on' : ''} onClick={() => setTab('step')}>
          Import STEP
        </button>
      </div>

      {tab === 'parametric' && (
        <>
          <p className="desc">
            The compression surface is built by tracing streamlines through the
            Taylor–Maccoll conical field, so the leading edge rides its own shock.
          </p>
          <Slider label="Mach number" min={2} max={15} step={0.1}
            value={p.mach} onChange={set('mach')} />
          <Slider label="Shock angle (deg)" min={5} max={45} step={0.5}
            value={p.shock_angle_deg} onChange={set('shock_angle_deg')} />
          <Slider label="Length (m)" min={0.5} max={10} step={0.1}
            value={p.length} onChange={set('length')} />
          <Slider label="Height ratio (d/Rₛ)" min={0.1} max={0.8} step={0.01}
            value={p.height_ratio} onChange={set('height_ratio')} />
          <div className="row2">
            <Field label="Span nodes" type="number" value={p.n_span}
              onChange={(v) => set('n_span')(Math.round(v))} />
            <Field label="Stream nodes" type="number" value={p.n_stream}
              onChange={(v) => set('n_stream')(Math.round(v))} />
          </div>
          <Btn onClick={generate} busy={busy}>Generate waverider</Btn>
        </>
      )}

      {tab === 'step' && (
        <>
          <p className="desc">
            Import a clean STEP solid (ISO 10303). It is tessellated with
            OpenCASCADE and re-gridded into structured lens cross-sections.
          </p>
          <div className="row2">
            <Field label="Span nodes" type="number" value={nSpan}
              onChange={(v) => setNSpan(Math.round(v))} />
            <Field label="Stream nodes" type="number" value={nStream}
              onChange={(v) => setNStream(Math.round(v))} />
          </div>
          <label className="filedrop">
            <input type="file" accept=".step,.stp,.STEP,.STP" onChange={upload} />
            <span>{busy ? 'Importing…' : 'Choose a .step / .stp file'}</span>
          </label>
        </>
      )}

      <ErrorMsg msg={err} />

      {info && (
        <div className="infobox">
          <div className="infotitle">Design summary</div>
          <InfoRows rows={formatInfo(info)} />
          <Btn ghost onClick={goNext}>Continue to flow →</Btn>
        </div>
      )}
    </div>
  )
}

function formatInfo(info) {
  const f = (x, d = 3) => (typeof x === 'number' ? x.toFixed(d) : x)
  if (info.source === 'step') {
    return [
      ['Source', info.filename || 'STEP'],
      ['Reference length', `${f(info.reference_length)} m`],
      ['Triangles', info.n_triangles],
    ]
  }
  return [
    ['Cone angle', `${f(info.cone_angle_deg, 2)}°`],
    ['Surface Mach', f(info.surface_mach, 2)],
    ['Span', `${f(info.span)} m`],
    ['Volume', `${f(info.volume, 4)} m³`],
    ['Planform area', `${f(info.planform_area)} m²`],
    ['Vol. efficiency', f(info.volumetric_efficiency, 3)],
  ]
}
