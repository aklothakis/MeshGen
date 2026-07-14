import React, { useState, useEffect } from 'react'
import { api } from '../api.js'
import { Slider, Btn, ErrorMsg, InfoRows } from './widgets.jsx'

export function FlowPanel({ geometryInfo, onDone, goNext }) {
  const refLen = geometryInfo?.length || geometryInfo?.reference_length || 2
  const [mach, setMach] = useState(geometryInfo?.mach || 6)
  const [alt, setAlt] = useState(30000)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [summary, setSummary] = useState(null)

  useEffect(() => { if (geometryInfo?.mach) setMach(geometryInfo.mach) }, [geometryInfo])

  async function compute() {
    setBusy(true); setErr(null)
    try {
      const res = await api.flow({ mach, altitude_m: alt, reference_length: refLen })
      setSummary(res)
      onDone({ ...res, mach, altitude_m: alt })
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <div className="panelbody">
      <h2>Flow conditions</h2>
      <p className="desc">
        Freestream state sets the Reynolds number and the wall-normal spacing:
        the first cell height is sized to a target y⁺ from these conditions.
      </p>
      <Slider label="Mach number" min={2} max={15} step={0.1} value={mach} onChange={setMach} />
      <Slider label="Altitude (m)" min={0} max={80000} step={500} value={alt} onChange={setAlt} />
      <Btn onClick={compute} busy={busy}>Compute conditions</Btn>
      <ErrorMsg msg={err} />

      {summary && (
        <div className="infobox">
          <div className="infotitle">Derived state (US Std Atmosphere)</div>
          <InfoRows rows={[
            ['Temperature', `${summary.temperature_K.toFixed(1)} K`],
            ['Pressure', `${summary.pressure_Pa.toFixed(1)} Pa`],
            ['Density', `${summary.density_kg_m3.toExponential(3)} kg/m³`],
            ['Velocity', `${summary.velocity_m_s.toFixed(0)} m/s`],
            ['Re (per m)', summary.reynolds_per_m.toExponential(3)],
            ['Re (body)', summary.reynolds_L.toExponential(3)],
            ['y⁺=1 cell', `${summary.first_cell_height_yplus1_m.toExponential(2)} m`],
            ['BL thickness', `${(summary.bl_thickness_m * 1000).toFixed(2)} mm`],
          ]} />
          <Btn ghost onClick={goNext}>Continue to mesh →</Btn>
        </div>
      )}
    </div>
  )
}
