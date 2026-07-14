import React, { useState } from 'react'
import { api } from '../api.js'
import { Btn, ErrorMsg } from './widgets.jsx'

const FORMATS = [
  { id: 'su2', name: 'SU2', ext: '.su2', desc: 'Native SU2 unstructured with tagged boundary markers' },
  { id: 'cgns', name: 'CGNS', ext: '.cgns', desc: 'Structured multiblock CGNS/HDF5 with ZoneBC' },
  { id: 'plot3d', name: 'Plot3D', ext: '.xyz', desc: 'Structured multiblock grid (formatted)' },
]

export function ExportPanel({ session, quality }) {
  const [busy, setBusy] = useState(null)
  const [err, setErr] = useState(null)

  async function download(fmt) {
    if (!session || !quality) { setErr('Generate a mesh first.'); return }
    setBusy(fmt); setErr(null)
    try {
      await api.export(session, fmt)
    } catch (e) { setErr(e.message) } finally { setBusy(null) }
  }

  return (
    <div className="panelbody">
      <h2>Export</h2>
      <p className="desc">Download the generated multiblock mesh for your solver.</p>

      {!quality && <div className="notemsg">No mesh yet — generate one in the previous step.</div>}

      <div className="formatlist">
        {FORMATS.map((f) => (
          <div className="formatcard" key={f.id}>
            <div className="fmthead">
              <span className="fmtname">{f.name}</span>
              <span className="fmtext">{f.ext}</span>
            </div>
            <div className="fmtdesc">{f.desc}</div>
            <Btn onClick={() => download(f.id)} busy={busy === f.id} disabled={!quality}>
              Download {f.name}
            </Btn>
          </div>
        ))}
      </div>
      <ErrorMsg msg={err} />

      {quality && (
        <div className="infobox">
          <div className="infotitle">Mesh contents</div>
          <div className="inforows">
            <div className="inforow"><span>Blocks</span><b>{quality.n_blocks}</b></div>
            <div className="inforow"><span>Cells</span><b>{quality.n_cells.toLocaleString()}</b></div>
            <div className="inforow"><span>Nodes</span><b>{quality.n_nodes.toLocaleString()}</b></div>
            <div className="inforow"><span>Boundary tags</span><b>wall · farfield · inflow · outflow</b></div>
          </div>
        </div>
      )}
    </div>
  )
}
