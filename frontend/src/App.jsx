import React, { useState, useCallback } from 'react'
import Viewport from './Viewport.jsx'
import { api } from './api.js'
import { GeometryPanel } from './panels/GeometryPanel.jsx'
import { FlowPanel } from './panels/FlowPanel.jsx'
import { MeshPanel } from './panels/MeshPanel.jsx'
import { ExportPanel } from './panels/ExportPanel.jsx'

const STEPS = [
  { id: 'geometry', label: 'Geometry', hint: 'Design or import the body' },
  { id: 'flow', label: 'Flow', hint: 'Freestream conditions' },
  { id: 'mesh', label: 'Domain & Mesh', hint: 'Generate the multiblock grid' },
  { id: 'export', label: 'Export', hint: 'SU2 · CGNS · Plot3D' },
]

export default function App() {
  const [active, setActive] = useState('geometry')
  const [session, setSession] = useState(null)
  const [geometry, setGeometry] = useState(null)
  const [geometryInfo, setGeometryInfo] = useState(null)
  const [flow, setFlow] = useState(null)
  const [mesh, setMesh] = useState(null)
  const [quality, setQuality] = useState(null)
  const [viewMode, setViewMode] = useState('geometry')

  const done = {
    geometry: !!geometry,
    flow: !!flow,
    mesh: !!mesh,
    export: !!mesh,
  }

  const onGeometry = useCallback((res) => {
    setSession(res.session_id)
    setGeometry(res.geometry)
    setGeometryInfo(res.info)
    setMesh(null)
    setQuality(null)
    setViewMode('geometry')
  }, [])

  const onMesh = useCallback((res) => {
    setMesh(res.mesh)
    setQuality(res.quality)
    setViewMode('mesh')
  }, [])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark">◹</span>
          <div>
            <div className="title">Waverider MeshGen</div>
            <div className="subtitle">Hypersonic multiblock structured mesh generator</div>
          </div>
        </div>
        <div className="topmeta">
          {session && <span className="pill">session {session.slice(0, 8)}</span>}
        </div>
      </header>

      <nav className="stepper">
        {STEPS.map((s, i) => (
          <button
            key={s.id}
            className={`step ${active === s.id ? 'active' : ''} ${done[s.id] ? 'done' : ''}`}
            onClick={() => setActive(s.id)}
          >
            <span className="idx">{done[s.id] ? '✓' : i + 1}</span>
            <span className="steptext">
              <span className="steplabel">{s.label}</span>
              <span className="stephint">{s.hint}</span>
            </span>
            {i < STEPS.length - 1 && <span className="arrow">→</span>}
          </button>
        ))}
      </nav>

      <main className="workspace">
        <aside className="panel">
          {active === 'geometry' && (
            <GeometryPanel onDone={onGeometry} goNext={() => setActive('flow')} />
          )}
          {active === 'flow' && (
            <FlowPanel geometryInfo={geometryInfo} onDone={setFlow}
              goNext={() => setActive('mesh')} />
          )}
          {active === 'mesh' && (
            <MeshPanel session={session} flow={flow} onDone={onMesh}
              goNext={() => setActive('export')} />
          )}
          {active === 'export' && (
            <ExportPanel session={session} quality={quality} />
          )}
        </aside>

        <section className="stage">
          <div className="viewtoggle">
            <button className={viewMode === 'geometry' ? 'on' : ''}
              onClick={() => setViewMode('geometry')} disabled={!geometry}>
              Geometry
            </button>
            <button className={viewMode === 'mesh' ? 'on' : ''}
              onClick={() => setViewMode('mesh')} disabled={!mesh}>
              Mesh
            </button>
          </div>

          <Viewport geometry={geometry} mesh={mesh} showMode={viewMode} />

          {!geometry && (
            <div className="emptyhint">Start by generating or importing geometry →</div>
          )}

          {quality && viewMode === 'mesh' && (
            <div className="qualitycard">
              <div className="qhead">
                Mesh quality {quality.valid
                  ? <span className="ok">● valid</span>
                  : <span className="bad">● invalid</span>}
              </div>
              <div className="qgrid">
                <Stat label="Blocks" value={quality.n_blocks} />
                <Stat label="Cells" value={quality.n_cells.toLocaleString()} />
                <Stat label="Nodes" value={quality.n_nodes.toLocaleString()} />
                <Stat label="Neg. cells" value={quality.n_negative_cells}
                  bad={quality.n_negative_cells > 0} />
                <Stat label="Mean ortho" value={`${(quality.mean_orthogonality_deg ?? 0).toFixed(1)}°`} />
                <Stat label="Min ortho" value={`${quality.min_orthogonality_deg.toFixed(1)}°`} />
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

function Stat({ label, value, bad }) {
  return (
    <div className="stat">
      <div className="statlabel">{label}</div>
      <div className={`statvalue ${bad ? 'bad' : ''}`}>{value}</div>
    </div>
  )
}
