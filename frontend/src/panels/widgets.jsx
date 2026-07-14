import React from 'react'

export function Field({ label, value, onChange, type = 'text', ...rest }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(type === 'number' ? Number(e.target.value) : e.target.value)}
        {...rest}
      />
    </label>
  )
}

export function Slider({ label, value, onChange, min, max, step }) {
  return (
    <label className="slider">
      <div className="sliderhead">
        <span>{label}</span>
        <span className="sliderval">{value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

export function Btn({ children, onClick, busy, ghost, disabled }) {
  return (
    <button className={`btn ${ghost ? 'ghost' : ''}`} onClick={onClick} disabled={busy || disabled}>
      {busy ? <span className="spinner" /> : children}
    </button>
  )
}

export function ErrorMsg({ msg }) {
  if (!msg) return null
  return <div className="errmsg">⚠ {msg}</div>
}

export function InfoRows({ rows }) {
  return (
    <div className="inforows">
      {rows.map(([k, v]) => (
        <div className="inforow" key={k}>
          <span>{k}</span><b>{v}</b>
        </div>
      ))}
    </div>
  )
}
