import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';

// ── API helpers ────────────────────────────────────────────────────────────
const API = '/api';

async function fetchMetrics() {
  const res = await fetch(`${API}/metrics`);
  if (!res.ok) throw new Error(`Backend error: ${res.status}`);
  return res.json();
}

async function fetchDecision() {
  const res = await fetch(`${API}/decision`);
  if (!res.ok) throw new Error(`Decision error: ${res.status}`);
  return res.json();
}

// ── Helpers ────────────────────────────────────────────────────────────────
function fmtProb(p) { return (p * 100).toFixed(1) + '%'; }
function fmtCpu(c)  { return (c * 100).toFixed(1) + '%'; }
function fmtMem(m)  { return m.toFixed(1) + ' GB'; }

function probColor(p) {
  if (p >= 0.7) return 'var(--accent-red)';
  if (p >= 0.45) return 'var(--accent-amber)';
  return 'var(--accent-green)';
}

function decisionKey(d) {
  if (!d) return 'STABLE';
  if (d.includes('SCALE UP'))   return 'SCALE_UP';
  if (d.includes('SCALE DOWN')) return 'SCALE_DOWN';
  return 'STABLE';
}

function timeLabel(ts) {
  try { return new Date(ts).toLocaleTimeString('en-US', { hour12: false }); }
  catch { return ts; }
}

// ── Custom chart tooltip ──────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip">
      <div className="label">{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="value" style={{ color: p.stroke || p.fill }}>
          {p.name}: {typeof p.value === 'number' ? (p.value * 100).toFixed(1) + '%' : p.value}
        </div>
      ))}
    </div>
  );
}

// ── Gauge SVG ─────────────────────────────────────────────────────────────
function GaugeChart({ value, color }) {
  const r = 70;
  const cx = 90, cy = 90;
  const circumference = Math.PI * r;  // half circle
  const offset = circumference * (1 - Math.min(Math.max(value, 0), 1));

  return (
    <div className="gauge-wrap">
      <svg width="180" height="110" className="gauge-svg">
        {/* Track */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke="#1e2d47"
          strokeWidth="14"
          strokeLinecap="round"
        />
        {/* Fill */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={`${circumference}`}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.8s ease, stroke 0.5s ease',
                   filter: `drop-shadow(0 0 8px ${color})` }}
        />
        {/* Centre text */}
        <text x={cx} y={cy - 10} textAnchor="middle" fill={color}
              fontSize="26" fontWeight="800" fontFamily="JetBrains Mono, monospace">
          {fmtProb(value)}
        </text>
        <text x={cx} y={cy + 10} textAnchor="middle" fill="#475569"
              fontSize="10" fontFamily="Inter, sans-serif" letterSpacing="1">
          FAILURE PROB
        </text>
      </svg>
    </div>
  );
}

// ── Decision Banner ────────────────────────────────────────────────────────
function DecisionBanner({ decision, metrics }) {
  const dk = decisionKey(decision?.decision);
  const icons = { STABLE: '⚖️', SCALE_UP: '🔺', SCALE_DOWN: '🔻' };
  const labels = { STABLE: 'System Stable', SCALE_UP: 'Scale Up Recommended', SCALE_DOWN: 'Scale Down Recommended' };

  return (
    <div className={`decision-banner ${dk} fade-in`}>
      <div className="decision-left">
        <div className={`decision-icon ${dk}`}>{icons[dk]}</div>
        <div>
          <div className="decision-label">AI Scaling Decision</div>
          <div className={`decision-text ${dk}`}>{labels[dk]}</div>
          {decision?.reason && (
            <div className="decision-reason">{decision.reason}</div>
          )}
        </div>
      </div>
      {metrics && (
        <div className="machine-badge">
          <div className="machine-id">🖥 {metrics.machineId}</div>
          <div className="machine-time">{timeLabel(metrics.timestamp)}</div>
        </div>
      )}
    </div>
  );
}

// ── Probability bar ────────────────────────────────────────────────────────
function ProbBar({ prob }) {
  const color = probColor(prob);
  return (
    <div className="prob-bar-wrap">
      <div className="prob-bar-bg" style={{ flex: 1 }}>
        <div className="prob-bar-fill"
          style={{ width: `${prob * 100}%`, background: color }} />
      </div>
      <span style={{ color, fontFamily: 'var(--font-mono)', fontSize: 12, width: 46, textAlign: 'right' }}>
        {fmtProb(prob)}
      </span>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const [metrics,  setMetrics]  = useState(null);
  const [decision, setDecision] = useState(null);
  const [history,  setHistory]  = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState(null);
  const [spinning, setSpinning] = useState(false);
  const [mlOnline, setMlOnline] = useState(null);
  const histRef = useRef([]);

  const refresh = useCallback(async (showSpin = true) => {
    if (showSpin) setSpinning(true);
    setError(null);
    try {
      const [m, d] = await Promise.all([fetchMetrics(), fetchDecision()]);
      setMetrics(m);
      setDecision(d);

      // Check ML service separately
      try {
        const r = await fetch('http://localhost:8000/health');
        setMlOnline(r.ok);
      } catch { setMlOnline(false); }

      // Append to history
      const point = {
        time: timeLabel(m.timestamp),
        failureProb: m.failureProbability,
        cpu: m.currentCpu,
        memory: m.assignedMemory / 32, // normalize to 0-1 for chart
      };
      histRef.current = [...histRef.current.slice(-29), point];
      setHistory([...histRef.current]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setSpinning(false);
    }
  }, []);

  // Initial load + auto-refresh every 6 seconds
  useEffect(() => {
    refresh(false);
    const id = setInterval(() => refresh(false), 6000);
    return () => clearInterval(id);
  }, [refresh]);

  if (loading) {
    return (
      <div className="dashboard">
        <Topbar spinning={false} onRefresh={() => {}} mlOnline={null} />
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <div className="loading-text">Connecting to backend...</div>
        </div>
      </div>
    );
  }

  const failColor = metrics ? probColor(metrics.failureProbability) : 'var(--accent-blue)';
  const dk = decisionKey(decision?.decision);

  return (
    <div className="dashboard">
      <Topbar spinning={spinning} onRefresh={() => refresh(true)} mlOnline={mlOnline} />

      <div className="main-content">
        {/* Error banner */}
        {error && (
          <div className="error-banner">
            ⚠️ {error} — Make sure the Spring Boot backend is running on port 8080.
          </div>
        )}

        {/* Decision banner */}
        {decision && metrics && (
          <DecisionBanner decision={decision} metrics={metrics} />
        )}

        {/* KPI row */}
        <div className="section-label">Live Metrics</div>
        <div className="kpi-row">
          <KpiCard
            label="Failure Probability"
            icon="🔥"
            value={metrics ? fmtProb(metrics.failureProbability) : '--'}
            sub={metrics ? `Confidence: ${metrics.confidence}` : ''}
            color={failColor}
            accentStyle={`linear-gradient(90deg, ${failColor}88, ${failColor}22)`}
          />
          <KpiCard
            label="Current CPU"
            icon="⚡"
            value={metrics ? fmtCpu(metrics.currentCpu) : '--'}
            sub="Avg CPU dist mean"
            color="var(--accent-cyan)"
            accentStyle="linear-gradient(90deg, var(--accent-cyan)88, var(--accent-cyan)22)"
          />
          <KpiCard
            label="Assigned Memory"
            icon="🧠"
            value={metrics ? fmtMem(metrics.assignedMemory) : '--'}
            sub="Allocated to task"
            color="var(--accent-purple)"
            accentStyle="linear-gradient(90deg, var(--accent-purple)88, var(--accent-purple)22)"
          />
          <KpiCard
            label="Prediction Status"
            icon={metrics?.failed ? '❌' : '✅'}
            value={metrics?.failed ? 'WILL FAIL' : 'HEALTHY'}
            sub={`Machine: ${metrics?.machineId || '--'}`}
            color={metrics?.failed ? 'var(--accent-red)' : 'var(--accent-green)'}
            accentStyle={metrics?.failed
              ? "linear-gradient(90deg, var(--accent-red)88, var(--accent-red)22)"
              : "linear-gradient(90deg, var(--accent-green)88, var(--accent-green)22)"}
          />
          <KpiCard
            label="Scaling Decision"
            icon={dk === 'SCALE_UP' ? '🔺' : dk === 'SCALE_DOWN' ? '🔻' : '⚖️'}
            value={decision?.decision?.replace(' (cooldown)', '') || 'STABLE'}
            sub={dk === 'STABLE' ? 'System balanced' : 'Action recommended'}
            color={dk === 'SCALE_UP' ? 'var(--accent-red)' : dk === 'SCALE_DOWN' ? 'var(--accent-blue)' : 'var(--accent-green)'}
            accentStyle={dk === 'SCALE_UP'
              ? "linear-gradient(90deg, var(--accent-red)88, var(--accent-red)22)"
              : dk === 'SCALE_DOWN'
              ? "linear-gradient(90deg, var(--accent-blue)88, var(--accent-blue)22)"
              : "linear-gradient(90deg, var(--accent-green)88, var(--accent-green)22)"}
          />
        </div>

        {/* Charts row */}
        <div className="two-col" style={{ marginBottom: 20 }}>
          {/* Failure prob over time */}
          <div className="card">
            <div className="card-title">📈 Failure Probability — 30 Observations</div>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={history} margin={{ top: 4, right: 8, left: -24, bottom: 0 }}>
                <defs>
                  <linearGradient id="failGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2d47" />
                <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 10 }} interval="preserveStartEnd" />
                <YAxis domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} tick={{ fill: '#475569', fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0.5} stroke="#ef444488" strokeDasharray="4 4" label={{ value: 'Threshold', fill: '#ef4444', fontSize: 10 }} />
                <Area type="monotone" dataKey="failureProb" name="Failure Prob" stroke="#ef4444" fill="url(#failGrad)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* CPU over time */}
          <div className="card">
            <div className="card-title">⚡ CPU Distribution Mean — 30 Observations</div>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={history} margin={{ top: 4, right: 8, left: -24, bottom: 0 }}>
                <defs>
                  <linearGradient id="cpuGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#06b6d4" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2d47" />
                <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 10 }} interval="preserveStartEnd" />
                <YAxis domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} tick={{ fill: '#475569', fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="cpu" name="CPU Mean" stroke="#06b6d4" fill="url(#cpuGrad)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Bottom row: Gauge + History table */}
        <div className="two-col">
          {/* Gauge */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            <div className="card-title" style={{ alignSelf: 'flex-start' }}>🎯 Current Failure Risk Gauge</div>
            {metrics && (
              <>
                <GaugeChart value={metrics.failureProbability} color={failColor} />
                <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
                  <span className={`confidence-badge ${metrics.confidence}`}>
                    ● {metrics.confidence} confidence
                  </span>
                  <span className="confidence-badge" style={{
                    background: metrics.failed ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                    color: metrics.failed ? 'var(--accent-red)' : 'var(--accent-green)',
                    border: `1px solid ${metrics.failed ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'}`
                  }}>
                    {metrics.failed ? '⚠ Failure Predicted' : '✓ No Failure'}
                  </span>
                </div>
                <div style={{ marginTop: 20, width: '100%' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    <MiniStat label="CPU Mean"    value={fmtCpu(metrics.currentCpu)}    color="var(--accent-cyan)" />
                    <MiniStat label="Memory"      value={fmtMem(metrics.assignedMemory)} color="var(--accent-purple)" />
                    <MiniStat label="Machine"     value={metrics.machineId}              color="var(--accent-blue)" />
                    <MiniStat label="Updated"     value={timeLabel(metrics.timestamp)}   color="var(--text-muted)" />
                  </div>
                </div>
              </>
            )}
          </div>

          {/* History table */}
          <div className="card scrollable">
            <div className="card-title">📋 Prediction History (Last {history.length})</div>
            {history.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 13, padding: '40px 0', textAlign: 'center' }}>
                Collecting data…
              </div>
            ) : (
              <table className="history-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Time</th>
                    <th>Failure Prob</th>
                    <th>CPU</th>
                  </tr>
                </thead>
                <tbody>
                  {[...history].reverse().slice(0, 15).map((row, i) => (
                    <tr key={i}>
                      <td style={{ color: 'var(--text-muted)' }}>{history.length - i}</td>
                      <td>{row.time}</td>
                      <td><ProbBar prob={row.failureProb} /></td>
                      <td style={{ color: 'var(--accent-cyan)' }}>{fmtCpu(row.cpu)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

function Topbar({ spinning, onRefresh, mlOnline }) {
  const backOnline = true; // we assume backend is up if we got this far
  return (
    <div className="topbar">
      <div className="topbar-brand">
        <div className="brand-icon">☁</div>
        <div>
          <div className="brand-name">CloudOpt</div>
          <div className="brand-sub">AI Resource Optimization</div>
        </div>
      </div>
      <div className="topbar-right">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div className={`status-dot ${backOnline ? '' : 'offline'}`} />
          <span className="status-text">Spring Boot :8080</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div className={`status-dot ${mlOnline === true ? '' : mlOnline === false ? 'offline' : ''}`} />
          <span className="status-text">FastAPI :8000</span>
        </div>
        <button className={`refresh-btn${spinning ? ' spinning' : ''}`} onClick={onRefresh}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
            <path d="M21 3v5h-5" />
            <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
            <path d="M8 16H3v5" />
          </svg>
          Refresh
        </button>
      </div>
    </div>
  );
}

function KpiCard({ label, icon, value, sub, color, accentStyle }) {
  return (
    <div className="kpi-card fade-in" style={{ '--kpi-accent': accentStyle }}>
      <div className="kpi-label">{icon} {label}</div>
      <div className="kpi-value" style={{ color }}>{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

function MiniStat({ label, value, color }) {
  return (
    <div style={{
      background: 'var(--bg-card2)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '10px 12px',
    }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 14, fontFamily: 'var(--font-mono)', color, fontWeight: 600 }}>
        {value}
      </div>
    </div>
  );
}
