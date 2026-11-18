import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  AlertCircle, ChevronDown, ChevronLeft, ChevronRight, ClipboardList, FileImage,
  Filter, HeartPulse, ImagePlus, Layers2, Menu, Microscope, ScanLine, Search,
  ShieldCheck, SlidersHorizontal, Sparkles, X,
} from 'lucide-react'
import './style.css'
import './status.css'
import './palette.css'

type CaseItem = { image_id: string; dataset: string; status: string; u_fg: number | null; u_mean: number | null; count_fg: number; dsc: number | null; nsd: number | null }
type Summary = { case_count: number; datasets: string[]; accepted_count: number; review_count: number; threshold: number; empty_mask_threshold: number }
type Result = { case: CaseItem & { score: number | null; threshold: number; margin: number | null; reason: string; clinical_note: string; source: string }; images: Record<string, string>; neighbors: { image_id: string; dataset: string; similarity: number; u_fg: number | null; dsc: number | null }[]; retrieval_metrics: Record<string, number> | null }

const api = async <T,>(path: string): Promise<T> => {
  const response = await fetch(path)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}
const fmt = (n: number | null | undefined, digits = 4) => n == null || !Number.isFinite(n) ? '—' : n.toFixed(digits)
const imageUrl = (dataset: string, id: string, kind: string) => `/api/image?dataset=${encodeURIComponent(dataset)}&image_id=${encodeURIComponent(id)}&kind=${kind}`

function App() {
  const [summary, setSummary] = useState<Summary | null>(null)
  const [cases, setCases] = useState<CaseItem[]>([])
  const [selected, setSelected] = useState<CaseItem | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [dataset, setDataset] = useState('All cohorts')
  const [statusFilter, setStatusFilter] = useState('All cases')
  const [search, setSearch] = useState('')
  const [activeView, setActiveView] = useState<'overlay' | 'uncertainty'>('overlay')
  const [opacity, setOpacity] = useState(48)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    Promise.all([api<Summary>('/api/summary'), api<{ items: CaseItem[] }>('/api/cases')])
      .then(([s, c]) => { setSummary(s); setCases(c.items); setSelected(c.items[0] ?? null) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!selected) return
    let cancelled = false
    setResult(null)
    api<Result>(`/api/cases/${encodeURIComponent(selected.dataset)}/${encodeURIComponent(selected.image_id)}/analyze`)
      .then(r => { if (!cancelled) setResult(r) })
      .catch(e => { if (!cancelled) setError(e.message) })
    return () => { cancelled = true }
  }, [selected])

  const visibleCases = useMemo(() => cases.filter(c =>
    (dataset === 'All cohorts' || c.dataset === dataset) &&
    (statusFilter === 'All cases' || c.status === (statusFilter === 'Needs review' ? 'Rejected' : 'Accepted')) &&
    c.image_id.toLowerCase().includes(search.toLowerCase()),
  ), [cases, dataset, statusFilter, search])
  const pendingCount = cases.filter(c => c.status === 'Rejected').length
  const activeIndex = selected ? visibleCases.findIndex(c => c.dataset === selected.dataset && c.image_id === selected.image_id) : -1
  const apiReady = Boolean(summary) && !error

  const selectCase = (item: CaseItem) => setSelected(item)
  const stepCase = (direction: number) => {
    if (!visibleCases.length) return
    selectCase(visibleCases[(Math.max(0, activeIndex) + direction + visibleCases.length) % visibleCases.length])
  }

  return <div className="app-shell">
    <aside className={`sidebar ${menuOpen ? 'sidebar-open' : ''}`}>
      <div className="brand"><div className="brand-icon"><HeartPulse size={21} strokeWidth={2.2}/></div><div><strong>MedCLIPSeg<span>++</span></strong><small>IMAGING RESEARCH</small></div><button className="icon-button mobile-close" onClick={() => setMenuOpen(false)} aria-label="Close menu"><X size={17}/></button></div>
      <div className="workspace-switch"><div className="workspace-mark">M</div><div className="workspace-copy"><b>Case review</b><small>BUSI + external cohorts</small></div></div>
      <div className="nav-caption">CASE LIST</div>
      <nav className="nav-list">
        <NavButton icon={<ScanLine/>} label="All evaluated cases" active={statusFilter === 'All cases'} onClick={() => setStatusFilter('All cases')}/>
        <NavButton icon={<ClipboardList/>} label="Needs review" count={pendingCount || undefined} active={statusFilter === 'Needs review'} onClick={() => setStatusFilter('Needs review')}/>
      </nav>
      <div className="sidebar-bottom"><div className="status-card"><div className="status-card-icon"><ShieldCheck size={16}/></div><div><b>Research prototype</b><p>Precomputed cases only</p></div></div><a className="methods-link" href="https://github.com/Fergarciamedina/MedCLIPSeg-plus-plus/blob/main/uncertainty_rejection/research/EXPERIMENTAL_PROTOCOL.md" target="_blank" rel="noreferrer">Study protocol <ChevronRight size={14}/></a></div>
    </aside>
    {menuOpen && <button className="scrim" onClick={() => setMenuOpen(false)} aria-label="Close navigation"/>}
    <main className="main-area">
      <header className="topbar"><div className="topbar-left"><button className="icon-button menu-toggle" onClick={() => setMenuOpen(true)} aria-label="Open case list"><Menu size={19}/></button><span className="crumb-muted">MedCLIPSeg++</span><ChevronRight size={14}/><span>Case review</span></div><div className="topbar-right"><div className={`api-status ${apiReady ? '' : 'api-offline'}`}><i/> {apiReady ? 'Benchmark service ready' : loading ? 'Connecting' : 'Service unavailable'}</div></div></header>
      <div className="page-content">
        <div className="page-heading"><div><div className="eyebrow"><span className="eyebrow-mark"><Microscope size={13}/></span> PRECOMPUTED STUDY CASES</div><h1>Case review</h1><p className="heading-sub">Review the scan, segmentation overlay, predictive uncertainty, policy result, and PABP references.</p></div><div className="heading-actions"><span className="not-evaluated-chip"><i/> New scans: Not Evaluated</span></div></div>

        {error && <div className="error-banner"><AlertCircle size={18}/><div><b>Dashboard could not load its benchmark services</b><p>{error}</p><small>Install the application requirements and check that MEDCLIPSEG_DATA points to the repository's data directory.</small></div></div>}

        <div className="content-grid">
          <section className="panel case-panel">
            <div className="panel-heading"><div><div className="panel-title">Study cases</div><div className="panel-sub">Select an evaluated ultrasound</div></div><span className="count-pill">{visibleCases.length}</span></div>
            <div className="case-controls"><label className="search-box"><Search size={15}/><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search case ID"/></label><label className="select-wrap"><Filter size={14}/><select value={dataset} onChange={e => setDataset(e.target.value)}><option>All cohorts</option>{summary?.datasets.map(d => <option key={d}>{d}</option>)}</select><ChevronDown size={13}/></label><label className="select-wrap status-select"><SlidersHorizontal size={14}/><select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}><option>All cases</option><option>Accepted</option><option>Needs review</option></select><ChevronDown size={13}/></label></div>
            <div className="case-list-head"><span>CASE / COHORT</span><span>POLICY</span></div>
            <div className="case-list">{loading ? <div className="list-empty">Loading benchmark manifest…</div> : visibleCases.length ? visibleCases.map(c => <button key={`${c.dataset}:${c.image_id}`} className={`case-row ${selected?.dataset === c.dataset && selected.image_id === c.image_id ? 'case-row-active' : ''}`} onClick={() => selectCase(c)}><div className="case-thumb"><img src={imageUrl(c.dataset, c.image_id, 'original')} loading="lazy"/><span/></div><div className="case-row-info"><b>{c.image_id}</b><small>{c.dataset} <span>·</span> {c.count_fg ? 'Lesion mask' : 'Empty mask'}</small></div><StatusPill status={c.status}/></button>) : <div className="list-empty">No cases match these filters.</div>}</div>
            <div className="case-panel-footer"><span>Showing {visibleCases.length} of {cases.length} cases</span><div><button className="mini-button" onClick={() => stepCase(-1)} aria-label="Previous case"><ChevronLeft size={15}/></button><button className="mini-button" onClick={() => stepCase(1)} aria-label="Next case"><ChevronRight size={15}/></button></div></div>
          </section>

          <section className="review-column">
            <div className="panel case-header-panel"><div className="case-title-block"><div className="case-icon"><FileImage size={18}/></div><div><div className="case-title-line"><h2>{selected?.image_id ?? (loading ? 'Loading case…' : 'No case selected')}</h2>{result && <StatusPill status={result.case.status}/>}</div><p>{selected?.dataset ?? 'Benchmark cohort'} <span className="bullet">·</span> Precomputed benchmark result</p></div></div><button className="button button-outline" onClick={() => stepCase(1)} disabled={!visibleCases.length}>Next case <ChevronRight size={15}/></button></div>

            <div className="image-workspace">
              <div className="image-toolbar"><div className="toolbar-tabs"><button className={activeView === 'overlay' ? 'toolbar-tab active' : 'toolbar-tab'} onClick={() => setActiveView('overlay')}><Layers2 size={15}/> Mask overlay</button><button className={activeView === 'uncertainty' ? 'toolbar-tab active' : 'toolbar-tab'} onClick={() => setActiveView('uncertainty')}><Sparkles size={15}/> Uncertainty</button></div><div className="image-tools"><span className="image-dimension">PRECOMPUTED</span></div></div>
              <div className={`image-panels ${activeView === 'uncertainty' ? 'uncertainty-mode' : ''}`}>
                <ImagePanel title="Original ultrasound" subtitle="Source image" image={selected ? imageUrl(selected.dataset, selected.image_id, 'original') : ''} badge="INPUT"/>
                {activeView === 'overlay' ? <ImagePanel title="Predicted segmentation" subtitle="Mask overlay" image={selected ? imageUrl(selected.dataset, selected.image_id, 'original') : ''} overlay={selected ? imageUrl(selected.dataset, selected.image_id, 'mask') : ''} opacity={opacity} badge="MODEL OUTPUT"/> : <ImagePanel title="Predictive uncertainty" subtitle="Precomputed entropy visualization" image={selected ? imageUrl(selected.dataset, selected.image_id, 'uncertainty') : ''} heat badge="MODEL OUTPUT"/>}
              </div>
              {activeView === 'overlay' && <div className="opacity-control"><span>Overlay opacity</span><input type="range" min="0" max="100" value={opacity} onChange={e => setOpacity(Number(e.target.value))}/><b>{opacity}%</b><span className="opacity-divider"/><span className="legend"><i/> Predicted foreground</span></div>}
              {activeView === 'uncertainty' && <div className="heatmap-legend"><span>Lower predictive entropy</span><div className="heat-gradient"/><span>Higher predictive entropy</span><small>Displayed from the available precomputed uncertainty image.</small></div>}
            </div>

            <div className="detail-grid">
              <section className="panel triage-panel"><div className="panel-heading detail-heading"><div><div className="panel-title">Uncertainty triage</div><div className="panel-sub">Frozen selective prediction policy</div></div><ShieldCheck size={17} className="heading-symbol"/></div>
                <div className="triage-status-row"><div className={`triage-status-icon ${result?.case.status === 'Rejected' ? 'review' : ''}`}>{result?.case.status === 'Rejected' ? <AlertCircle size={19}/> : <ShieldCheck size={19}/>}</div><div><b>{result ? (result.case.status === 'Rejected' ? 'Needs review' : 'Within study policy') : 'Loading policy result'}</b><small>{result ? (result.case.count_fg ? 'Foreground uncertainty rule' : 'Empty-mask fallback rule') : 'Applying precomputed case policy'}</small></div><span className="policy-tag">POLICY</span></div>
                <div className="metric-pair"><Metric label={result?.case.count_fg ? 'U_fg score' : 'U_mean score'} value={fmt(result?.case.score, 5)} note="Predictive entropy"/><Metric label="Applied threshold" value={fmt(result?.case.threshold, 5)} note="Frozen in study protocol"/></div>
                <div className="threshold-track"><div className="track-label"><span>Score vs threshold</span><span>{result?.case.margin != null ? (result.case.margin >= 0 ? 'Within policy' : 'Above threshold') : '—'}</span></div><div className="track"><div className="track-fill" style={{width: result?.case.score != null && result.case.threshold ? `${Math.min(100, Math.max(4, result.case.score / result.case.threshold * 100))}%` : '0%'}}/><i className="track-marker"/></div><div className="track-values"><span>0</span><span>τ = {fmt(result?.case.threshold, 5)}</span></div></div>
                <div className="triage-note"><AlertCircle size={15}/><span>{result ? (result.case.count_fg > 0 ? 'U_fg summarizes entropy within the predicted foreground. It does not guarantee that the mask is accurate.' : 'The empty-mask fallback uses U_mean. A low value does not establish that the scan is lesion-free.') : 'Loading the precomputed policy result.'}</span></div>
              </section>

              <section className="panel retrieval-panel"><div className="panel-heading detail-heading"><div><div className="panel-title">Similar benchmark cases</div><div className="panel-sub">PABP retrieval · accepted references</div></div><span className="retrieval-count">{result?.neighbors.length ?? '—'} <small>matches</small></span></div>
                {result?.case.status === 'Rejected' ? <div className="retrieval-locked"><div className="lock-icon"><ShieldCheck size={19}/></div><b>Retrieval paused by policy</b><p>This case is flagged for review. Reference retrieval is gated to accepted cases in the existing workflow.</p></div> : result?.neighbors.length ? <div className="neighbor-list">{result.neighbors.slice(0, 4).map((n, i) => <div className="neighbor-row" key={`${n.dataset}:${n.image_id}`}><div className="neighbor-thumb"><img src={imageUrl(n.dataset, n.image_id, 'original')} loading="lazy"/></div><div className="neighbor-main"><b>{n.image_id}</b><small>{n.dataset} <span>·</span> Similarity rank {i + 1}</small><div className="similarity-track"><i style={{width: `${Math.max(2, Math.min(100, n.similarity * 100))}%`}}/></div></div><div className="neighbor-score"><b>{fmt(n.similarity, 3)}</b><small>similarity</small></div></div>)}</div> : <div className="retrieval-empty"><div className="lock-icon"><Search size={19}/></div><b>{result ? 'No reference cases returned' : 'Preparing retrieval'}</b><p>{result ? 'There are no eligible similar cases to show for this selection.' : 'Loading precomputed embeddings and search index.'}</p></div>}
                {result?.retrieval_metrics && <div className="pabp-metrics"><div><span>Blocks pruned</span><b>{result.retrieval_metrics.blocks_pruned_pabp ?? '—'}</b></div><div><span>Vector distances</span><b>{result.retrieval_metrics.vector_distances_computed ?? '—'}</b></div><div><span>Candidate savings</span><b>{fmt((result.retrieval_metrics.overall_candidate_saving_ratio ?? 0) * 100, 1)}%</b></div></div>}
                <div className="retrieval-footer"><span><i className="live-dot"/> Gated by frozen policy</span><a className="text-button" href="https://github.com/Fergarciamedina/MedCLIPSeg-plus-plus/blob/main/uncertainty_rejection/research/EXPERIMENTAL_PROTOCOL.md" target="_blank" rel="noreferrer">Study protocol <ChevronRight size={13}/></a></div>
              </section>
            </div>
            <div className="research-note"><ShieldCheck size={15}/><span><b>Research review only.</b> These are precomputed benchmark outputs. Uncertainty and policy status do not guarantee segmentation accuracy.</span></div>
          </section>
        </div>
        <footer className="page-footer"><span>MedCLIPSeg++ <b>·</b> Reliability-aware segmentation research</span><span>Local research environment <i/></span></footer>
      </div>
    </main>
  </div>
}

function NavButton({ icon, label, active, count, onClick }: { icon: React.ReactNode; label: string; active: boolean; count?: number; onClick: () => void }) {
  return <button className={`nav-item ${active ? 'nav-active' : ''}`} onClick={onClick}><span className="nav-icon">{icon}</span><span>{label}</span>{count ? <small>{count}</small> : null}</button>
}
function StatusPill({ status }: { status: string }) {
  const review = status === 'Rejected'
  return <span className={`status-pill ${review ? 'status-review' : 'status-accepted'}`}><i/>{review ? 'Review' : 'Within policy'}</span>
}
function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="metric-cell"><span>{label}</span><b>{value}</b><small>{note}</small></div>
}
function ImagePanel({ title, subtitle, image, overlay, opacity = 0, heat = false, badge }: { title: string; subtitle: string; image: string; overlay?: string; opacity?: number; heat?: boolean; badge: string }) {
  return <div className="image-panel"><div className="image-panel-head"><div><b>{title}</b><small>{subtitle}</small></div><span>{badge}</span></div><div className={`image-canvas ${heat ? 'image-heat' : ''}`}>
    {image ? <><img className="canvas-image" src={image} alt={title}/>{overlay && <img className="canvas-overlay" src={overlay} alt="Predicted segmentation mask" style={{opacity: opacity / 100, mixBlendMode: 'screen'}}/>}{heat && <div className="image-corner-label">ENTROPY MAP</div>}</> : <div className="image-placeholder"><ImagePlus size={25}/><span>Select a benchmark case</span></div>}
    <div className="canvas-corner">US <span>·</span> 2D</div></div><div className="image-panel-foot"><span><i/> {overlay ? 'Mask overlay' : heat ? 'Predictive entropy' : 'Original image'}</span><span>Precomputed</span></div></div>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>)
