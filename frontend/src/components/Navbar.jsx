import React from 'react';
import { ShieldAlert, Activity, BarChart3, Cpu } from 'lucide-react';

export default function Navbar({ onOpenMetrics, onOpenResearch }) {
  return (
    <header className="navbar">
      <div className="brand-section">
        <div className="brand-icon">
          <ShieldAlert size={24} />
        </div>
        <div>
          <div className="brand-title">
            VoiceShield
            <span className="brand-badge">AI Forensic</span>
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            MODEL: VoiceShieldNet | Kaggle Benchmark
          </div>
        </div>
      </div>

      <div className="nav-actions">
        <div className="system-status-pill">
          <span className="status-dot"></span>
          <span>Engine: Local PyTorch</span>
        </div>

        <button 
          onClick={onOpenMetrics}
          style={{
            background: 'rgba(6, 182, 212, 0.1)',
            border: '1px solid rgba(6, 182, 212, 0.35)',
            color: 'var(--cyan-primary)',
            padding: '0.4rem 0.85rem',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.78rem',
            fontWeight: '600',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            fontFamily: 'var(--font-mono)'
          }}
        >
          <BarChart3 size={15} />
          <span>Evaluation Metrics</span>
        </button>

        <button 
          onClick={onOpenResearch}
          style={{
            background: 'rgba(139, 92, 246, 0.1)',
            border: '1px solid rgba(139, 92, 246, 0.35)',
            color: 'var(--purple-accent)',
            padding: '0.4rem 0.85rem',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.78rem',
            fontWeight: '600',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            fontFamily: 'var(--font-mono)'
          }}
        >
          <Cpu size={15} />
          <span>Research Architecture</span>
        </button>
      </div>
    </header>
  );
}
