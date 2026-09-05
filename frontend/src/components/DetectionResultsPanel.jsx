import React from 'react';
import { AlertCircle, CheckCircle2, AlertTriangle, ShieldAlert, Cpu, Lock, Layers } from 'lucide-react';

export default function DetectionResultsPanel({ results }) {
  if (!results) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3.5rem 1.5rem' }}>
        <ShieldAlert size={48} style={{ color: 'var(--text-dim)', margin: '0 auto 1rem auto' }} />
        <h3 style={{ fontFamily: 'var(--font-display)', color: 'var(--text-muted)' }}>
          Awaiting Speech Input
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-dim)', maxWidth: '400px', margin: '0.5rem auto 0 auto' }}>
          Select a 1-click demo sample, upload an audio file, or record speech with your microphone, then click Execute Analysis.
        </p>
      </div>
    );
  }

  const isHighRisk = results.voice_risk_score >= 70;
  const isMedRisk = results.voice_risk_score >= 40 && results.voice_risk_score < 70;
  const alertClass = isHighRisk ? 'alert-high' : (isMedRisk ? 'alert-medium' : 'alert-low');

  const aiPct = Math.round(results.ai_generated_probability * 100);
  const humanPct = Math.round(results.human_probability * 100);

  return (
    <div className="results-container">
      {/* 1. Security Alert & Recommended Action Card */}
      <div className={`security-alert-card ${alertClass}`}>
        <div className="alert-header">
          {isHighRisk ? (
            <AlertCircle size={26} color="var(--danger-red)" />
          ) : isMedRisk ? (
            <AlertTriangle size={26} color="var(--warn-amber)" />
          ) : (
            <CheckCircle2 size={26} color="var(--safe-green)" />
          )}
          <div>
            <div className="alert-title">
              {results.threat_level || (isHighRisk ? "CRITICAL THREAT" : isMedRisk ? "ELEVATED ADVISORY" : "SAFE / VERIFIED")}
            </div>
            <div style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', opacity: 0.85 }}>
              STATUS: {results.voice_risk_status} | INCIDENT ACTION CODE: {results.action_code || "EVALUATED"}
            </div>
          </div>
        </div>

        <div className="alert-text">
          {results.security_alert}
        </div>

        <div className="recommendation-box">
          <Lock size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong style={{ display: 'block', marginBottom: '2px', textTransform: 'uppercase', fontSize: '0.75rem', letterSpacing: '0.05em' }}>
              Recommended Security Action:
            </strong>
            {results.recommended_action}
          </div>
        </div>
      </div>

      {/* 2. Core Metrics Row: AI Prob, Human Prob, Voice Risk, Contextual Risk */}
      <div className="metrics-row">
        {/* Metric 1: AI-Generated Probability */}
        <div className="metric-card">
          <div className="metric-top">
            <span>AI-Generated Voice</span>
            <Cpu size={15} color="var(--cyan-primary)" />
          </div>
          <div className="metric-value-box">
            <span className="metric-number" style={{ color: isHighRisk ? 'var(--danger-red)' : (isMedRisk ? 'var(--warn-amber)' : 'var(--safe-green)') }}>
              {aiPct}
            </span>
            <span className="metric-unit">%</span>
          </div>
          <div className="risk-bar-container">
            <div 
              className="risk-bar-fill" 
              style={{ 
                width: `${aiPct}%`,
                background: isHighRisk ? 'var(--danger-red)' : (isMedRisk ? 'var(--warn-amber)' : 'var(--safe-green)')
              }}
            />
          </div>
          <div style={{ marginTop: '0.6rem', fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
            <span>Model Confidence</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}>
              {Math.round(results.confidence * 100)}%
            </span>
          </div>
        </div>

        {/* Metric 2: Human Probability */}
        <div className="metric-card">
          <div className="metric-top">
            <span>Genuine Human</span>
            <ShieldAlert size={15} color="var(--safe-green)" />
          </div>
          <div className="metric-value-box">
            <span className="metric-number" style={{ color: humanPct >= 60 ? 'var(--safe-green)' : 'var(--text-muted)' }}>
              {humanPct}
            </span>
            <span className="metric-unit">%</span>
          </div>
          <div className="risk-bar-container">
            <div 
              className="risk-bar-fill" 
              style={{ 
                width: `${humanPct}%`,
                background: humanPct >= 60 ? 'var(--safe-green)' : 'var(--cyan-primary)'
              }}
            />
          </div>
          <div style={{ marginTop: '0.6rem', fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
            <span>Classification</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-main)', fontSize: '0.7rem' }}>
              {results.classification}
            </span>
          </div>
        </div>

        {/* Metric 3: Pure ML Voice Risk Score */}
        <div className="metric-card">
          <div className="metric-top">
            <span>Voice Risk Score</span>
            <span className={`metric-badge ${isHighRisk ? 'demo-badge-high' : (isMedRisk ? 'demo-badge-low' : 'demo-badge-low')}`}>
              {results.voice_risk_status}
            </span>
          </div>
          <div className="metric-value-box">
            <span className="metric-number" style={{ color: isHighRisk ? 'var(--danger-red)' : (isMedRisk ? 'var(--warn-amber)' : 'var(--safe-green)') }}>
              {results.voice_risk_score}
            </span>
            <span className="metric-unit">/100</span>
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
            Thresholds: Low &lt;40 | Med 40–70 | High &gt;70
          </div>
          <div style={{ marginTop: '0.4rem', fontSize: '0.72rem', color: 'var(--text-dim)' }}>
            Pure acoustic ML probability score
          </div>
        </div>

        {/* Metric 4: Contextual Enterprise Threat Score */}
        <div className="metric-card" style={{ borderLeft: '3px solid var(--purple-accent)' }}>
          <div className="metric-top">
            <span>Contextual Threat</span>
            <Layers size={15} color="var(--purple-accent)" />
          </div>
          <div className="metric-value-box">
            <span className="metric-number" style={{ color: 'var(--purple-accent)' }}>
              {results.contextual_risk_score}
            </span>
            <span className="metric-unit">/100</span>
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
            Multi-factor enterprise transaction risk
          </div>
          <div style={{ marginTop: '0.4rem', fontSize: '0.72rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
            Includes Amount, Urgency & Caller Authority
          </div>
        </div>
      </div>
    </div>
  );
}
