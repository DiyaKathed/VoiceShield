import React, { useState } from 'react';
import { 
  ArrowLeft, RefreshCw, AlertCircle, AlertTriangle, CheckCircle2, 
  Lock, Shield, Volume2, Activity, Info, ChevronDown, ChevronUp,
  Cpu, FileText
} from 'lucide-react';

export default function AnalysisDashboardStage({
  results,
  currentAudioUrl,
  currentFilename,
  onBackToInput,
  onAnalyzeAnother,
  onOpenMetrics,
  onOpenResearch
}) {
  const [showDebug, setShowDebug] = useState(false);

  if (!results) {
    return (
      <div className="stage-analysis-loading">
        <RefreshCw size={36} className="spin" color="var(--accent-cyan)" />
        <p>Loading forensic security analysis...</p>
      </div>
    );
  }

  const aiProb = results.ai_generated_probability ?? (results.voice_analysis?.ai_probability ?? 0.5);
  const humanProb = results.human_probability ?? (results.voice_analysis?.human_probability ?? 0.5);
  const aiPct = Math.round(aiProb * 100);
  const humanPct = Math.round(humanProb * 100);

  const voiceRisk = results.voice_risk_score ?? (results.voice_analysis?.risk_score ?? Math.round(aiProb * 100));
  const isHighRisk = voiceRisk >= 70;
  const isMedRisk = voiceRisk >= 40 && voiceRisk < 70;
  const riskStatus = results.voice_risk_status || (results.voice_analysis?.risk_level || (isHighRisk ? "HIGH RISK" : isMedRisk ? "MEDIUM RISK" : "LOW RISK"));
  const riskClass = isHighRisk ? 'risk-high' : (isMedRisk ? 'risk-med' : 'risk-low');

  const classification = results.classification || (results.voice_analysis?.classification || (isHighRisk ? "SYNTHETIC / AI VOICE CLONE" : isMedRisk ? "SUSPICIOUS / ANOMALOUS SPEECH" : "GENUINE HUMAN SPEECH"));
  const confidencePct = Math.round((results.confidence ?? 0.5) * 100);

  // Segments timeline
  const segments = results.chunk_timeline || results.segments || [];

  // Transaction context
  const context = results.transaction_context || {};
  const contextRisk = results.contextual_risk_score || 0;
  const threatLevel = results.threat_level || (isHighRisk ? "CRITICAL THREAT" : isMedRisk ? "ELEVATED ADVISORY" : "VERIFIED SAFE");

  // Dynamic Security Alert & Action Recommendation based on real model output
  const securityAlert = isHighRisk 
    ? "Possible AI-generated or voice-cloned speech detected." 
    : isMedRisk 
      ? "Anomalous or degraded spectral signatures detected in voice recording." 
      : "No strong evidence of AI-generated speech was detected. Continue to follow normal verification procedures.";

  const recommendedAction = isHighRisk 
    ? "Verify the caller through an independent communication channel before authorizing sensitive actions." 
    : isMedRisk 
      ? "Request additional biometric verification or secondary supervisor approval before authorization." 
      : "Permit transaction processing under standard operational oversight protocols.";

  return (
    <div className="stage-analysis-container">
      {/* Top Navigation Bar */}
      <div className="analysis-top-nav">
        <button className="btn-nav-back" onClick={onBackToInput}>
          <ArrowLeft size={18} />
          <span>Back to Audio Input</span>
        </button>

        <div className="analysis-page-title">
          <Shield size={20} color="var(--accent-cyan)" />
          <span>VoiceShield Voice Security Analysis</span>
        </div>

        <button className="btn-nav-reset" onClick={onAnalyzeAnother}>
          <RefreshCw size={16} />
          <span>Analyze Another Voice</span>
        </button>
      </div>

      {/* Audio Playback Bar */}
      <div className="analysis-audio-bar">
        <div className="audio-meta-left">
          <Volume2 size={20} color="var(--accent-cyan)" />
          <span className="audio-filename">{currentFilename || "Analyzed Voice Stream"}</span>
          <span className="audio-duration-badge">{results.duration_sec ? `${results.duration_sec}s` : 'Standard Audio'}</span>
        </div>
        {currentAudioUrl && (
          <audio controls src={currentAudioUrl} className="analysis-audio-player" />
        )}
      </div>

      {/* Main Forensic Grid */}
      <div className="analysis-main-grid">
        {/* Left Column: Voice ML Analysis */}
        <div className="analysis-column-left">
          {/* Overall Voice Analysis Card */}
          <div className={`forensic-card ${riskClass}-card`}>
            <div className="card-section-label">VOICE ANALYSIS</div>

            <div className="classification-row">
              <div className="classification-title">{classification}</div>
              <div className={`status-pill ${riskClass}`}>{riskStatus}</div>
            </div>

            {/* Large Probability Gauges */}
            <div className="probability-display-grid">
              <div className="prob-box ai-box">
                <div className="prob-label">AI GENERATED</div>
                <div className="prob-value ai-value">{aiPct}%</div>
                <div className="prob-bar-track">
                  <div className="prob-bar-fill ai-fill" style={{ width: `${aiPct}%` }} />
                </div>
              </div>

              <div className="prob-box human-box">
                <div className="prob-label">HUMAN SPEECH</div>
                <div className="prob-value human-value">{humanPct}%</div>
                <div className="prob-bar-track">
                  <div className="prob-bar-fill human-fill" style={{ width: `${humanPct}%` }} />
                </div>
              </div>
            </div>

            {/* Risk & Confidence Badges */}
            <div className="metrics-summary-row">
              <div className="metric-pill">
                <span className="m-label">Voice Risk:</span>
                <span className="m-val" style={{ color: isHighRisk ? 'var(--danger-red)' : isMedRisk ? 'var(--warn-amber)' : 'var(--safe-green)' }}>
                  {voiceRisk}/100
                </span>
              </div>
              <div className="metric-pill">
                <span className="m-label">Model Confidence:</span>
                <span className="m-val">{confidencePct}%</span>
              </div>
              <div className="metric-pill">
                <span className="m-label">Model Engine:</span>
                <span className="m-val">VoiceShield IndicTTS</span>
              </div>
            </div>
          </div>

          {/* Security Alert & Recommendation Card */}
          <div className={`security-alert-box ${riskClass}`}>
            <div className="alert-header-row">
              {isHighRisk ? (
                <AlertCircle size={24} color="var(--danger-red)" />
              ) : isMedRisk ? (
                <AlertTriangle size={24} color="var(--warn-amber)" />
              ) : (
                <CheckCircle2 size={24} color="var(--safe-green)" />
              )}
              <div className="alert-title-group">
                <div className="alert-headline">{threatLevel}</div>
                <div className="alert-subhead">INCIDENT POLICY: {results.action_code || "FORENSIC_EVALUATED"}</div>
              </div>
            </div>

            <div className="alert-body-text">{securityAlert}</div>

            <div className="action-recommendation-box">
              <Lock size={18} color="var(--accent-cyan)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong>Recommended Action:</strong> {recommendedAction}
              </div>
            </div>
          </div>

          {/* Segment Analysis Timeline & Probability Graph */}
          <div className="timeline-card">
            <div className="card-section-label">
              <span>SEGMENT ANALYSIS & TEMPORAL TIMELINE</span>
              {results.partial_spoof_detected && (
                <span className="badge-spliced-alert">
                  <AlertTriangle size={14} /> SPLICED SPOOF DETECTED
                </span>
              )}
            </div>
            <p className="timeline-subtitle">
              Sliding-window forensic analysis across 2.0s segments. Identifies localized voice synthesis or spliced directives.
            </p>

            <div className="timeline-segment-list">
              {segments.map((seg, idx) => {
                const segAi = seg.ai_probability ?? 0;
                const segPct = Math.round(segAi * 100);
                const segRisk = segAi >= 0.70 ? 'HIGH' : segAi >= 0.40 ? 'MED' : 'LOW';
                const segColor = segAi >= 0.70 ? 'var(--danger-red)' : segAi >= 0.40 ? 'var(--warn-amber)' : 'var(--safe-green)';
                const startTime = seg.start_time !== undefined ? seg.start_time : seg.start;
                const endTime = seg.end_time !== undefined ? seg.end_time : seg.end;

                return (
                  <div key={idx} className="timeline-segment-row">
                    <div className="segment-timestamp">
                      {startTime.toFixed(1)}s ───── {endTime.toFixed(1)}s
                    </div>
                    <div className="segment-graph-track">
                      <div 
                        className="segment-graph-fill" 
                        style={{ width: `${segPct}%`, backgroundColor: segColor }} 
                      />
                    </div>
                    <div className="segment-prob" style={{ color: segColor }}>
                      AI: {segPct}% <span className="seg-risk-tag">[{segRisk}]</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Transaction Context Risk & Model Info */}
        <div className="analysis-column-right">
          {/* Transaction Context Card */}
          <div className="context-card">
            <div className="card-section-label">TRANSACTION CONTEXT RISK</div>
            <div className="context-notice">
              Evaluated separately from acoustic ML model to prevent operational score pollution.
            </div>

            <div className="context-details-list">
              <div className="context-detail-item">
                <span className="cd-label">Caller Identity</span>
                <span className="cd-val">{context.caller_name || "Unknown Caller"}</span>
              </div>
              <div className="context-detail-item">
                <span className="cd-label">Claimed Role</span>
                <span className="cd-val">{context.caller_role || "Standard Employee"}</span>
              </div>
              <div className="context-detail-item">
                <span className="cd-label">Transfer Amount</span>
                <span className="cd-val">
                  ${Number(context.amount || 0).toLocaleString()}
                </span>
              </div>
              <div className="context-detail-item">
                <span className="cd-label">Urgency Level</span>
                <span className="cd-val">{context.urgency || "Normal"}</span>
              </div>
              <div className="context-detail-item">
                <span className="cd-label">Speaker Verification</span>
                <span className="cd-val" style={{ color: context.speaker_verification?.includes('Failed') || context.speaker_verification?.includes('Mismatch') ? 'var(--danger-red)' : 'inherit' }}>
                  {context.speaker_verification || "Unregistered"}
                </span>
              </div>
            </div>

            <div className="context-score-footer">
              <div className="cs-label">Contextual Risk Score:</div>
              <div className="cs-val" style={{ color: contextRisk >= 70 ? 'var(--danger-red)' : contextRisk >= 40 ? 'var(--warn-amber)' : 'var(--safe-green)' }}>
                {contextRisk}/100
              </div>
            </div>
          </div>

          {/* Model Information & Research Badges */}
          <div className="model-info-card">
            <div className="card-section-label">MODEL SPECIFICATIONS</div>
            <div className="model-specs-list">
              <div className="spec-row">
                <span className="spec-k">Model Name:</span>
                <span className="spec-v">VoiceShield IndicTTS</span>
              </div>
              <div className="spec-row">
                <span className="spec-k">Dataset:</span>
                <span className="spec-v">IndicTTS Challenge (16 Indian Langs)</span>
              </div>
              <div className="spec-row">
                <span className="spec-k">Architecture:</span>
                <span className="spec-v">Spectro-Temporal Residual CNN</span>
              </div>
              <div className="spec-row">
                <span className="spec-k">Calibration:</span>
                <span className="spec-v">Temperature Scaling (Validation fit)</span>
              </div>
              <div className="spec-row">
                <span className="spec-k">Sampling Rate:</span>
                <span className="spec-v">16,000 Hz Mono</span>
              </div>
            </div>

            <div className="modal-triggers-row">
              <button className="btn-secondary-link" onClick={onOpenMetrics}>
                <Activity size={15} />
                <span>View Full Test Benchmarks</span>
              </button>
              <button className="btn-secondary-link" onClick={onOpenResearch}>
                <Info size={15} />
                <span>Research Considerations</span>
              </button>
            </div>
          </div>

          {/* Developer / Forensic Debug Card */}
          {results.debug && (
            <div className="debug-card">
              <button className="debug-toggle-btn" onClick={() => setShowDebug(!showDebug)}>
                <Cpu size={16} color="var(--accent-cyan)" />
                <span>Forensic Pipeline Debug Metrics</span>
                {showDebug ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>

              {showDebug && (
                <div className="debug-details">
                  <div className="debug-row">
                    <span>Duration:</span>
                    <span>{results.debug.audio_duration_sec}s ({results.debug.sample_rate} Hz)</span>
                  </div>
                  <div className="debug-row">
                    <span>Total Segments:</span>
                    <span>{results.debug.total_segments} chunks</span>
                  </div>
                  <div className="debug-row">
                    <span>Min AI Probability:</span>
                    <span>{(results.debug.min_ai_probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="debug-row">
                    <span>Max AI Probability:</span>
                    <span>{(results.debug.max_ai_probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="debug-row">
                    <span>Mean AI Probability:</span>
                    <span>{(results.debug.mean_ai_probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="debug-row">
                    <span>Median AI Probability:</span>
                    <span>{(results.debug.median_ai_probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="debug-row">
                    <span>Trimmed Mean AI Probability:</span>
                    <span>{(results.debug.trimmed_mean_ai_probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="debug-row">
                    <span>Aggregation Method:</span>
                    <span>Speech-weighted median / trimmed mean</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
