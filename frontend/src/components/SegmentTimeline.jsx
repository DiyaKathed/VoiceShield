import React, { useState } from 'react';
import { Activity, AlertTriangle, Clock, Info } from 'lucide-react';

export default function SegmentTimeline({ timeline, partialSpoofDetected, durationSec }) {
  const [selectedChunk, setSelectedChunk] = useState(null);

  if (!timeline || timeline.length === 0) {
    return null;
  }

  return (
    <div className="timeline-card">
      <div className="card-header">
        <div>
          <div className="card-title">
            <Activity size={18} />
            <span>Real-Time Segment-Level Timeline</span>
          </div>
          <div className="timeline-header-info">
            Audio duration: {durationSec}s divided into {timeline.length} sliding-window chunks (2.0s segments)
          </div>
        </div>

        {partialSpoofDetected && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid rgba(239, 68, 68, 0.4)',
            color: '#f87171',
            padding: '0.25rem 0.65rem',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.74rem',
            fontFamily: 'var(--font-mono)',
            fontWeight: 600
          }}>
            <AlertTriangle size={14} />
            <span>PARTIAL / SPLICED SPOOF DETECTED</span>
          </div>
        )}
      </div>

      {/* Chunk timeline bars */}
      <div className="timeline-track">
        {timeline.map((chunk, idx) => {
          const aiPct = Math.round(chunk.ai_probability * 100);
          const isHigh = chunk.risk_level === 'HIGH';
          const isMed = chunk.risk_level === 'MEDIUM';
          const chunkClass = isHigh ? 'chunk-high' : (isMed ? 'chunk-med' : 'chunk-low');
          const barColor = isHigh ? 'var(--danger-red)' : (isMed ? 'var(--warn-amber)' : 'var(--safe-green)');

          return (
            <div 
              key={idx} 
              className={`timeline-chunk-item ${chunkClass}`}
              onClick={() => setSelectedChunk(chunk)}
              title={`Chunk #${chunk.chunk_id}: ${chunk.time_label} | AI Prob: ${aiPct}% | ${chunk.risk_level} RISK`}
            >
              <div className="chunk-pct" style={{ color: barColor }}>
                {aiPct}%
              </div>

              <div className="chunk-bar-visual">
                <div 
                  className="chunk-bar-inner" 
                  style={{ 
                    height: `${Math.max(10, aiPct)}%`,
                    background: barColor 
                  }}
                />
              </div>

              <div className="chunk-time-tag">
                {chunk.start_time}s
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Chunk Details / Inspector */}
      {selectedChunk && (
        <div style={{
          marginTop: '0.75rem',
          padding: '0.65rem 0.85rem',
          background: 'rgba(10, 17, 30, 0.7)',
          border: '1px solid var(--border-dim)',
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.78rem',
          fontFamily: 'var(--font-mono)'
        }}>
          <div>
            <span style={{ color: 'var(--cyan-primary)' }}>Segment #{selectedChunk.chunk_id}: </span>
            <span>{selectedChunk.time_label}</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>AI Cloned: </span>
            <strong>{Math.round(selectedChunk.ai_probability * 100)}%</strong>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Genuine: </span>
            <strong>{Math.round(selectedChunk.human_probability * 100)}%</strong>
          </div>
          <div>
            <span className={selectedChunk.risk_level === 'HIGH' ? "demo-badge-high" : "demo-badge-low"}>
              {selectedChunk.risk_level} RISK
            </span>
          </div>
        </div>
      )}

      {/* Architecture Disclaimer */}
      <div className="research-badge-banner">
        <Info size={16} style={{ flexShrink: 0 }} />
        <span>
          <strong>Real-Time Pipeline Note:</strong> Analyzes speech via sliding temporal windows (1.5s segment / 0.75s step) to identify spliced attacks where an adversary injects synthetic voice segments into real speech. Demonstrates near-real-time streaming analysis (not cellular-call interception).
        </span>
      </div>
    </div>
  );
}
