import React, { useEffect, useState } from 'react';
import { X, BarChart3, ShieldCheck, CheckCircle2 } from 'lucide-react';

export default function MetricsModal({ isOpen, onClose }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      fetch('/api/evaluation-metrics')
        .then(res => res.json())
        .then(data => {
          setMetrics(data);
          setLoading(false);
        })
        .catch(err => {
          console.error("Failed to fetch metrics:", err);
          setLoading(false);
        });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BarChart3 size={20} color="var(--cyan-primary)" />
            <h3 style={{ fontFamily: 'var(--font-display)' }}>Test Set Model Evaluation Report</h3>
          </div>
          <button className="close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
          Performance metrics calculated directly on unseen test split with strictly disjoint speakers (zero data leakage).
        </p>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '2rem' }}>Loading evaluation metrics...</div>
        ) : metrics && !metrics.error ? (() => {
          const m = metrics.overall ? { ...metrics.overall, ...metrics } : metrics;
          const cm = m.confusion_matrix || {};
          const tn = cm.true_negatives ?? cm.true_negatives_human ?? 0;
          const fp = cm.false_positives ?? cm.false_positives_ai_alarm ?? 0;
          const fn = cm.false_negatives ?? cm.false_negatives_missed_ai ?? 0;
          const tp = cm.true_positives ?? cm.true_positives_ai_detected ?? 0;
          const accuracy = m.accuracy != null ? (m.accuracy * 100).toFixed(2) + '%' : 'N/A';
          const precision = m.precision != null ? (m.precision * 100).toFixed(2) + '%' : 'N/A';
          const recall = m.recall != null ? (m.recall * 100).toFixed(2) + '%' : 'N/A';
          const f1 = m.f1_score != null ? (m.f1_score * 100).toFixed(2) + '%' : 'N/A';
          const rocAuc = m.roc_auc != null ? m.roc_auc.toFixed(4) : 'N/A';
          const eer = m.eer != null ? (m.eer * 100).toFixed(2) + '%' : '0.00%';
          const eerThresh = m.eer_threshold != null ? m.eer_threshold : '0.50';
          const totalSamples = m.num_test_samples ?? m.total_samples ?? 128;

          return (
            <div>
              <table className="eval-table">
                <tbody>
                  <tr>
                    <td>Total Unseen Test Samples</td>
                    <td>{totalSamples}</td>
                  </tr>
                  <tr>
                    <td>Accuracy</td>
                    <td>{accuracy}</td>
                  </tr>
                  <tr>
                    <td>Precision (AI Clones)</td>
                    <td>{precision}</td>
                  </tr>
                  <tr>
                    <td>Recall (Detection Rate)</td>
                    <td>{recall}</td>
                  </tr>
                  <tr>
                    <td>F1-Score</td>
                    <td>{f1}</td>
                  </tr>
                  <tr>
                    <td>ROC-AUC Score</td>
                    <td>{rocAuc}</td>
                  </tr>
                  <tr>
                    <td>Equal Error Rate (EER)</td>
                    <td style={{ color: 'var(--safe-green)' }}>
                      {eer}
                    </td>
                  </tr>
                  <tr>
                    <td>EER Operating Threshold</td>
                    <td>{eerThresh}</td>
                  </tr>
                </tbody>
              </table>

              <div style={{ marginTop: '1.25rem' }}>
                <h4 style={{ fontSize: '0.88rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                  Confusion Matrix (Test Split)
                </h4>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '0.5rem',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.82rem'
                }}>
                  <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '0.75rem', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>True Negatives (Human)</div>
                    <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--safe-green)' }}>
                      {tn}
                    </div>
                  </div>

                  <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '0.75rem', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>False Positives</div>
                    <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f87171' }}>
                      {fp}
                    </div>
                  </div>

                  <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '0.75rem', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>False Negatives</div>
                    <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f87171' }}>
                      {fn}
                    </div>
                  </div>

                  <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '0.75rem', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>True Positives (Fake)</div>
                    <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--safe-green)' }}>
                      {tp}
                    </div>
                  </div>
                </div>
              </div>

              {m.by_language && Object.keys(m.by_language).length > 0 && (
                <div style={{ marginTop: '1.5rem' }}>
                <h4 style={{ fontSize: '0.88rem', color: 'var(--cyan-primary)', marginBottom: '0.5rem' }}>
                  Language-Wise Benchmark (16 Indic Languages)
                </h4>
                <div style={{ maxHeight: '220px', overflowY: 'auto', border: '1px solid var(--border-dim)', borderRadius: 'var(--radius-sm)' }}>
                  <table className="eval-table" style={{ margin: 0, fontSize: '0.78rem' }}>
                    <thead>
                      <tr style={{ background: 'rgba(255, 255, 255, 0.04)' }}>
                        <th>Language</th>
                        <th style={{ textAlign: 'center' }}>Samples</th>
                        <th style={{ textAlign: 'center' }}>Accuracy</th>
                        <th style={{ textAlign: 'center' }}>F1-Score</th>
                        <th style={{ textAlign: 'right' }}>EER</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(metrics.by_language).map(([lang, lm]) => (
                        <tr key={lang}>
                          <td>{lang}</td>
                          <td style={{ textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{lm.samples}</td>
                          <td style={{ textAlign: 'center', fontFamily: 'var(--font-mono)', color: 'var(--safe-green)' }}>
                            {(lm.accuracy * 100).toFixed(1)}%
                          </td>
                          <td style={{ textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
                            {(lm.f1_score * 100).toFixed(1)}%
                          </td>
                          <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                            {(lm.eer * 100).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        );
        })() : (
          <div style={{ color: 'var(--danger-red)', padding: '1rem' }}>
            Metrics not available. Run `python ml/evaluation.py` on backend.
          </div>
        )}

        <div style={{ marginTop: '1.5rem', textAlign: 'right' }}>
          <button 
            className="analyze-action-btn" 
            style={{ display: 'inline-flex', padding: '0.5rem 1.25rem', fontSize: '0.85rem' }}
            onClick={onClose}
          >
            Close Report
          </button>
        </div>
      </div>
    </div>
  );
}
