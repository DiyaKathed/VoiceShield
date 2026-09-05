import React from 'react';
import { X, Cpu, Shield, Layers, Radio, Volume2, MicOff, Network } from 'lucide-react';

export default function ResearchModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  const topics = [
    {
      icon: <Cpu size={20} color="var(--cyan-primary)" />,
      title: "1. Unseen Synthetic Generators & Vocoder Artifacts",
      desc: "Modern cloning models (ElevenLabs, Tortoise, VITS, HiFi-GAN) synthesize speech through neural vocoders that leave characteristic phase irregularities in upper frequency bands (>2.5kHz). VoiceShield extracts Log-Mel Spectrograms with 1st and 2nd order temporal deltas to detect transitional phase incoherence without needing prior exposure to specific vocoder architectures."
    },
    {
      icon: <Layers size={20} color="var(--warn-amber)" />,
      title: "2. Partial / Segment-Level Spoofing",
      desc: "Attackers commonly disguise impersonation attempts by having a human speak mundane phrases, splicing in short synthetic cloned directives (e.g. 'authorize wire transfer'). VoiceShield executes a 2.0s sliding window chunk analyzer that evaluates speech segments chronologically and alerts when anomalous spikes occur."
    },
    {
      icon: <Radio size={20} color="var(--blue-accent)" />,
      title: "3. Compression & Channel Distortion",
      desc: "Telephony channels (G.711, AMR-WB, Opus) introduce lossy frequency cutoffs (300Hz–3.4kHz). VoiceShield standardizes inputs to 16kHz mono and concentrates spectral modeling on mid-frequency formant trajectories where cloning artifacts persist despite lossy compression."
    },
    {
      icon: <Volume2 size={20} color="var(--purple-accent)" />,
      title: "4. Background Noise & SNR Robustness",
      desc: "Acoustic field recordings exhibit fluctuating signal-to-noise ratios. During training, SpecAugment frequency/time masking and additive Gaussian noise are applied to force the neural network to identify vocal tract biometric dynamics rather than ambient room acoustics."
    },
    {
      icon: <MicOff size={20} color="#f87171" />,
      title: "5. Silence & Non-Speech Artifact Gating",
      desc: "Silent pauses and non-speech breathing sounds can cause false positives if a model overfits to background silence. VoiceShield implements energy-based Voice Activity Detection (VAD) to trim or gate unvoiced periods before classification."
    },
    {
      icon: <Network size={20} color="var(--safe-green)" />,
      title: "6. Cross-Dataset Generalization & Disjoint Splitting",
      desc: "To prevent artificial inflated benchmarks caused by speaker leakage, the dataset pipeline enforces strict speaker-level disjoint splits across training, validation, and test sets. Acoustic feature representations remain invariant to individual speaker identities."
    }
  ];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" style={{ maxWidth: '750px' }} onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Cpu size={22} color="var(--purple-accent)" />
            <h3 style={{ fontFamily: 'var(--font-display)' }}>
              VoiceShield Research Architecture & Defense Design
            </h3>
          </div>
          <button className="close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <p style={{ fontSize: '0.84rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
          Design implementation addressing fundamental research challenges and limitations in real-world audio deepfake detection:
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {topics.map((t, idx) => (
            <div 
              key={idx}
              style={{
                background: 'rgba(10, 17, 30, 0.6)',
                border: '1px solid var(--border-dim)',
                borderRadius: 'var(--radius-md)',
                padding: '0.85rem 1rem'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.35rem' }}>
                {t.icon}
                <strong style={{ fontSize: '0.9rem', color: '#ffffff' }}>{t.title}</strong>
              </div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: '1.45' }}>
                {t.desc}
              </p>
            </div>
          ))}
        </div>

        <div style={{ marginTop: '1.5rem', textAlign: 'right' }}>
          <button 
            className="analyze-action-btn" 
            style={{ display: 'inline-flex', padding: '0.5rem 1.25rem', fontSize: '0.85rem' }}
            onClick={onClose}
          >
            Close Overview
          </button>
        </div>
      </div>
    </div>
  );
}
