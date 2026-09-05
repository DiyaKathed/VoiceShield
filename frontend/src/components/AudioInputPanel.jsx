import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, Mic, Square, Play, Sparkles, DollarSign, UserCheck, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function AudioInputPanel({
  onAnalyze,
  isLoading,
  currentAudioUrl,
  setCurrentAudioUrl,
  currentAudioBlob,
  setCurrentAudioBlob,
  currentFilename,
  setCurrentFilename,
  transactionContext,
  setTransactionContext
}) {
  const [activeTab, setActiveTab] = useState('demo'); // 'demo', 'upload', 'record'
  const [samples, setSamples] = useState([]);
  const [selectedDemoId, setSelectedDemoId] = useState('sample_ai_cloned_fraud.wav');
  
  // Microphone recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  // Load demo samples from backend
  useEffect(() => {
    fetch('/api/samples')
      .then(res => res.json())
      .then(data => {
        if (data.samples) {
          setSamples(data.samples);
          // Set initial demo sample
          handleSelectDemo(data.samples[1]); // Default to AI Clone sample
        }
      })
      .catch(err => console.error("Error fetching samples:", err));
  }, []);

  const handleSelectDemo = async (sample) => {
    setSelectedDemoId(sample.id);
    setCurrentFilename(sample.id);

    // Fetch sample audio as blob
    const audioUrl = `/api/sample-audio/${sample.id}`;
    setCurrentAudioUrl(audioUrl);

    try {
      const response = await fetch(audioUrl);
      const blob = await response.blob();
      setCurrentAudioBlob(blob);
    } catch (e) {
      console.error("Failed to load sample blob:", e);
    }

    // Auto-fill transaction context from sample
    setTransactionContext({
      caller_name: sample.caller_name || "David Sterling",
      caller_role: sample.caller_role || "Chief Executive Officer",
      amount: sample.amount || 250000.0,
      urgency: sample.urgency || "Immediate",
      speaker_verification: sample.speaker_verification || "Mismatch / Failed"
    });
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setCurrentFilename(file.name);
    setCurrentAudioBlob(file);
    setCurrentAudioUrl(URL.createObjectURL(file));
  };

  // Helper to convert recorded audio blob to true 16-bit PCM WAV
  const convertBlobToWav = async (blob) => {
    try {
      const arrayBuffer = await blob.arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      
      const numChannels = 1;
      const sampleRate = audioBuffer.sampleRate;
      const bitDepth = 16;
      const bytesPerSample = bitDepth / 8;
      const blockAlign = numChannels * bytesPerSample;
      const channelData = audioBuffer.getChannelData(0);
      const dataLength = channelData.length * bytesPerSample;
      const bufferLength = 44 + dataLength;
      
      const buffer = new ArrayBuffer(bufferLength);
      const view = new DataView(buffer);
      
      const writeString = (v, offset, str) => {
        for (let i = 0; i < str.length; i++) {
          v.setUint8(offset + i, str.charCodeAt(i));
        }
      };
      
      writeString(view, 0, 'RIFF');
      view.setUint32(4, 36 + dataLength, true);
      writeString(view, 8, 'WAVE');
      writeString(view, 12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true); // PCM
      view.setUint16(22, numChannels, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * blockAlign, true);
      view.setUint16(32, blockAlign, true);
      view.setUint16(34, bitDepth, true);
      writeString(view, 36, 'data');
      view.setUint32(40, dataLength, true);
      
      let offset = 44;
      for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
      }
      
      audioCtx.close();
      return new Blob([view], { type: 'audio/wav' });
    } catch (err) {
      console.warn("WAV conversion fallback:", err);
      return blob;
    }
  };

  // Microphone recording functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const rawBlob = new Blob(audioChunksRef.current, { 
          type: mediaRecorderRef.current.mimeType || 'audio/webm' 
        });
        const wavBlob = await convertBlobToWav(rawBlob);
        setCurrentAudioBlob(wavBlob);
        setCurrentAudioUrl(URL.createObjectURL(wavBlob));
        setCurrentFilename("microphone_capture.wav");
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (err) {
      alert("Microphone access denied or unavailable: " + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!currentAudioBlob) {
      alert("Please select or record an audio file first!");
      return;
    }
    onAnalyze(currentAudioBlob, currentFilename);
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <Sparkles size={18} />
          <span>Speech Input & Target Audio</span>
        </div>
      </div>

      {/* Input Mode Selector Tabs */}
      <div className="tabs-header">
        <button 
          type="button" 
          className={`tab-btn ${activeTab === 'demo' ? 'active' : ''}`}
          onClick={() => setActiveTab('demo')}
        >
          <Sparkles size={14} />
          <span>1-Click Demos</span>
        </button>
        <button 
          type="button" 
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          <UploadCloud size={14} />
          <span>Upload File</span>
        </button>
        <button 
          type="button" 
          className={`tab-btn ${activeTab === 'record' ? 'active' : ''}`}
          onClick={() => setActiveTab('record')}
        >
          <Mic size={14} />
          <span>Record Mic</span>
        </button>
      </div>

      {/* TAB 1: 1-Click Interactive Demos */}
      {activeTab === 'demo' && (
        <div className="demo-selector">
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Select a preloaded speech sample for instant testing:
          </div>
          <div className="demo-pills">
            {samples.map((sample) => {
              const isSelected = selectedDemoId === sample.id;
              const isHigh = sample.expected_risk === 'HIGH';
              return (
                <button
                  key={sample.id}
                  type="button"
                  className={`demo-btn ${isSelected ? 'active' : ''}`}
                  onClick={() => handleSelectDemo(sample)}
                >
                  <div style={{ flex: 1 }}>
                    <div className="demo-btn-title">
                      <span>{sample.title}</span>
                      <span className={isHigh ? "demo-badge-high" : "demo-badge-low"}>
                        {sample.type}
                      </span>
                    </div>
                    <div className="demo-desc">{sample.description}</div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* TAB 2: File Upload */}
      {activeTab === 'upload' && (
        <div 
          className="dropzone"
          onClick={() => document.getElementById('file-upload-input').click()}
        >
          <input 
            type="file" 
            id="file-upload-input" 
            accept="audio/wav, audio/mp3, audio/m4a, audio/ogg" 
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />
          <div className="dropzone-icon">
            <UploadCloud size={24} />
          </div>
          <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>
            Click or drag & drop audio here
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
            Supports WAV, MP3, M4A (16kHz auto-resampled)
          </div>
        </div>
      )}

      {/* TAB 3: Mic Recording */}
      {activeTab === 'record' && (
        <div className="record-box">
          {!isRecording ? (
            <div>
              <button 
                type="button" 
                className="record-btn" 
                onClick={startRecording}
                title="Start Recording"
              >
                <Mic size={28} />
              </button>
              <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>Click to Record Voice</div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Record speech via microphone for live analysis
              </div>
            </div>
          ) : (
            <div>
              <button 
                type="button" 
                className="record-btn recording" 
                onClick={stopRecording}
                title="Stop Recording"
              >
                <Square size={24} />
              </button>
              <div style={{ color: 'var(--danger-red)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                RECORDING: {Math.floor(recordingTime / 60)}:{(recordingTime % 60).toString().padStart(2, '0')}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                Speak clearly into your microphone... Click square to finish.
              </div>
            </div>
          )}
        </div>
      )}

      {/* Audio Playback Preview */}
      {currentAudioUrl && (
        <div className="audio-player-card">
          <div className="audio-file-info">
            <span>FILE: {currentFilename || "Target Audio"}</span>
            <span>PCM / 16kHz</span>
          </div>
          <audio controls src={currentAudioUrl} />
        </div>
      )}

      {/* Simulated Transaction Context Form */}
      <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-dim)' }}>
        <div style={{ 
          fontSize: '0.82rem', 
          fontWeight: 600, 
          color: 'var(--cyan-primary)', 
          marginBottom: '0.75rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem'
        }}>
          <DollarSign size={15} />
          <span>Simulated High-Risk Transaction Context</span>
        </div>

        <form onSubmit={handleSubmit} className="context-form">
          <div className="form-group">
            <label className="form-label">Claimed Caller Name</label>
            <input 
              type="text" 
              className="form-input" 
              value={transactionContext.caller_name}
              onChange={e => setTransactionContext({ ...transactionContext, caller_name: e.target.value })}
              required
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div className="form-group">
              <label className="form-label">Caller Role / Authority</label>
              <select 
                className="form-select"
                value={transactionContext.caller_role}
                onChange={e => setTransactionContext({ ...transactionContext, caller_role: e.target.value })}
              >
                <option value="Chief Executive Officer">CEO / Executive</option>
                <option value="Chief Financial Officer">CFO / Treasury</option>
                <option value="Senior Accountant">Senior Accountant</option>
                <option value="Vendor / Partner">External Vendor</option>
                <option value="Unknown Caller">Unknown Caller</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Transfer Amount ($)</label>
              <input 
                type="number" 
                className="form-input" 
                value={transactionContext.amount}
                onChange={e => setTransactionContext({ ...transactionContext, amount: parseFloat(e.target.value) || 0 })}
                required
              />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div className="form-group">
              <label className="form-label">Urgency Pressure</label>
              <select 
                className="form-select"
                value={transactionContext.urgency}
                onChange={e => setTransactionContext({ ...transactionContext, urgency: e.target.value })}
              >
                <option value="Normal">Normal / Scheduled</option>
                <option value="Urgent">Urgent / Same-Day</option>
                <option value="Immediate">Immediate Transfer</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Speaker Biometric Match</label>
              <select 
                className="form-select"
                value={transactionContext.speaker_verification}
                onChange={e => setTransactionContext({ ...transactionContext, speaker_verification: e.target.value })}
              >
                <option value="Verified Enrolled">Verified Enrolled Voice</option>
                <option value="Unregistered / Unknown">Unregistered / Guest</option>
                <option value="Mismatch / Failed">Mismatch / Failed Biometric</option>
              </select>
            </div>
          </div>

          <button 
            type="submit" 
            className="analyze-action-btn"
            disabled={isLoading || !currentAudioBlob}
          >
            {isLoading ? (
              <span>Running Local PyTorch Analysis...</span>
            ) : (
              <>
                <ShieldCheck size={18} />
                <span>Execute VoiceShield ML Analysis</span>
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
