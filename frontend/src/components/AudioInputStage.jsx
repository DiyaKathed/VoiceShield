import React, { useState, useRef, useEffect } from 'react';
import {
  UploadCloud, Mic, Square, Play, Sparkles, DollarSign,
  UserCheck, AlertTriangle, ShieldCheck, FileAudio, Trash2,
  Shield, CheckCircle2, RefreshCw, Volume2, ArrowRight
} from 'lucide-react';

export default function AudioInputStage({
  onAnalyze,
  isLoading,
  currentAudioUrl,
  setCurrentAudioUrl,
  currentAudioBlob,
  setCurrentAudioBlob,
  currentFilename,
  setCurrentFilename,
  transactionContext,
  setTransactionContext,
  onProceedToAnalysis
}) {
  const [activeTab, setActiveTab] = useState('demo'); // 'upload', 'record', 'demo'
  const [samples, setSamples] = useState([]);
  const [selectedDemoId, setSelectedDemoId] = useState('sample_fake_01.wav');
  const [audioDuration, setAudioDuration] = useState(null);

  // Microphone recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioRef = useRef(null);

  // Load demo samples from backend
  useEffect(() => {
    fetch('/api/samples')
      .then(res => res.json())
      .then(data => {
        if (data.samples && data.samples.length > 0) {
          setSamples(data.samples);
          // Set initial demo sample
          handleSelectDemo(data.samples[0]);
        }
      })
      .catch(err => console.error("Error fetching samples:", err));
  }, []);

  const handleSelectDemo = async (sample) => {
    setSelectedDemoId(sample.id);
    setCurrentFilename(sample.id);

    const audioUrl = `/api/sample-audio/${sample.id}`;
    setCurrentAudioUrl(audioUrl);

    try {
      const response = await fetch(audioUrl);
      const blob = await response.blob();
      setCurrentAudioBlob(blob);
    } catch (e) {
      console.error("Failed to load sample blob:", e);
    }

    setTransactionContext({
      caller_name: sample.caller_name || "Vikram Singhania",
      caller_role: sample.caller_role || "Managing Director",
      amount: sample.amount || 5000000.0,
      urgency: sample.urgency || "Immediate",
      speaker_verification: sample.speaker_verification || "Mismatch / Failed"
    });
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setCurrentFilename(file.name);
    setCurrentAudioBlob(file);
    const url = URL.createObjectURL(file);
    setCurrentAudioUrl(url);
    setSelectedDemoId(null);
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
      view.setUint16(20, 1, true);
      view.setUint16(22, numChannels, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * blockAlign, true);
      view.setUint16(32, blockAlign, true);
      view.setUint16(34, bitDepth, true);
      writeString(view, 36, 'data');
      view.setUint32(40, dataLength, true);

      let offset = 44;
      for (let i = 0; i < channelData.length; i++) {
        let sample = Math.max(-1, Math.min(1, channelData[i]));
        sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(offset, sample, true);
        offset += 2;
      }

      return new Blob([view], { type: 'audio/wav' });
    } catch (err) {
      console.warn("PCM conversion fallback:", err);
      return blob;
    }
  };

  const startRecording = async () => {
    audioChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const rawBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const wavBlob = await convertBlobToWav(rawBlob);
        const url = URL.createObjectURL(wavBlob);
        setCurrentAudioBlob(wavBlob);
        setCurrentAudioUrl(url);
        const name = `live_recording_${Date.now()}.wav`;
        setCurrentFilename(name);
        setSelectedDemoId(null);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start(250);
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (err) {
      console.error("Microphone access error:", err);
      alert("Microphone permission denied or unsupported audio hardware.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
    }
  };

  const handleClearAudio = () => {
    setCurrentAudioUrl('');
    setCurrentAudioBlob(null);
    setCurrentFilename('');
    setSelectedDemoId(null);
    setAudioDuration(null);
  };

  const handleAudioLoadedMetadata = (e) => {
    if (e.target.duration && !isNaN(e.target.duration)) {
      setAudioDuration(e.target.duration);
    }
  };

  const handleTriggerAnalysis = () => {
    if (!currentAudioBlob && !currentAudioUrl) {
      alert("Please upload, record, or select an audio sample first.");
      return;
    }
    onProceedToAnalysis();
  };

  return (
    <div className="stage-input-container">
      {/* Product Hero Header */}
      <div className="stage-hero">
        <div className="hero-shield-badge">
          <Shield size={20} color="var(--accent-cyan)" />
          <span>CYBER FORENSIC SUITE</span>
        </div>
        <h1 className="hero-title">VoiceShield</h1>
        <p className="hero-subtitle">
          AI-Powered Voice Deepfake Detection & Impersonation Attack Prevention
        </p>
        <p className="hero-description">
          Upload or record suspected voice audio for forensic neural analysis across 16 Indian languages.
          Detects vocoder anomalies, neural cloning artifacts, and spliced transaction directives.
        </p>
      </div>

      {/* Input Selection Card */}
      <div className="input-card">
        {/* Tab Navigation */}
        <div className="tab-nav">
          <button
            className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <UploadCloud size={18} />
            <span>Upload Audio</span>
          </button>

          <button
            className={`tab-btn ${activeTab === 'record' ? 'active' : ''}`}
            onClick={() => setActiveTab('record')}
          >
            <Mic size={18} />
            <span>Record Voice Live</span>
          </button>

          <button
            className={`tab-btn ${activeTab === 'demo' ? 'active' : ''}`}
            onClick={() => setActiveTab('demo')}
          >
            <Sparkles size={18} />
            <span>Preset Indian Samples</span>
          </button>
        </div>

        {/* Tab 1: File Upload */}
        {activeTab === 'upload' && (
          <div className="tab-content upload-tab">
            <label className="upload-dropzone">
              <input
                type="file"
                accept=".mp3,.wav,audio/mp3,audio/wav,audio/mpeg,audio/*"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
              />
              <div className="dropzone-icon">
                <UploadCloud size={48} color="var(--accent-cyan)" />
              </div>
              <div className="dropzone-text">
                <strong>Click to Browse</strong> or drag and drop audio file
              </div>
              <div className="dropzone-hint">
                Supported formats: MP3, WAV
              </div>
            </label>
          </div>
        )}

        {/* Tab 2: Live Recording */}
        {activeTab === 'record' && (
          <div className="tab-content record-tab">
            <div className="record-box">
              <div className={`record-ring ${isRecording ? 'pulse' : ''}`}>
                <Mic size={36} color={isRecording ? 'var(--danger-red)' : 'var(--accent-cyan)'} />
              </div>

              <div className="record-timer">
                {isRecording ? (
                  <span style={{ color: 'var(--danger-red)' }}>
                    RECORDING: {Math.floor(recordingTime / 60)}:{(recordingTime % 60).toString().padStart(2, '0')}
                  </span>
                ) : (
                  <span>Ready to capture microphone stream</span>
                )}
              </div>

              <div className="record-actions">
                {!isRecording ? (
                  <button className="btn-record-start" onClick={startRecording}>
                    <Mic size={18} /> Start Recording
                  </button>
                ) : (
                  <button className="btn-record-stop" onClick={stopRecording}>
                    <Square size={18} /> Stop & Preview
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Preset Indian Samples */}
        {activeTab === 'demo' && (
          <div className="tab-content demo-tab">
            <p className="demo-hint">
              Select an authentic voice sample from the deepfake benchmark test split:
            </p>
            <div className="demo-grid">
              {samples.map(sample => {
                const isSelected = selectedDemoId === sample.id;
                const isAI = sample.type.includes('AI') || sample.type.includes('SPLICED');
                return (
                  <button
                    key={sample.id}
                    className={`demo-card ${isSelected ? 'selected' : ''}`}
                    onClick={() => handleSelectDemo(sample)}
                  >
                    <div className="demo-card-header">
                      <span className={`pill-badge ${isAI ? 'pill-danger' : 'pill-safe'}`}>
                        {sample.type}
                      </span>
                      <span className="demo-lang">{sample.language}</span>
                    </div>
                    <div className="demo-card-title">{sample.title}</div>
                    <div className="demo-card-desc">{sample.description}</div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Active Audio Preview & Metadata Card */}
        {currentAudioUrl && (
          <div className="selected-audio-card">
            <div className="audio-header">
              <div className="audio-info">
                <FileAudio size={24} color="var(--accent-cyan)" />
                <div>
                  <div className="audio-filename">{currentFilename || "Selected Audio"}</div>
                  <div className="audio-meta">
                    <span style={{
                      display: 'inline-block',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      background: 'rgba(6, 182, 212, 0.15)',
                      color: 'var(--accent-cyan)',
                      fontWeight: 700,
                      marginRight: '8px',
                      fontSize: '0.75rem',
                      letterSpacing: '0.5px'
                    }}>
                      {currentFilename ? (currentFilename.toLowerCase().endsWith('.mp3') ? 'FORMAT: MP3' : (currentFilename.toLowerCase().endsWith('.wav') ? 'FORMAT: WAV' : 'AUDIO')) : 'MP3 / WAV'}
                    </span>
                    {audioDuration ? `${audioDuration.toFixed(1)}s Duration` : 'Standard 16kHz'} • Forensic Buffer Ready
                  </div>
                </div>
              </div>

              <button className="btn-clear-audio" onClick={handleClearAudio} title="Remove audio">
                <Trash2 size={16} />
                <span>Replace</span>
              </button>
            </div>

            <audio
              ref={audioRef}
              controls
              src={currentAudioUrl}
              className="native-audio-player"
              onLoadedMetadata={handleAudioLoadedMetadata}
            />
          </div>
        )}

        {/* Transaction Context Settings */}
        <div className="context-settings-card">
          <div className="context-title">
            <DollarSign size={18} color="var(--accent-cyan)" />
            <span>High-Value Transaction Context (Enterprise Risk Engine)</span>
          </div>
          <div className="context-inputs-grid">
            <div className="form-group">
              <label>Caller Name</label>
              <input
                type="text"
                value={transactionContext.caller_name}
                onChange={(e) => setTransactionContext({ ...transactionContext, caller_name: e.target.value })}
                placeholder="e.g. Vikram Singhania"
              />
            </div>

            <div className="form-group">
              <label>Caller Role</label>
              <input
                type="text"
                value={transactionContext.caller_role}
                onChange={(e) => setTransactionContext({ ...transactionContext, caller_role: e.target.value })}
                placeholder="e.g. Managing Director"
              />
            </div>

            <div className="form-group">
              <label>Amount (USD / INR)</label>
              <input
                type="number"
                value={transactionContext.amount}
                onChange={(e) => setTransactionContext({ ...transactionContext, amount: parseFloat(e.target.value) || 0 })}
                placeholder="5000000"
              />
            </div>

            <div className="form-group">
              <label>Biometric Verification</label>
              <select
                value={transactionContext.speaker_verification}
                onChange={(e) => setTransactionContext({ ...transactionContext, speaker_verification: e.target.value })}
              >
                <option value="Mismatch / Failed">Mismatch / Failed</option>
                <option value="Verified Enrolled">Verified Enrolled</option>
                <option value="Unregistered / Unknown">Unregistered / Unknown</option>
              </select>
            </div>
          </div>
        </div>

        {/* Primary Action Button: Analyze the Voice */}
        <div className="analyze-action-container">
          <button
            className="btn-primary-analyze"
            onClick={handleTriggerAnalysis}
            disabled={isLoading || (!currentAudioBlob && !currentAudioUrl)}
          >
            {isLoading ? (
              <>
                <RefreshCw size={22} className="spin" />
                <span>Running Neural Forensic Inference...</span>
              </>
            ) : (
              <>
                <ShieldCheck size={22} />
                <span>ANALYZE THE VOICE</span>
                <ArrowRight size={20} />
              </>
            )}
          </button>
          <div className="analyze-hint">
            Direct local PyTorch inference via VoiceShieldNet model. Zero external APIs.
          </div>
        </div>
      </div>
    </div>
  );
}
