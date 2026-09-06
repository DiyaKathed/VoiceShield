import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import AudioInputStage from './components/AudioInputStage';
import AnalysisDashboardStage from './components/AnalysisDashboardStage';
import MetricsModal from './components/MetricsModal';
import ResearchModal from './components/ResearchModal';

export default function App() {
  // Stage management: 'input' (Stage 1) or 'analysis' (Stage 2)
  const [stage, setStage] = useState('input');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  
  // Audio state
  const [currentAudioUrl, setCurrentAudioUrl] = useState('');
  const [currentAudioBlob, setCurrentAudioBlob] = useState(null);
  const [currentFilename, setCurrentFilename] = useState('sample_fake_01.wav');

  // Modals
  const [isMetricsOpen, setIsMetricsOpen] = useState(false);
  const [isResearchOpen, setIsResearchOpen] = useState(false);

  // Transaction Context State
  const [transactionContext, setTransactionContext] = useState({
    caller_name: "Finance Director",
    caller_role: "Managing Director",
    amount: 2500000.0,
    urgency: "Immediate",
    speaker_verification: "Mismatch / Failed"
  });

  // Pre-load default demo audio so it is immediately playable on Stage 1
  useEffect(() => {
    const initDefaultSample = async () => {
      try {
        const audioUrl = '/api/sample-audio/sample_fake_01.wav';
        setCurrentAudioUrl(audioUrl);
        const res = await fetch(audioUrl);
        if (res.ok) {
          const blob = await res.blob();
          setCurrentAudioBlob(blob);
        }
      } catch (e) {
        console.log("Waiting for backend server initialization:", e);
      }
    };
    initDefaultSample();
  }, []);

  // Execute Analysis via FastAPI backend and transition to Stage 2
  const handleProceedToAnalysis = async () => {
    let blobToSend = currentAudioBlob;
    let filenameToSend = currentFilename || 'audio.wav';

    // If blob not in state yet, try fetching from currentAudioUrl
    if (!blobToSend && currentAudioUrl) {
      try {
        const res = await fetch(currentAudioUrl);
        blobToSend = await res.blob();
        setCurrentAudioBlob(blobToSend);
      } catch (e) {
        console.error("Failed to fetch audio blob:", e);
      }
    }

    if (!blobToSend) {
      alert("Please upload, record, or select an audio sample first.");
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', blobToSend, filenameToSend);
      formData.append('caller_name', transactionContext.caller_name || "Unknown");
      formData.append('caller_role', transactionContext.caller_role || "Employee");
      formData.append('amount', (transactionContext.amount || 0).toString());
      formData.append('urgency', transactionContext.urgency || "Normal");
      formData.append('speaker_verification', transactionContext.speaker_verification || "Unregistered");

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Inference failed");
      }

      const data = await response.json();
      setResults(data);
      // Navigate to Stage 2
      setStage('analysis');
    } catch (err) {
      console.error("Analysis Error:", err);
      alert("Inference Error: " + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleBackToInput = () => {
    setStage('input');
  };

  const handleAnalyzeAnother = () => {
    setResults(null);
    setCurrentAudioUrl('');
    setCurrentAudioBlob(null);
    setCurrentFilename('');
    setStage('input');
  };

  return (
    <div className="app-container">
      {/* Top Cybersecurity Navbar */}
      <Navbar 
        onOpenMetrics={() => setIsMetricsOpen(true)}
        onOpenResearch={() => setIsResearchOpen(true)}
      />

      {/* Main Content Area: Stage 1 vs Stage 2 */}
      <main className="main-content">
        {stage === 'input' ? (
          <AudioInputStage 
            onAnalyze={handleProceedToAnalysis}
            isLoading={isLoading}
            currentAudioUrl={currentAudioUrl}
            setCurrentAudioUrl={setCurrentAudioUrl}
            currentAudioBlob={currentAudioBlob}
            setCurrentAudioBlob={setCurrentAudioBlob}
            currentFilename={currentFilename}
            setCurrentFilename={setCurrentFilename}
            transactionContext={transactionContext}
            setTransactionContext={setTransactionContext}
            onProceedToAnalysis={handleProceedToAnalysis}
          />
        ) : (
          <AnalysisDashboardStage 
            results={results}
            currentAudioUrl={currentAudioUrl}
            currentFilename={currentFilename}
            onBackToInput={handleBackToInput}
            onAnalyzeAnother={handleAnalyzeAnother}
            onOpenMetrics={() => setIsMetricsOpen(true)}
            onOpenResearch={() => setIsResearchOpen(true)}
          />
        )}
      </main>

      {/* Evaluation Metrics Modal */}
      <MetricsModal 
        isOpen={isMetricsOpen} 
        onClose={() => setIsMetricsOpen(false)} 
      />

      {/* Research Architecture Modal */}
      <ResearchModal 
        isOpen={isResearchOpen} 
        onClose={() => setIsResearchOpen(false)} 
      />

      {/* Global Footer */}
      <footer className="footer">
        VoiceShield MVP &bull; Smart India Hackathon &bull; Local PyTorch Deepfake Forensic Engine &bull; Zero External API Dependency
      </footer>
    </div>
  );
}
