import React, { useEffect, useState, useRef } from "react";
import "./App.css";
import ReactMarkdown from "react-markdown";

const API_URL = "http://localhost:8000";

// Utility for copying text to clipboard
const copyToClipboard = (text: string) => {
  navigator.clipboard.writeText(text);
};

function App() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [leadingModel, setLeadingModel] = useState<string>("");
  const [prompt, setPrompt] = useState("");
  const [think, setThink] = useState(true);
  const [web_search, setWebSearch] = useState(true);
  const [loading, setLoading] = useState(false);
  const [modelResults, setModelResults] = useState<Record<string, string>>({});
  const [summary, setSummary] = useState<string>("");
  const [conclusion, setConclusion] = useState<string>("");
  const [pendingModels, setPendingModels] = useState<string[]>([]);
  const [error, setError] = useState<string>("");
  const resultsRef = useRef<HTMLDivElement>(null);
  const [stage, setStage] = useState<'idle' | 'models' | 'summary'>('idle');
  const summaryReceivedRef = useRef(false);
  const modelResultsCountRef = useRef(0);

  useEffect(() => {
    fetch(`${API_URL}/models`)
      .then((res) => res.json())
      .then(setModels)
      .catch(() => setError("Failed to load models"));
  }, []);

  useEffect(() => {
    if (resultsRef.current) {
      resultsRef.current.scrollTop = resultsRef.current.scrollHeight;
    }
  }, [modelResults, summary, conclusion]);

  const handleModelChange = (model: string) => {
    setSelectedModels((prev) =>
      prev.includes(model)
        ? prev.filter((m) => m !== model)
        : [...prev, model]
    );
    if (leadingModel === model && selectedModels.includes(model)) {
      setLeadingModel("");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setModelResults({});
    setSummary("");
    setConclusion("");
    setPendingModels([...selectedModels]);
    setStage('models');
    summaryReceivedRef.current = false;
    modelResultsCountRef.current = 0;
    if (!prompt.trim() || selectedModels.length === 0 || !leadingModel) {
      setError("Prompt, at least one model, and a leading model are required.");
      return;
    }
    setLoading(true);
    try {
      // Build query params for SSE
      const params = new URLSearchParams({
        prompt,
        leading_model: leadingModel,
        think: String(think),
        web_search: String(web_search),
      });
      selectedModels.forEach((m) => params.append("models", m));
      const eventSource = new EventSource(`${API_URL}/prompt/stream?${params.toString()}`);
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.model) {
            setModelResults((prev) => {
              const updated = { ...prev, [data.model]: data.response };
              const doneCount = Object.keys(updated).length;
              if (doneCount === selectedModels.length) {
                setStage('summary');
              }
              return updated;
            });
            setPendingModels((pending) => pending.filter((m) => m !== data.model));
          } else if (data.summary || data.conclusion) {
            setSummary(data.summary || "");
            setConclusion(data.conclusion || "");
            setStage('idle');
            summaryReceivedRef.current = true;
          }
        } catch (err) {
          // ignore parse errors
        }
      };
      eventSource.onerror = () => {
        // Only show error if summary/conclusion have not been received
        if (!summaryReceivedRef.current) {
          setError("Error contacting backend or models.");
        }
        setLoading(false);
        setStage('idle');
        eventSource.close();
      };
      // End loading when summary/conclusion arrives
      const cleanup = () => {
        setLoading(false);
        setPendingModels([]);
        setStage('idle');
        eventSource.close();
      };
      const summaryWatcher = setInterval(() => {
        if (summary || conclusion) {
          cleanup();
          clearInterval(summaryWatcher);
        }
      }, 500);
    } catch (err) {
      setError("Error contacting backend or models.");
      setLoading(false);
      setPendingModels([]);
      setStage('idle');
    }
  };

  return (
    <div className="app-container large">
      <h1>Deep Research Agent</h1>
      <form className="prompt-form" onSubmit={handleSubmit}>
        <div style={{ position: 'relative', width: '100%' }}>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type your research prompt..."
            rows={3}
            required
            style={{ width: '100%', paddingRight: '2.5rem', boxSizing: 'border-box', minWidth: 0, maxWidth: '100%' }}
          />
          {prompt && (
            <button
              type="button"
              className="copy-btn prompt-copy-btn"
              onClick={() => copyToClipboard(prompt)}
              title="Copy initial prompt to clipboard"
              style={{
                position: 'absolute',
                top: 8,
                right: 8,
                background: 'rgba(255,255,255,0.85)', // subtle background for contrast
                border: '1px solid #ccc',
                borderRadius: '6px',
                fontSize: '1.2rem',
                cursor: 'pointer',
                padding: '2px 6px',
                zIndex: 2
              }}
            >
              📋
            </button>
          )}
        </div>
        <div className="model-select">
          <label>Models:</label>
          <div className="model-list">
            {models.map((model) => (
              <label key={model}>
                <input
                  type="checkbox"
                  checked={selectedModels.includes(model)}
                  onChange={() => handleModelChange(model)}
                />
                {model}
              </label>
            ))}
          </div>
        </div>
        <div className="leading-model-select">
          <label>Leading Model:</label>
          <select
            value={leadingModel}
            onChange={(e) => setLeadingModel(e.target.value)}
            required
          >
            <option value="">Select leading model</option>
            {selectedModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>
        <div className="options">
          <label>
            <input
              type="checkbox"
              checked={think}
              onChange={() => setThink((v) => !v)}
            />
            Think
          </label>
          <label>
            <input
              type="checkbox"
              checked={web_search}
              onChange={() => setWebSearch((v) => !v)}
            />
            Web Search
          </label>
        </div>
        <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
          <button type="submit" disabled={loading}>
            {loading ? "Researching..." : "Send"}
          </button>
          <button
            type="button"
            onClick={() => {
              setPrompt("");
              setModelResults({});
              setSummary("");
              setConclusion("");
              setPendingModels([]);
              setError("");
              setSelectedModels([]);
              setLeadingModel("");
            }}
            disabled={loading}
            style={{ background: '#f3f4f6', color: '#222', border: '1px solid #ccc' }}
          >
            New Research
          </button>
        </div>
      </form>
      {error && <div className="error">{error}</div>}
      <div className="results-multi">
        {selectedModels.map((model) => (
          <div key={model} className="model-window">
            <div className="model-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span>{model}</span>
              {modelResults[model] && (
                <button
                  className="copy-btn"
                  onClick={() => copyToClipboard(modelResults[model])}
                  title={`Copy ${model} result to clipboard`}
                  style={{ marginLeft: 8 }}
                >
                  📋
                </button>
              )}
            </div>
            <div className="model-stream">
              {modelResults[model] ? (
                <ReactMarkdown>{modelResults[model]}</ReactMarkdown>
              ) : pendingModels.includes(model) || loading ? (
                <span className="spinner">Waiting for {model}...</span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
      {/* Animated spinner for overall progress */}
      {loading && (
        <div className="animated-spinner-block">
          <div className="animated-spinner"></div>
          <div className="animated-spinner-label">
            {stage === 'models' && 'Researching in models...'}
            {stage === 'summary' && 'Summarizing results...'}
          </div>
        </div>
      )}
      {(summary || conclusion) && (
        <div className="summary-block">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0 }}>Summary</h3>
            {summary && (
              <button
                className="copy-btn summary-copy-btn"
                onClick={() => copyToClipboard(summary)}
                title="Copy summary to clipboard"
                style={{ background: 'rgba(99,102,241,0.12)', border: '1px solid #6366f1', borderRadius: 6, color: '#3730a3', fontWeight: 600 }}
              >
                📋
              </button>
            )}
          </div>
          <div><ReactMarkdown>{summary}</ReactMarkdown></div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 16 }}>
            <h4 style={{ margin: 0 }}>Conclusion</h4>
            {conclusion && (
              <button
                className="copy-btn summary-copy-btn"
                onClick={() => copyToClipboard(conclusion)}
                title="Copy conclusion to clipboard"
                style={{ background: 'rgba(99,102,241,0.12)', border: '1px solid #6366f1', borderRadius: 6, color: '#3730a3', fontWeight: 600 }}
              >
                📋
              </button>
            )}
          </div>
          <div><ReactMarkdown>{conclusion}</ReactMarkdown></div>
        </div>
      )}
    </div>
  );
}

export default App;
