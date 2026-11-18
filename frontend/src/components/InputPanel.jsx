
import { useState, useRef } from "react";

export const InputPanel = () => {
  const [prompt, setPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [percent, setPercent] = useState(0);
  const [liveText, setLiveText] = useState("");
  const [stopped, setStopped] = useState(false);
  const [copied, setCopied] = useState(false);
  const [pasted, setPasted] = useState(false);
  const [error, setError] = useState("");

  const controllerRef = useRef(null);

  const handleSubmit = async () => {
    if (!prompt.trim() || isGenerating) return;

    setIsGenerating(true);
    setPercent(0);
    setLiveText("");
    setStopped(false);
    setError("");

    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/generate-stream",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ prompt }),
          signal: controller.signal,
        }
      );

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      if (!response.body) {
        throw new Error("Streaming response unavailable");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";

      const processLine = (line) => {
        if (!line.trim()) return;

        const message = JSON.parse(line);

        if (typeof message.percent === "number") {
          setPercent(message.percent);
        }

        if (typeof message.text_so_far === "string") {
          setLiveText(message.text_so_far);
        }
      };

      while (true) {
        const { value, done } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          processLine(line);
        }
      }

      buffer += decoder.decode();

      if (buffer.trim()) {
        processLine(buffer);
      }

    } catch (err) {
      if (err.name === "AbortError") {
        setStopped(true);
      } else {
        console.error(err);
        setError("Unable to generate text. Please check your backend.");
      }
    } finally {
      setIsGenerating(false);
      controllerRef.current = null;
    }
  };

  const handleStop = () => {
    controllerRef.current?.abort();
  };

  const handlePaste = async () => {
    try {
      const clipboardText = await navigator.clipboard.readText();

      setPrompt(clipboardText);
      setPasted(true);

      setTimeout(() => setPasted(false), 1500);
    } catch (err) {
      console.error("Clipboard read failed:", err);
      setError("Unable to access clipboard. Please paste manually.");
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(liveText);

      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      console.error("Clipboard write failed:", err);
      setError("Unable to copy the generated text.");
    }
  };

  return (
    <div className="input-page">

      <div className="page-heading">
        <div>
          <span className="page-eyebrow">TEXT GENERATION</span>

          <h2>Generate text</h2>

          <p>
            Enter a prompt to generate text and begin
            the watermarking process.
          </p>
        </div>
      </div>

      <section className="prompt-section">

        <div className="section-heading">
          <div>
            <h3>Input prompt</h3>
            <p>What would you like the language model to generate?</p>
          </div>

          <span className="character-count">
            {prompt.length} characters
          </span>
        </div>

        <div className="editor">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Start typing your prompt..."
            disabled={isGenerating}
          />

          <div className="editor-bottom">
            <span>Plain text</span>

            <button
              type="button"
              onClick={handlePaste}
              disabled={isGenerating}
            >
              {pasted ? "Pasted ✓" : "Paste"}
            </button>
          </div>
        </div>

        {error && (
          <p className="error-message">{error}</p>
        )}

        <div className="action-row">
          <span className="helper-text">
            Make your prompt clear and descriptive.
          </span>

          <div className="action-buttons">

            {isGenerating && (
              <button
                className="stop-button"
                onClick={handleStop}
              >
                Stop
              </button>
            )}

            <button
              className="generate-button"
              onClick={handleSubmit}
              disabled={isGenerating || !prompt.trim()}
            >
              {isGenerating ? "Generating..." : "Generate text"}
              {!isGenerating && <span>→</span>}
            </button>

          </div>
        </div>

      </section>

      {(isGenerating || liveText) && (
        <section className="output-section">

          <div className="output-heading">
            <div>
              <span className="output-label">MODEL RESPONSE</span>
              <h3>Generated output</h3>
            </div>

            <span className={`output-status ${
              stopped ? "stopped" :
              isGenerating ? "running" : "completed"
            }`}>
              <span className="status-dot"></span>
              {stopped
                ? "Stopped"
                : isGenerating
                ? "Generating"
                : "Completed"}
            </span>
          </div>

          <div className="progress-info">
            <span>Progress</span>
            <span>{percent}%</span>
          </div>

          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${percent}%` }}
            />
          </div>

          <div className="output-box">
            <div className="output-box-header">
              <span>OUTPUT</span>

              {liveText && (
                <button onClick={handleCopy}>
                  {copied ? "Copied ✓" : "Copy"}
                </button>
              )}
            </div>

            <div className="output-content">
              {liveText || "Waiting for model response..."}
            </div>
          </div>

        </section>
      )}

    </div>
  );
};
