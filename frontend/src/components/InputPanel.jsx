import { useState, useRef } from "react";

export const InputPanel = () => {
    const [prompt, setPrompt] = useState("");
    const [isGenerating, setIsGenerating] = useState(false);
    const [percent, setPercent] = useState(0);
    const [liveText, setLiveText] = useState("");
    const [stopped, setStopped] = useState(false);
    const [copied, setCopied] = useState(false);
    const [pasted, setPasted] = useState(false);

    const controllerRef = useRef(null);

    const handleSubmit = async () => {
        setIsGenerating(true);
        setPercent(0);
        setLiveText("");
        setStopped(false);

        const controller = new AbortController();
        controllerRef.current = controller;

        try {
            const response = await fetch("http://127.0.0.1:8000/generate-stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt }),
                signal: controller.signal,
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    const message = JSON.parse(line);
                    setPercent(message.percent);
                    setLiveText(message.text_so_far);
                }
            }
        } catch (err) {
            if (err.name === "AbortError") {
                setStopped(true);
            } else {
                console.error(err);
            }
        } finally {
            setIsGenerating(false);
            controllerRef.current = null;
        }
    };

    const handleStop = () => {
        if (controllerRef.current) {
            controllerRef.current.abort();
        }
    };

    const handlePaste = async () => {
        try {
            const clipboardText = await navigator.clipboard.readText();
            setPrompt(clipboardText);
            setPasted(true);
            setTimeout(() => setPasted(false), 1500);
        } catch (err) {
            console.error("Clipboard read failed:", err);
        }
    };

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(liveText);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        } catch (err) {
            console.error("Clipboard write failed:", err);
        }
    };

    return (
        <div className="flex-1 flex flex-col gap-5 m-3 p-6 rounded-3xl border-2 border-cyan-500/60 bg-black shadow-[0_0_25px_rgba(6,182,212,0.15)]">
            <div>
                <h2 className="text-2xl font-bold text-slate-100">
                    LLM Input Prompt
                </h2>
                <p className="mt-1 text-sm text-cyan-300">
                    Paste or type the text you want WatermarkX to generate and analyze.
                </p>
            </div>

            <div className="relative">
                <textarea
                    placeholder="Enter your prompt here..."
                    className="h-64 w-full resize-none rounded-2xl border border-cyan-500/40 bg-black p-4 text-slate-100 placeholder:text-slate-500 focus:border-cyan-400 focus:shadow-[0_0_15px_rgba(6,182,212,0.35)] focus:outline-none disabled:opacity-50 transition"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    disabled={isGenerating}
                />
                <button
                    onClick={handlePaste}
                    disabled={isGenerating}
                    className="absolute top-3 right-3 cursor-pointer rounded-lg border border-cyan-500/40 bg-black px-3 py-1 text-xs text-cyan-300 hover:border-cyan-400 hover:shadow-[0_0_10px_rgba(6,182,212,0.5)] disabled:opacity-50 transition"
                >
                    {pasted ? "Pasted!" : "Paste"}
                </button>
            </div>

            <div className="flex gap-3 self-end">
                {isGenerating && (
                    <button
                        onClick={handleStop}
                        className="w-32 cursor-pointer rounded-2xl border-2 border-red-500 bg-black py-3 text-lg font-semibold text-red-400 transition hover:shadow-[0_0_20px_rgba(239,68,68,0.5)]"
                    >
                        Stop
                    </button>
                )}
                <button
                    onClick={handleSubmit}
                    disabled={isGenerating || !prompt}
                    className="w-52 cursor-pointer rounded-2xl border-2 border-cyan-500/70 bg-black py-3 text-lg font-semibold text-cyan-300 transition hover:shadow-[0_0_20px_rgba(6,182,212,0.6)] disabled:opacity-40 disabled:cursor-not-allowed"
                >
                    {isGenerating ? "Generating..." : "Submit Prompt"}
                </button>
            </div>

            {(isGenerating || liveText) && (
                <div className="mt-2">
                    <div className="flex justify-between text-sm text-cyan-300 mb-1">
                        <span>{stopped ? "Stopped" : isGenerating ? "Generating..." : "Done"}</span>
                        <span>{percent}%</span>
                    </div>
                    <div className="w-full h-3 rounded-full bg-black border border-cyan-500/30 overflow-hidden">
                        <div
                            className="h-full bg-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.8)] transition-all duration-150"
                            style={{ width: `${percent}%` }}
                        />
                    </div>
                    <div className="relative mt-4">
                        <p className="rounded-xl border border-cyan-500/30 bg-black p-3 pr-16 text-slate-300 text-sm whitespace-pre-wrap min-h-16">
                            {liveText}
                        </p>
                        {liveText && (
                            <button
                                onClick={handleCopy}
                                className="absolute top-3 right-3 cursor-pointer rounded-lg border border-cyan-500/40 bg-black px-3 py-1 text-xs text-cyan-300 hover:border-cyan-400 hover:shadow-[0_0_10px_rgba(6,182,212,0.5)] transition"
                            >
                                {copied ? "Copied!" : "Copy"}
                            </button>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};