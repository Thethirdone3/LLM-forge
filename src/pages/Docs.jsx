import { useState } from "react";
import { Link } from "react-router-dom";

const SECTIONS = [
  {
    id: "quickstart",
    title: "Quick Start",
    icon: "⚡",
    content: [
      {
        type: "text",
        body: "LLM Forge takes you from zero to a fine-tuned model in 5 steps. No Python knowledge required to configure — but you'll need Python installed to run the generated script.",
      },
      {
        type: "steps",
        items: [
          { n: "1", title: "Open the wizard", body: "Click 'Launch Forge' in the nav. The wizard walks you through each step." },
          { n: "2", title: "Choose your task", body: "Select Chatbot, Q&A Bot, Instruction Follower, or Creative Writer. Write a system prompt describing your bot's persona and behavior." },
          { n: "3", title: "Upload training data", body: "Drop a .jsonl file. Each line is one JSON object — either a conversation with a messages array, or a plain text field." },
          { n: "4", title: "Pick a model", body: "GPT-2 runs on CPU. TinyLlama needs ~4GB VRAM. Phi-2 needs ~8GB VRAM. Start small if you're experimenting." },
          { n: "5", title: "Export and run", body: "Download train.py and requirements.txt. Run them locally. A FastAPI inference server starts automatically after training." },
        ],
      },
    ],
  },
  {
    id: "data",
    title: "Training Data",
    icon: "📂",
    content: [
      {
        type: "text",
        body: "LLM Forge accepts .jsonl files — one JSON object per line. Two formats are supported:",
      },
      {
        type: "code",
        label: "Conversation format (recommended for chatbots)",
        code: `{"messages":[{"role":"user","content":"What hours are you open?"},{"role":"assistant","content":"We're open Monday–Friday, 9am–6pm."}]}
{"messages":[{"role":"user","content":"Do you offer refunds?"},{"role":"assistant","content":"Yes, within 30 days of purchase."}]}`,
      },
      {
        type: "code",
        label: "Text format (for creative writing or document-style tasks)",
        code: `{"text":"Fine-tuning is the process of continuing to train a pre-trained model on a smaller, domain-specific dataset."}
{"text":"LoRA (Low-Rank Adaptation) reduces the number of trainable parameters by decomposing weight updates into low-rank matrices."}`,
      },
      {
        type: "callout",
        color: "#f59e0b",
        body: "Minimum recommended: 50 examples. Sweet spot: 200–500. Quality matters more than quantity — 100 great examples beats 1,000 mediocre ones.",
      },
    ],
  },
  {
    id: "models",
    title: "Base Models",
    icon: "🤖",
    content: [
      {
        type: "text",
        body: "LLM Forge supports four base models. All are downloaded from HuggingFace on first run.",
      },
      {
        type: "table",
        headers: ["Model", "Params", "VRAM", "Best for"],
        rows: [
          ["GPT-2", "124M", "~1 GB (CPU ok)", "Quick experiments, no GPU needed"],
          ["GPT-2 Medium", "355M", "~2 GB", "Better quality, still runs on most laptops"],
          ["TinyLlama 1.1B", "1.1B", "~4 GB", "Modern architecture, instruction-tuned, great for chatbots"],
          ["Phi-2", "2.7B", "~8 GB", "Best reasoning quality, needs a real GPU"],
        ],
      },
    ],
  },
  {
    id: "hyperparams",
    title: "Hyperparameters",
    icon: "🎛️",
    content: [
      {
        type: "text",
        body: "Every hyperparameter in the wizard has an in-app tooltip. Here's a deeper reference:",
      },
      {
        type: "table",
        headers: ["Parameter", "Default", "What it controls", "When to change"],
        rows: [
          ["Epochs", "3", "How many times the model sees your full dataset", "Increase to 5–7 if responses feel generic. Decrease if the model sounds robotic."],
          ["Batch Size", "2", "Samples processed before a weight update", "Keep at 1–2 on consumer GPUs. Only increase with 16GB+ VRAM."],
          ["Learning Rate", "2e-4", "Step size for weight updates", "Almost never change this. 2e-4 is the LoRA sweet spot."],
          ["Max Length", "512", "Max tokens per training example", "Use 1024+ for multi-turn conversations or long documents."],
          ["LoRA r", "16", "Size of LoRA adapter matrices", "r=8 for tiny datasets. r=32–64 for complex tasks with lots of data."],
          ["LoRA alpha", "32", "Scaling factor for LoRA influence", "Keep at 2× your r value. Not worth tuning independently."],
        ],
      },
      {
        type: "callout",
        color: "#10b981",
        body: "Safe starting config: Epochs 3, Batch 2, LR 2e-4, Max Length 512, LoRA r=16, alpha=32. These work for 90% of fine-tuning tasks.",
      },
    ],
  },
  {
    id: "exports",
    title: "Export Formats",
    icon: "📦",
    content: [
      {
        type: "text",
        body: "LLM Forge generates six files. Here's what each one does and when to use it.",
      },
      {
        type: "table",
        headers: ["File", "Purpose", "Run with"],
        rows: [
          ["train.py", "Full training pipeline + FastAPI server", "python train.py"],
          ["requirements.txt", "All Python dependencies", "pip install -r requirements.txt"],
          ["Dockerfile", "Containerized training + inference", "docker build -t my-llm . && docker run --gpus all -p 8000:8000 my-llm"],
          ["notebook.ipynb", "Interactive Jupyter notebook", "jupyter notebook notebook.ipynb"],
          ["server.js", "Node.js proxy + embedded chat UI", "node server.js (after train.py is running)"],
          ["README.md", "Setup guide with your config", "Read in GitHub or any markdown viewer"],
        ],
      },
      {
        type: "callout",
        color: "#6366f1",
        body: "Start with train.py + requirements.txt. That's all you need to train and serve your model. The other formats are optional extras.",
      },
    ],
  },
  {
    id: "running",
    title: "Running Locally",
    icon: "🖥️",
    content: [
      {
        type: "text",
        body: "After exporting, run your model on your own machine in three commands:",
      },
      {
        type: "code",
        label: "Install dependencies",
        code: `pip install -r requirements.txt`,
      },
      {
        type: "code",
        label: "Train the model (this downloads the base model on first run)",
        code: `python train.py`,
      },
      {
        type: "code",
        label: "Test the inference API",
        code: `curl -X POST http://localhost:8000/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Hello! Who are you?"}'`,
      },
      {
        type: "callout",
        color: "#f59e0b",
        body: "Training time depends on your hardware and dataset size. GPT-2 with 100 examples trains in ~5 minutes on CPU. TinyLlama with 500 examples takes ~30 minutes on an RTX 3080.",
      },
    ],
  },
  {
    id: "cloud",
    title: "Cloud Training",
    icon: "☁️",
    content: [
      {
        type: "text",
        body: "Don't have a GPU? Run training on a rented cloud GPU for $0.50–2.00 per session.",
      },
      {
        type: "steps",
        items: [
          { n: "1", title: "Export your files from LLM Forge", body: "Download train.py and requirements.txt." },
          { n: "2", title: "Sign up for RunPod or Lambda Labs", body: "Both offer pay-per-hour GPU rentals. RunPod is slightly cheaper. Lambda Labs has a cleaner interface." },
          { n: "3", title: "Launch a GPU instance", body: "Pick any RTX 3090 or A4000 instance. 24GB VRAM handles all four models." },
          { n: "4", title: "Upload your files and training data", body: "Use the built-in file manager or scp." },
          { n: "5", title: "Run the training script", body: "pip install -r requirements.txt && python train.py. Done in 15–60 minutes." },
          { n: "6", title: "Download your model", body: "The trained model is saved to ./model-output. Download it and run it locally." },
        ],
      },
      {
        type: "callout",
        color: "#0ea5e9",
        body: "Typical cost: $1–3 for a full fine-tuning run. Shut down the instance immediately after training to avoid ongoing charges.",
      },
    ],
  },
];

// ── Renderers ─────────────────────────────────────────────────────────────────

function RenderContent({ blocks }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {blocks.map((block, i) => {
        if (block.type === "text") {
          return <p key={i} style={{ color: "#888", fontSize: 14, lineHeight: 1.8 }}>{block.body}</p>;
        }
        if (block.type === "code") {
          return (
            <div key={i}>
              {block.label && <div style={{ fontSize: 11, color: "#555", textTransform: "uppercase", letterSpacing: "0.8px", marginBottom: 8 }}>{block.label}</div>}
              <pre style={{ background: "#080810", border: "1px solid #1e1e2a", borderRadius: 10, padding: "14px 16px", fontFamily: "'DM Mono', monospace", fontSize: 12, color: "#7a9", lineHeight: 1.8, overflowX: "auto", whiteSpace: "pre-wrap", wordBreak: "break-all", margin: 0 }}>
                {block.code}
              </pre>
            </div>
          );
        }
        if (block.type === "callout") {
          return (
            <div key={i} style={{ background: block.color + "0d", border: `1px solid ${block.color}33`, borderRadius: 10, padding: "14px 16px", display: "flex", gap: 10 }}>
              <span style={{ color: block.color, fontSize: 16, flexShrink: 0, marginTop: 1 }}>💡</span>
              <p style={{ color: "#888", fontSize: 13, lineHeight: 1.75, margin: 0 }}>{block.body}</p>
            </div>
          );
        }
        if (block.type === "steps") {
          return (
            <div key={i} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {block.items.map((step, j) => (
                <div key={j} style={{ display: "flex", gap: 14 }}>
                  <div style={{ width: 26, height: 26, borderRadius: "50%", background: "#f59e0b18", border: "1px solid #f59e0b33", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: "#f59e0b", fontWeight: 700, fontFamily: "monospace", flexShrink: 0, marginTop: 1 }}>{step.n}</div>
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "#e2e2ea", marginBottom: 4 }}>{step.title}</div>
                    <div style={{ fontSize: 13, color: "#666", lineHeight: 1.7 }}>{step.body}</div>
                  </div>
                </div>
              ))}
            </div>
          );
        }
        if (block.type === "table") {
          return (
            <div key={i} style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    {block.headers.map(h => (
                      <th key={h} style={{ textAlign: "left", padding: "10px 12px", fontSize: 11, color: "#444", textTransform: "uppercase", letterSpacing: "0.8px", borderBottom: "1px solid #1e1e2a", fontWeight: 700 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {block.rows.map((row, ri) => (
                    <tr key={ri}>
                      {row.map((cell, ci) => (
                        <td key={ci} style={{ padding: "11px 12px", color: ci === 0 ? "#e2e2ea" : "#666", borderBottom: ri < block.rows.length - 1 ? "1px solid #16161f" : "none", fontFamily: ci === 0 ? "'DM Mono', monospace" : "inherit", fontSize: ci === 0 ? 12 : 13, lineHeight: 1.6 }}>{cell}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        return null;
      })}
    </div>
  );
}

// ── Main Docs Page ─────────────────────────────────────────────────────────────

export default function Docs() {
  const [active, setActive] = useState("quickstart");
  const section = SECTIONS.find(s => s.id === active);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", maxWidth: 1060, margin: "0 auto", padding: "40px 32px", gap: 48, minHeight: "calc(100vh - 58px)" }}>

      {/* Sidebar */}
      <aside>
        <div style={{ position: "sticky", top: 80 }}>
          <div style={{ fontSize: 11, color: "#444", letterSpacing: "1px", textTransform: "uppercase", marginBottom: 16 }}>Documentation</div>
          <nav style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {SECTIONS.map(s => (
              <button key={s.id} onClick={() => setActive(s.id)}
                style={{ background: active === s.id ? "#f59e0b0d" : "transparent", border: `1px solid ${active === s.id ? "#f59e0b33" : "transparent"}`, borderRadius: 8, padding: "9px 12px", textAlign: "left", cursor: "pointer", display: "flex", gap: 10, alignItems: "center", transition: "all 0.15s" }}>
                <span style={{ fontSize: 14 }}>{s.icon}</span>
                <span style={{ fontSize: 13, color: active === s.id ? "#f59e0b" : "#555", fontWeight: active === s.id ? 600 : 400 }}>{s.title}</span>
              </button>
            ))}
          </nav>
          <div style={{ marginTop: 32, padding: "14px", background: "#0d0d14", border: "1px solid #1e1e2a", borderRadius: 10 }}>
            <div style={{ fontSize: 12, color: "#555", marginBottom: 10 }}>Ready to build?</div>
            <Link to="/forge" style={{ display: "block", textAlign: "center", background: "#f59e0b", color: "#000", borderRadius: 8, padding: "9px", fontSize: 12, fontWeight: 700 }}>
              Open the Wizard →
            </Link>
          </div>
        </div>
      </aside>

      {/* Content */}
      <main style={{ minWidth: 0 }}>
        <div className="fade-up" key={active}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
            <span style={{ fontSize: 28 }}>{section.icon}</span>
            <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 32, color: "#e2e2ea" }}>{section.title}</h1>
          </div>
          <div style={{ background: "#0d0d14", border: "1px solid #1e1e2a", borderRadius: 14, padding: "28px 28px" }}>
            <RenderContent blocks={section.content} />
          </div>

          {/* Prev/Next */}
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 28, gap: 12 }}>
            {SECTIONS.findIndex(s => s.id === active) > 0 ? (
              <button onClick={() => setActive(SECTIONS[SECTIONS.findIndex(s => s.id === active) - 1].id)}
                style={{ background: "transparent", border: "1px solid #2a2a38", borderRadius: 9, padding: "10px 18px", color: "#888", fontSize: 13, cursor: "pointer", display: "flex", gap: 8, alignItems: "center" }}>
                ← {SECTIONS[SECTIONS.findIndex(s => s.id === active) - 1].title}
              </button>
            ) : <div />}
            {SECTIONS.findIndex(s => s.id === active) < SECTIONS.length - 1 && (
              <button onClick={() => setActive(SECTIONS[SECTIONS.findIndex(s => s.id === active) + 1].id)}
                style={{ background: "#f59e0b0d", border: "1px solid #f59e0b33", borderRadius: 9, padding: "10px 18px", color: "#f59e0b", fontSize: 13, cursor: "pointer", display: "flex", gap: 8, alignItems: "center" }}>
                {SECTIONS[SECTIONS.findIndex(s => s.id === active) + 1].title} →
              </button>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
