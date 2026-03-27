import { useState, useRef, useEffect } from "react";

// ── Constants ─────────────────────────────────────────────────────────────────

const BASE_MODELS = [
  { id: "gpt2", label: "GPT-2", size: "124M params", desc: "Great for quick experiments, runs on CPU", vram: "~1 GB" },
  { id: "gpt2-medium", label: "GPT-2 Medium", size: "355M params", desc: "Better quality, still lightweight", vram: "~2 GB" },
  { id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", label: "TinyLlama 1.1B", size: "1.1B params", desc: "Instruction-tuned, modern architecture", vram: "~4 GB" },
  { id: "microsoft/phi-2", label: "Phi-2", size: "2.7B params", desc: "Excellent reasoning for its size", vram: "~8 GB" },
];

const TASK_TYPES = [
  { id: "chat", label: "Chatbot", icon: "💬", desc: "Conversational assistant with a custom persona" },
  { id: "qa", label: "Q&A Bot", icon: "❓", desc: "Answers questions from a knowledge domain" },
  { id: "instruct", label: "Instruction Follower", icon: "📋", desc: "Follows specific task instructions" },
  { id: "creative", label: "Creative Writer", icon: "✍️", desc: "Generates creative text in a specific style" },
];

const STEPS = ["Task", "Data", "Model", "Training", "Export"];

const HYPERPARAM_INFO = {
  epochs: { title: "Epochs", icon: "🔁", what: "One epoch = the model sees your entire dataset once from start to finish.", effect: "More epochs = more learning, but too many causes overfitting — the model memorizes your data instead of generalizing.", tip: "Start with 3. If responses feel generic, try 5–7. If the model sounds robotic or repetitive, reduce it.", range: "Typical: 2–5 for small datasets, 1–3 for large." },
  batchSize: { title: "Batch Size", icon: "📦", what: "How many training samples are processed at once before the model updates its weights.", effect: "Larger batches = faster training but higher GPU memory usage. Smaller = slower but works on less hardware.", tip: "Use 1–2 on a laptop or consumer GPU. Use 4–8 only if you have 16 GB+ VRAM.", range: "Typical: 1–4." },
  learningRate: { title: "Learning Rate", icon: "📈", what: "Controls how large a step the model takes when adjusting weights after each batch.", effect: "Too high = unstable training (model forgets everything). Too low = training barely progresses.", tip: "2e-4 is a safe default for LoRA. Only adjust if your training loss isn't steadily decreasing.", range: "Typical for LoRA: 1e-4 to 5e-4." },
  maxLength: { title: "Max Sequence Length", icon: "📏", what: "The maximum number of tokens (roughly words) allowed per training example.", effect: "Longer = handles longer conversations, but memory usage grows. Short examples get zero-padded.", tip: "512 works for short Q&A. Use 1024–2048 for long documents or multi-turn conversations.", range: "Typical: 256–1024. Match to your actual data." },
  loraR: { title: "LoRA Rank (r)", icon: "🧬", what: "Sets the size of low-rank adapter matrices inserted into the model layers.", effect: "Higher rank = more trainable parameters = more capacity, but slower and more memory.", tip: "r=16 is a great default. Use r=8 for tiny datasets. Use r=32–64 for complex tasks with lots of data.", range: "Typical: 8–32. Use powers of 2." },
  loraAlpha: { title: "LoRA Alpha", icon: "⚖️", what: "A scaling factor applied to LoRA adapters — controls how strongly they influence the output.", effect: "Higher alpha = stronger LoRA influence. Usually kept at 2× the rank value.", tip: "Set to 2× your LoRA rank. e.g. r=16 → alpha=32. Rarely needs independent tuning.", range: "Typical: 16–64. Rule: alpha = 2 × r." },
};

// ── Code Generators ───────────────────────────────────────────────────────────

function genPython(c) {
  return `#!/usr/bin/env python3
"""LLM Forge — Fine-tuning Script
Bot: ${c.botName || "My Assistant"} | Model: ${c.baseModel}"""

import json, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PM
import uvicorn

MODEL_NAME    = "${c.baseModel}"
OUTPUT_DIR    = "./model-output"
EPOCHS        = ${c.epochs}
BATCH_SIZE    = ${c.batchSize}
LR            = ${c.learningRate}
MAX_LENGTH    = ${c.maxLength}
LORA_R        = ${c.loraR}
LORA_ALPHA    = ${c.loraAlpha}
SYSTEM_PROMPT = """${c.systemPrompt}"""

with open("training_data.jsonl") as f:
    raw = [json.loads(l) for l in f if l.strip()]
print(f"Loaded {len(raw)} training samples")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto", trust_remote_code=True,
)

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    target_modules=["q_proj","v_proj"] if any(x in MODEL_NAME.lower() for x in ["llama","phi"]) else ["c_attn"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

def fmt(s):
    if "messages" in s:
        t = SYSTEM_PROMPT + "\\n"
        for m in s["messages"]:
            t += f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}\\n"
        return {"text": t}
    return {"text": SYSTEM_PROMPT + "\\n" + s.get("text", "")}

ds = Dataset.from_list([fmt(s) for s in raw])
tokenized = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"), batched=True, remove_columns=["text"])

trainer = Trainer(
    model=model, train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, learning_rate=LR,
        fp16=torch.cuda.is_available(), logging_steps=10,
        save_strategy="epoch", report_to="none",
    ),
)
print("\\n🚀 Training...\\n")
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\\n✅ Saved to {OUTPUT_DIR}")

# ── Inference server ──────────────────────────────────────────────────────────
app = FastAPI(title="${c.botName || "My LLM"} API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatReq(PM):
    message: str
    history: list = []

@app.post("/chat")
def chat(req: ChatReq):
    prompt = SYSTEM_PROMPT + "\\n"
    for h in req.history[-6:]:
        prompt += f"{'User' if h['role']=='user' else 'Assistant'}: {h['content']}\\n"
    prompt += f"User: {req.message}\\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.7,
                              do_sample=True, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"reply": reply.split("User:")[0].strip()}

@app.get("/health")
def health(): return {"status": "ok", "model": MODEL_NAME}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
`;
}

function genRequirements() {
  return `torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
datasets>=2.14.0
accelerate>=0.24.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
sentencepiece>=0.1.99
bitsandbytes>=0.41.0
`;
}

function genDockerfile(c) {
  return `# LLM Forge — Dockerfile
# Bot: ${c.botName || "My Assistant"} | Model: ${c.baseModel}
# Build: docker build -t my-llm .
# Run:   docker run --gpus all -p 8000:8000 -v $(pwd):/workspace my-llm

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
WORKDIR /workspace

RUN pip install --no-cache-dir \\
    transformers>=4.35.0 peft>=0.6.0 datasets>=2.14.0 \\
    accelerate>=0.24.0 fastapi uvicorn pydantic \\
    sentencepiece bitsandbytes

COPY training_data.jsonl .
COPY train.py .

ENV MODEL_NAME="${c.baseModel}"
ENV EPOCHS="${c.epochs}"
ENV BATCH_SIZE="${c.batchSize}"
ENV LR="${c.learningRate}"
ENV MAX_LENGTH="${c.maxLength}"

EXPOSE 8000
CMD ["python", "train.py"]
`;
}

function genNodeJS(c) {
  return `// LLM Forge — Node.js Inference Server
// Bot: ${c.botName || "My Assistant"}
// npm install express cors node-fetch@2
// node server.js   (run train.py first!)

const express = require("express");
const cors = require("cors");
const fetch = require("node-fetch");
const app = express();
app.use(cors()); app.use(express.json());
const BACKEND = "http://localhost:8000";

app.post("/chat", async (req, res) => {
  try {
    const r = await fetch(BACKEND + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    res.json(await r.json());
  } catch {
    res.status(503).json({ error: "Run train.py first to start the Python server!" });
  }
});

app.get("/health", async (req, res) => {
  try {
    const r = await fetch(BACKEND + "/health");
    res.json({ node: "ok", python: (await r.json()).status });
  } catch {
    res.json({ node: "ok", python: "unreachable — run train.py" });
  }
});

app.listen(3001, () => console.log("🌐 http://localhost:3001"));
`;
}

function genReadme(c) {
  return `# ${c.botName || "My LLM"} — Fine-tuning Project
> Generated by LLM Forge

## Quick Start

\`\`\`bash
pip install -r requirements.txt
python train.py
\`\`\`

## Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | \`${c.baseModel}\` |
| Task Type | ${c.taskType} |
| Epochs | ${c.epochs} |
| Batch Size | ${c.batchSize} |
| Learning Rate | ${c.learningRate} |
| Max Sequence Length | ${c.maxLength} |
| LoRA Rank (r) | ${c.loraR} |
| LoRA Alpha | ${c.loraAlpha} |

## System Prompt

\`\`\`
${c.systemPrompt}
\`\`\`

## Files

- \`train.py\` — Training script + FastAPI inference server
- \`server.js\` — Node.js proxy server (optional)
- \`Dockerfile\` — Containerized training
- \`requirements.txt\` — Python dependencies
- \`training_data.jsonl\` — Your training data (add this yourself)

## After Training

The model is saved to \`./model-output\`. The training script also
starts a FastAPI server at \`http://localhost:8000/chat\`.

Test it:
\`\`\`bash
curl -X POST http://localhost:8000/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Hello!"}'
\`\`\`
`;
}

function genJupyter(c) {
  const cells = [
    { type: "markdown", source: `# 🔬 LLM Forge Notebook\n**Bot:** ${c.botName || "My Assistant"} | **Model:** \`${c.baseModel}\`` },
    { type: "code", source: `!pip install transformers peft datasets accelerate torch fastapi uvicorn pydantic sentencepiece -q` },
    { type: "code", source: `import json, torch\nfrom datasets import Dataset\nfrom transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling\nfrom peft import LoraConfig, get_peft_model, TaskType\n\nMODEL_NAME = "${c.baseModel}"\nOUTPUT_DIR = "./model-output"\nEPOCHS = ${c.epochs}; BATCH_SIZE = ${c.batchSize}; LR = ${c.learningRate}\nMAX_LENGTH = ${c.maxLength}; LORA_R = ${c.loraR}; LORA_ALPHA = ${c.loraAlpha}\nSYSTEM_PROMPT = """${c.systemPrompt}"""\nprint("Config ready ✅")` },
    { type: "markdown", source: `## Load Data\nUpload your \`training_data.jsonl\` file.` },
    { type: "code", source: `with open("training_data.jsonl") as f:\n    raw = [json.loads(l) for l in f if l.strip()]\nprint(f"Loaded {len(raw)} samples")` },
    { type: "markdown", source: `## Load Model + Apply LoRA` },
    { type: "code", source: `tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)\nif tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token\nmodel = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto", trust_remote_code=True)\nlora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05)\nmodel = get_peft_model(model, lora_cfg)\nmodel.print_trainable_parameters()` },
    { type: "markdown", source: `## Train` },
    { type: "code", source: `def fmt(s):\n    if "messages" in s:\n        t = SYSTEM_PROMPT + "\\n"\n        for m in s["messages"]: t += f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}\\n"\n        return {"text": t}\n    return {"text": SYSTEM_PROMPT + "\\n" + s.get("text", "")}\n\nds = Dataset.from_list([fmt(s) for s in raw])\ntokenized = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"), batched=True, remove_columns=["text"])\ntrainer = Trainer(model=model, train_dataset=tokenized, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),\n    args=TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE, learning_rate=LR, fp16=torch.cuda.is_available(), logging_steps=10, save_strategy="epoch", report_to="none"))\ntrainer.train()\nmodel.save_pretrained(OUTPUT_DIR); tokenizer.save_pretrained(OUTPUT_DIR)\nprint("✅ Done!")` },
    { type: "code", source: `def chat(msg, history=[]):\n    prompt = SYSTEM_PROMPT + "\\n"\n    for h in history[-6:]: prompt += f"{'User' if h['role']=='user' else 'Assistant'}: {h['content']}\\n"\n    prompt += f"User: {msg}\\nAssistant:"\n    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)\n    with torch.no_grad(): out = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)\n    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).split("User:")[0].strip()\n\nprint("Bot:", chat("Hello! Who are you?"))` },
  ];
  return JSON.stringify({
    nbformat: 4, nbformat_minor: 5,
    metadata: { kernelspec: { display_name: "Python 3", language: "python", name: "python3" }, language_info: { name: "python", version: "3.10.0" } },
    cells: cells.map((cell, i) => ({ id: `cell-${i}`, cell_type: cell.type, metadata: {}, source: cell.source, ...(cell.type === "code" ? { outputs: [], execution_count: null } : {}) })),
  }, null, 2);
}

const EXPORT_FORMATS = [
  { id: "python",  label: "Python Script",    ext: "train.py",          icon: "🐍", badge: "Recommended", badgeColor: "#10b981", desc: "Full training pipeline + FastAPI inference server.",   gen: genPython },
  { id: "req",     label: "Requirements",     ext: "requirements.txt",  icon: "📦", badge: "Essential",   badgeColor: "#6366f1", desc: "All Python dependencies.",                             gen: genRequirements },
  { id: "docker",  label: "Dockerfile",       ext: "Dockerfile",        icon: "🐳", badge: "Portable",    badgeColor: "#0ea5e9", desc: "Containerized training + inference, GPU-ready.",        gen: genDockerfile },
  { id: "jupyter", label: "Jupyter Notebook", ext: "notebook.ipynb",    icon: "📓", badge: "Interactive", badgeColor: "#f59e0b", desc: "Step-by-step notebook with explanations.",              gen: genJupyter },
  { id: "node",    label: "Node.js Server",   ext: "server.js",         icon: "💚", badge: "Frontend",    badgeColor: "#84cc16", desc: "Express proxy + embedded chat UI.",                     gen: genNodeJS },
  { id: "readme",  label: "README",           ext: "README.md",         icon: "📄", badge: "Docs",        badgeColor: "#94a3b8", desc: "Setup guide with your config parameters.",              gen: genReadme },
];

// ── UI Primitives ─────────────────────────────────────────────────────────────

function Tooltip({ info }) {
  const [open, setOpen] = useState(false);
  const ref = useRef();
  useEffect(() => {
    const h = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);
  return (
    <div ref={ref} style={{ position: "relative", display: "inline-flex" }}>
      <button onClick={() => setOpen(o => !o)}
        style={{ width: 20, height: 20, borderRadius: "50%", flexShrink: 0, border: `1px solid ${open ? "#f59e0b" : "#2a2a3a"}`, background: open ? "#f59e0b22" : "#1a1a28", color: open ? "#f59e0b" : "#555", fontSize: 11, fontWeight: 700, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.15s" }}>?</button>
      {open && (
        <div style={{ position: "fixed", zIndex: 1000, background: "#14141f", border: "1px solid #2e2e44", borderRadius: 12, padding: "16px 18px", width: 310, boxShadow: "0 12px 40px #00000099" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, borderBottom: "1px solid #22222e", paddingBottom: 10 }}>
            <span style={{ fontSize: 18 }}>{info.icon}</span>
            <span style={{ color: "#f59e0b", fontWeight: 700, fontSize: 14 }}>{info.title}</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 9, fontSize: 13, lineHeight: 1.65 }}>
            <p style={{ color: "#888" }}><span style={{ color: "#ccc", fontWeight: 600 }}>What it is: </span>{info.what}</p>
            <p style={{ color: "#888" }}><span style={{ color: "#ccc", fontWeight: 600 }}>Effect: </span>{info.effect}</p>
            <div style={{ background: "#f59e0b0d", border: "1px solid #f59e0b22", borderRadius: 8, padding: "9px 12px" }}>
              <p style={{ color: "#ccc" }}><span style={{ color: "#f59e0b", fontWeight: 600 }}>💡 </span>{info.tip}</p>
            </div>
            <p style={{ color: "#555", fontSize: 12 }}>Range: {info.range}</p>
          </div>
        </div>
      )}
    </div>
  );
}

const Card = ({ children, style = {} }) => (
  <div style={{ background: "#13131a", border: "1px solid #22222e", borderRadius: 16, padding: "22px 26px", ...style }}>{children}</div>
);

const SecTitle = ({ children }) => (
  <div style={{ fontSize: 11, letterSpacing: "1px", textTransform: "uppercase", color: "#444", fontWeight: 600, marginBottom: 18 }}>{children}</div>
);

function Slider({ label, value, onChange, min, max, step, format, tooltip }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 9 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{ fontSize: 12, color: "#666", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.8px" }}>{label}</span>
          {tooltip && <Tooltip info={tooltip} />}
        </div>
        <span style={{ fontSize: 13, color: "#f59e0b", fontFamily: "'DM Mono', monospace", fontWeight: 600 }}>{format ? format(value) : value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: "#f59e0b", cursor: "pointer", display: "block" }} />
    </div>
  );
}

// ── Step Components ───────────────────────────────────────────────────────────

function StepTask({ config, setConfig }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, color: "#e2e2ea", marginBottom: 8 }}>What will your LLM do?</h2>
        <p style={{ color: "#555", fontSize: 14, lineHeight: 1.7 }}>Choose a task type — this shapes how training data is formatted and how the model learns to respond.</p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {TASK_TYPES.map(t => (
          <button key={t.id} onClick={() => setConfig(c => ({ ...c, taskType: t.id }))}
            style={{ background: config.taskType === t.id ? "#f59e0b0a" : "#0d0d14", border: `1px solid ${config.taskType === t.id ? "#f59e0b" : "#22222e"}`, borderRadius: 12, padding: "18px 20px", cursor: "pointer", textAlign: "left", transition: "all 0.2s" }}>
            <div style={{ fontSize: 24, marginBottom: 8 }}>{t.icon}</div>
            <div style={{ color: "#e2e2ea", fontWeight: 600, fontSize: 15, marginBottom: 4 }}>{t.label}</div>
            <div style={{ color: "#555", fontSize: 13 }}>{t.desc}</div>
          </button>
        ))}
      </div>
      <Card>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div>
            <div style={{ fontSize: 11, color: "#555", letterSpacing: "1px", textTransform: "uppercase", fontWeight: 600, marginBottom: 8 }}>Bot Name</div>
            <input value={config.botName} onChange={e => setConfig(c => ({ ...c, botName: e.target.value }))} placeholder="e.g. Aria, TechBot, LegalGPT..."
              style={{ width: "100%", background: "#0d0d14", border: "1px solid #22222e", borderRadius: 8, padding: "11px 14px", color: "#e2e2ea", fontFamily: "'DM Mono', monospace", fontSize: 13, outline: "none" }} />
          </div>
          <div>
            <div style={{ fontSize: 11, color: "#555", letterSpacing: "1px", textTransform: "uppercase", fontWeight: 600, marginBottom: 8 }}>System Prompt</div>
            <textarea value={config.systemPrompt} onChange={e => setConfig(c => ({ ...c, systemPrompt: e.target.value }))} placeholder="Describe persona, rules, and focus..." rows={4}
              style={{ width: "100%", background: "#0d0d14", border: "1px solid #22222e", borderRadius: 8, padding: "11px 14px", color: "#e2e2ea", fontFamily: "'DM Mono', monospace", fontSize: 13, outline: "none", resize: "vertical" }} />
          </div>
        </div>
      </Card>
    </div>
  );
}

function StepData({ config, setConfig }) {
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState([]);
  const fileRef = useRef();
  const handleFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
      const lines = e.target.result.trim().split("\n").filter(Boolean);
      setConfig(c => ({ ...c, trainingFile: file, trainingLineCount: lines.length }));
      setPreview(lines.slice(0, 3));
    };
    reader.readAsText(file);
  };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, color: "#e2e2ea", marginBottom: 8 }}>Upload your training data</h2>
        <p style={{ color: "#555", fontSize: 14, lineHeight: 1.7 }}>Provide a <code style={{ color: "#f59e0b", fontFamily: "'DM Mono', monospace" }}>.jsonl</code> file — one JSON object per line. Each line should be either a conversation with a <code style={{ color: "#f59e0b", fontFamily: "'DM Mono', monospace" }}>messages</code> array or a <code style={{ color: "#f59e0b", fontFamily: "'DM Mono', monospace" }}>text</code> field.</p>
      </div>
      <div onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
        onClick={() => fileRef.current.click()}
        style={{ border: `2px dashed ${dragging ? "#f59e0b" : config.trainingFile ? "#10b981" : "#2a2a38"}`, borderRadius: 14, padding: "40px 24px", textAlign: "center", cursor: "pointer", background: dragging ? "#f59e0b0a" : "#0d0d14", transition: "all 0.2s" }}>
        <input ref={fileRef} type="file" accept=".jsonl" style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />
        <div style={{ fontSize: 32, marginBottom: 12 }}>{config.trainingFile ? "✅" : "📂"}</div>
        {config.trainingFile ? (
          <>
            <div style={{ color: "#10b981", fontWeight: 600, fontSize: 15 }}>{config.trainingFile.name}</div>
            <div style={{ color: "#555", fontSize: 13, marginTop: 4 }}>{config.trainingLineCount} training examples</div>
          </>
        ) : (
          <>
            <div style={{ color: "#888", fontSize: 15 }}>Drop your .jsonl file here or click to browse</div>
            <div style={{ color: "#444", fontSize: 12, marginTop: 6 }}>Max 100MB · One JSON per line</div>
          </>
        )}
      </div>
      {preview.length > 0 && (
        <Card>
          <SecTitle>Preview (first 3 lines)</SecTitle>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {preview.map((line, i) => (
              <div key={i} style={{ background: "#08080f", borderRadius: 6, padding: "10px 14px", fontFamily: "'DM Mono', monospace", fontSize: 11, color: "#7a9", lineHeight: 1.6, wordBreak: "break-all" }}>
                {line.length > 180 ? line.slice(0, 180) + "…" : line}
              </div>
            ))}
          </div>
        </Card>
      )}
      <Card>
        <SecTitle>Data Format Examples</SecTitle>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {[
            { label: "Conversation format", code: `{"messages":[{"role":"user","content":"What is LoRA?"},{"role":"assistant","content":"LoRA is..."}]}` },
            { label: "Text format", code: `{"text":"Fine-tuning is the process of adapting a pre-trained model..."}` },
          ].map(ex => (
            <div key={ex.label}>
              <div style={{ fontSize: 11, color: "#555", marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.8px" }}>{ex.label}</div>
              <div style={{ background: "#08080f", borderRadius: 6, padding: "10px 14px", fontFamily: "'DM Mono', monospace", fontSize: 11, color: "#7a9", wordBreak: "break-all" }}>{ex.code}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function StepModel({ config, setConfig }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, color: "#e2e2ea", marginBottom: 8 }}>Choose your base model</h2>
        <p style={{ color: "#555", fontSize: 14, lineHeight: 1.7 }}>Pick a pre-trained model to fine-tune. Smaller = runs on your laptop. Larger = better quality, needs a GPU.</p>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {BASE_MODELS.map(m => (
          <button key={m.id} onClick={() => setConfig(c => ({ ...c, baseModel: m.id }))}
            style={{ background: config.baseModel === m.id ? "#f59e0b08" : "#0d0d14", border: `1px solid ${config.baseModel === m.id ? "#f59e0b" : "#22222e"}`, borderRadius: 12, padding: "16px 20px", cursor: "pointer", textAlign: "left", display: "flex", justifyContent: "space-between", alignItems: "center", transition: "all 0.2s" }}>
            <div>
              <div style={{ color: "#e2e2ea", fontWeight: 600, fontSize: 15, marginBottom: 4 }}>{m.label}</div>
              <div style={{ color: "#555", fontSize: 13 }}>{m.desc}</div>
            </div>
            <div style={{ textAlign: "right", flexShrink: 0, marginLeft: 16 }}>
              <div style={{ color: "#f59e0b", fontFamily: "'DM Mono', monospace", fontSize: 12, marginBottom: 2 }}>{m.vram} VRAM</div>
              <div style={{ color: "#444", fontSize: 11 }}>{m.size}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function StepTraining({ config, setConfig }) {
  const set = (key) => (val) => setConfig(c => ({ ...c, [key]: val }));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, color: "#e2e2ea", marginBottom: 8 }}>Configure training</h2>
        <p style={{ color: "#555", fontSize: 14, lineHeight: 1.7 }}>Adjust hyperparameters. Click the <strong style={{ color: "#f59e0b" }}>?</strong> next to any slider for an explanation of what it does.</p>
      </div>
      <Card>
        <SecTitle>Training Parameters</SecTitle>
        <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
          <Slider label="Epochs" value={config.epochs} onChange={set("epochs")} min={1} max={10} step={1} tooltip={HYPERPARAM_INFO.epochs} />
          <Slider label="Batch Size" value={config.batchSize} onChange={set("batchSize")} min={1} max={8} step={1} tooltip={HYPERPARAM_INFO.batchSize} />
          <Slider label="Learning Rate" value={config.learningRate} onChange={set("learningRate")} min={0.00001} max={0.001} step={0.00001} format={v => v.toExponential(0)} tooltip={HYPERPARAM_INFO.learningRate} />
          <Slider label="Max Sequence Length" value={config.maxLength} onChange={set("maxLength")} min={128} max={2048} step={128} tooltip={HYPERPARAM_INFO.maxLength} />
        </div>
      </Card>
      <Card>
        <SecTitle>LoRA Parameters</SecTitle>
        <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
          <Slider label="LoRA Rank (r)" value={config.loraR} onChange={set("loraR")} min={4} max={64} step={4} tooltip={HYPERPARAM_INFO.loraR} />
          <Slider label="LoRA Alpha" value={config.loraAlpha} onChange={set("loraAlpha")} min={8} max={128} step={8} tooltip={HYPERPARAM_INFO.loraAlpha} />
        </div>
      </Card>
      <Card style={{ background: "#f59e0b09", border: "1px solid #f59e0b22" }}>
        <div style={{ display: "flex", gap: 10 }}>
          <span>💡</span>
          <div style={{ color: "#888", fontSize: 13, lineHeight: 1.75 }}>
            <strong style={{ color: "#f59e0b" }}>Recommended starting config:</strong> Epochs 3, Batch Size 2, LR 2e-4, Max Length 512, LoRA r=16, alpha=32. These are safe defaults for most use cases.
          </div>
        </div>
      </Card>
    </div>
  );
}

function StepExport({ config }) {
  const [selected, setSelected] = useState(new Set(["python", "req"]));
  const [preview, setPreview] = useState("python");
  const [copied, setCopied] = useState(null);

  const toggle = (id) => setSelected(s => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const getContent = (fmt) => fmt.gen(config);
  const copy = (fmt) => { navigator.clipboard?.writeText(getContent(fmt)); setCopied(fmt.id); setTimeout(() => setCopied(null), 2000); };
  const download = (fmt) => {
    const blob = new Blob([getContent(fmt)], { type: "text/plain" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = fmt.ext; a.click();
  };
  const downloadAll = () => EXPORT_FORMATS.filter(f => selected.has(f.id)).forEach((f, i) => setTimeout(() => download(f), i * 150));
  const previewFmt = EXPORT_FORMATS.find(f => f.id === preview);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, color: "#e2e2ea", marginBottom: 8 }}>Export your project</h2>
        <p style={{ color: "#555", fontSize: 14, lineHeight: 1.7 }}>Your fine-tuning pipeline is ready. Download the files and run them on your own machine.</p>
      </div>
      <Card>
        <SecTitle>Configuration Summary</SecTitle>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {[
            { label: "Model", val: config.baseModel.split("/").pop() },
            { label: "Task", val: config.taskType },
            { label: "Epochs", val: config.epochs },
            { label: "LoRA r", val: config.loraR },
            { label: "Batch", val: config.batchSize },
          ].map(chip => (
            <div key={chip.label} style={{ background: "#f59e0b0d", border: "1px solid #f59e0b22", borderRadius: 20, padding: "5px 12px", display: "flex", gap: 6, alignItems: "center" }}>
              <span style={{ color: "#666", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.5px" }}>{chip.label}</span>
              <span style={{ color: "#f59e0b", fontFamily: "'DM Mono', monospace", fontSize: 12, fontWeight: 600 }}>{chip.val}</span>
            </div>
          ))}
        </div>
      </Card>
      <div>
        <SecTitle>Select Export Formats</SecTitle>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {EXPORT_FORMATS.map(fmt => {
            const isSel = selected.has(fmt.id);
            return (
              <div key={fmt.id} style={{ background: isSel ? "#f59e0b07" : "#0d0d14", border: `1px solid ${isSel ? "#f59e0b33" : "#1e1e2a"}`, borderRadius: 12, padding: "13px 15px", display: "flex", alignItems: "center", gap: 12, transition: "all 0.2s" }}>
                <div onClick={() => toggle(fmt.id)} style={{ width: 20, height: 20, borderRadius: 5, flexShrink: 0, border: `2px solid ${isSel ? "#f59e0b" : "#333"}`, background: isSel ? "#f59e0b" : "transparent", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", color: "#000", fontSize: 12, fontWeight: 800, transition: "all 0.15s" }}>{isSel ? "✓" : ""}</div>
                <span style={{ fontSize: 20 }}>{fmt.icon}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 2, flexWrap: "wrap" }}>
                    <span style={{ color: "#e2e2ea", fontWeight: 600, fontSize: 14 }}>{fmt.label}</span>
                    <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 20, background: fmt.badgeColor + "22", color: fmt.badgeColor }}>{fmt.badge}</span>
                    <span style={{ color: "#383848", fontFamily: "'DM Mono', monospace", fontSize: 11 }}>{fmt.ext}</span>
                  </div>
                  <div style={{ color: "#555", fontSize: 12 }}>{fmt.desc}</div>
                </div>
                <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                  <button onClick={() => setPreview(fmt.id)} style={{ background: preview === fmt.id ? "#f59e0b22" : "#1a1a28", border: `1px solid ${preview === fmt.id ? "#f59e0b55" : "#2a2a38"}`, borderRadius: 7, padding: "6px 12px", color: preview === fmt.id ? "#f59e0b" : "#555", fontSize: 12, cursor: "pointer" }}>Preview</button>
                  <button onClick={() => copy(fmt)} style={{ background: "#1a1a28", border: "1px solid #2a2a38", borderRadius: 7, padding: "6px 12px", color: copied === fmt.id ? "#10b981" : "#555", fontSize: 12, cursor: "pointer" }}>{copied === fmt.id ? "✓ Copied" : "Copy"}</button>
                  <button onClick={() => download(fmt)} style={{ background: "#f59e0b", border: "none", borderRadius: 7, padding: "6px 16px", color: "#000", fontSize: 13, fontWeight: 800, cursor: "pointer" }}>↓</button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      {selected.size > 1 && (
        <button onClick={downloadAll} style={{ background: "#f59e0b", border: "none", borderRadius: 10, padding: "13px 26px", color: "#000", fontSize: 14, fontWeight: 700, cursor: "pointer", alignSelf: "flex-start", boxShadow: "0 0 24px #f59e0b44" }}>
          ↓ Download All {selected.size} Files
        </button>
      )}
      {previewFmt && (
        <Card>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 17 }}>{previewFmt.icon}</span>
              <span style={{ color: "#888", fontFamily: "'DM Mono', monospace", fontSize: 13 }}>{previewFmt.ext}</span>
            </div>
            <button onClick={() => copy(previewFmt)} style={{ background: "#1a1a28", border: "1px solid #2a2a38", borderRadius: 7, padding: "6px 12px", color: copied === previewFmt.id ? "#10b981" : "#555", fontSize: 12, cursor: "pointer" }}>{copied === previewFmt.id ? "✓ Copied" : "Copy"}</button>
          </div>
          <div style={{ background: "#08080f", borderRadius: 8, padding: 16, fontFamily: "'DM Mono', monospace", fontSize: 11, color: "#7a9", lineHeight: 1.8, maxHeight: 380, overflowY: "auto", whiteSpace: "pre", wordBreak: "break-all" }}>
            {getContent(previewFmt)}
          </div>
        </Card>
      )}
    </div>
  );
}

// ── Main Forge Page ───────────────────────────────────────────────────────────

const DEFAULT_CONFIG = {
  botName: "", taskType: "chat",
  systemPrompt: "You are a helpful, honest, and concise assistant.",
  trainingFile: null, trainingLineCount: 0,
  baseModel: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  epochs: 3, batchSize: 2, learningRate: 0.0002,
  maxLength: 512, loraR: 16, loraAlpha: 32,
};

export default function Forge() {
  const [step, setStep] = useState(0);
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const canAdvance = () => step === 0 ? (config.taskType && config.systemPrompt.trim()) : true;
  const pages = [
    <StepTask config={config} setConfig={setConfig} />,
    <StepData config={config} setConfig={setConfig} />,
    <StepModel config={config} setConfig={setConfig} />,
    <StepTraining config={config} setConfig={setConfig} />,
    <StepExport config={config} />,
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "210px 1fr", maxWidth: 1100, margin: "0 auto", width: "100%", padding: "40px 32px", gap: 48 }}>
      {/* Sidebar */}
      <aside>
        <div style={{ position: "sticky", top: 80 }}>
          <div style={{ fontSize: 11, color: "#444", letterSpacing: "1px", textTransform: "uppercase", marginBottom: 20 }}>Progress</div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            {STEPS.map((s, i) => (
              <div key={s} style={{ display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, cursor: i < step ? "pointer" : "default" }} onClick={() => i < step && setStep(i)}>
                  <div style={{ width: 28, height: 28, borderRadius: "50%", flexShrink: 0, border: `2px solid ${i < step ? "#10b981" : i === step ? "#f59e0b" : "#1e1e2a"}`, background: i < step ? "#10b981" : i === step ? "#f59e0b18" : "transparent", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700, color: i < step ? "#fff" : i === step ? "#f59e0b" : "#333", transition: "all 0.3s" }}>
                    {i < step ? "✓" : i + 1}
                  </div>
                  <span style={{ fontSize: 14, color: i === step ? "#e2e2ea" : i < step ? "#10b981" : "#444", fontWeight: i === step ? 600 : 400 }}>{s}</span>
                </div>
                {i < STEPS.length - 1 && <div style={{ width: 2, height: 20, background: i < step ? "#10b981" : "#1a1a24", marginLeft: 13, marginTop: 2, marginBottom: 2, transition: "background 0.3s" }} />}
              </div>
            ))}
          </div>
          {config.botName && (
            <div style={{ marginTop: 32, padding: 14, background: "#13131a", borderRadius: 10, border: "1px solid #1e1e2a" }}>
              <div style={{ fontSize: 11, color: "#444", marginBottom: 6, letterSpacing: "1px", textTransform: "uppercase" }}>Building</div>
              <div style={{ color: "#f59e0b", fontWeight: 600 }}>{config.botName}</div>
              <div style={{ color: "#444", fontSize: 12, marginTop: 3 }}>{TASK_TYPES.find(t => t.id === config.taskType)?.label}</div>
            </div>
          )}
        </div>
      </aside>

      {/* Main content */}
      <main>
        <div key={step} className="fade-up">{pages[step]}</div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 36 }}>
          <button onClick={() => setStep(s => s - 1)} disabled={step === 0}
            style={{ background: "transparent", border: "1px solid #22222e", borderRadius: 10, padding: "12px 24px", color: step === 0 ? "#2a2a38" : "#888", fontSize: 14, cursor: step === 0 ? "not-allowed" : "pointer" }}>← Back</button>
          {step < STEPS.length - 1 && (
            <button onClick={() => canAdvance() && setStep(s => s + 1)}
              style={{ background: canAdvance() ? "#f59e0b" : "#1a1a24", border: "none", borderRadius: 10, padding: "12px 28px", color: canAdvance() ? "#000" : "#444", fontSize: 14, fontWeight: 700, cursor: canAdvance() ? "pointer" : "not-allowed", transition: "all 0.2s", boxShadow: canAdvance() ? "0 0 24px #f59e0b44" : "none" }}>Continue →</button>
          )}
        </div>
      </main>
    </div>
  );
}
