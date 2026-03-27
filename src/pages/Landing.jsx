import { Link } from "react-router-dom";

const FEATURES = [
  { icon: "🖱️", title: "5-step wizard", desc: "Task → Data → Model → Training → Export. No config files, no YAML, no boilerplate." },
  { icon: "🔒", title: "100% local", desc: "Your training data never leaves your machine. No API keys, no cloud compute, no vendor lock-in." },
  { icon: "⬇️", title: "6 export formats", desc: "Python, Node.js, Dockerfile, Jupyter notebook, requirements.txt, and README — all generated for you." },
  { icon: "🧬", title: "LoRA / PEFT built in", desc: "State-of-the-art parameter-efficient fine-tuning out of the box. Train on a laptop, not a datacenter." },
  { icon: "📖", title: "Hyperparameter tooltips", desc: "Every slider explains what it does, what goes wrong if you get it wrong, and the safe default." },
  { icon: "🤖", title: "4 base models", desc: "GPT-2, GPT-2 Medium, TinyLlama 1.1B, and Phi-2. From CPU-friendly to production-grade." },
];

const STEPS = [
  { n: "01", title: "Pick your task", desc: "Chatbot, Q&A Bot, Instruction Follower, or Creative Writer. Write your system prompt." },
  { n: "02", title: "Upload your data", desc: "Drop a .jsonl file. We validate it, show a preview, and count your samples." },
  { n: "03", title: "Choose your model", desc: "Pick a base model that fits your hardware. We show VRAM requirements upfront." },
  { n: "04", title: "Set hyperparameters", desc: "Epochs, batch size, learning rate, LoRA rank. Every slider has a tooltip explaining the tradeoffs." },
  { n: "05", title: "Export and run", desc: "Download your Python script, Dockerfile, Jupyter notebook. Run it on your machine." },
];

const COMPARISON = [
  { feature: "Runs locally", forge: true, hf: "Cloud only", monster: "Cloud only" },
  { feature: "No API key needed", forge: true, hf: false, monster: false },
  { feature: "Exports runnable code", forge: true, hf: false, monster: false },
  { feature: "Free forever", forge: true, hf: "Paid tiers", monster: "Paid tiers" },
  { feature: "LoRA/PEFT support", forge: true, hf: true, monster: true },
  { feature: "Jupyter export", forge: true, hf: false, monster: false },
  { feature: "Your data stays yours", forge: true, hf: false, monster: false },
];

export default function Landing() {
  return (
    <div style={{ background: "#0a0a10" }}>

      {/* Hero */}
      <section style={{ maxWidth: 1000, margin: "0 auto", padding: "100px 32px 80px", textAlign: "center" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: "#13131a", border: "1px solid #f59e0b22", borderRadius: 20, padding: "5px 14px", marginBottom: 28 }}>
          <span style={{ fontSize: 12 }}>🔬</span>
          <span style={{ fontSize: 12, color: "#f59e0b", fontWeight: 600 }}>Open-source fine-tuning wizard</span>
        </div>
        <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: "clamp(38px, 6vw, 68px)", lineHeight: 1.08, letterSpacing: "-2px", color: "#e2e2ea", marginBottom: 24 }}>
          Fine-tune your own LLM.<br />
          <span style={{ background: "linear-gradient(90deg, #f59e0b, #ea580c)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            No cloud. No code. No limits.
          </span>
        </h1>
        <p style={{ fontSize: 18, color: "#555", maxWidth: 500, margin: "0 auto 40px", lineHeight: 1.7 }}>
          A 5-step wizard that generates a complete fine-tuning pipeline — Python, Docker, Jupyter — that runs entirely on your own machine.
        </p>
        <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
          <Link to="/forge" style={{
            background: "#f59e0b", color: "#000", borderRadius: 10,
            padding: "14px 28px", fontSize: 15, fontWeight: 700,
            boxShadow: "0 4px 28px #f59e0b44", display: "inline-block",
          }}>
            Launch the wizard →
          </Link>
          <a href="https://github.com" target="_blank" rel="noreferrer" style={{
            background: "transparent", color: "#888", border: "1px solid #2a2a38",
            borderRadius: 10, padding: "14px 28px", fontSize: 15, display: "inline-block",
          }}>
            View on GitHub
          </a>
        </div>
        <p style={{ color: "#333", fontSize: 12, marginTop: 16 }}>Free forever · MIT license · No signup required</p>
      </section>

      {/* How it works */}
      <section style={{ background: "#080810", borderTop: "1px solid #16161f", borderBottom: "1px solid #16161f", padding: "72px 32px" }}>
        <div style={{ maxWidth: 900, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 52 }}>
            <div style={{ fontSize: 11, color: "#f59e0b", letterSpacing: "1.5px", textTransform: "uppercase", fontWeight: 700, marginBottom: 12 }}>How it works</div>
            <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 36, color: "#e2e2ea" }}>From zero to fine-tuned in 5 steps</h2>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {STEPS.map((s, i) => (
              <div key={s.n} style={{ display: "flex", gap: 24, alignItems: "flex-start", paddingBottom: i < STEPS.length - 1 ? 36 : 0, marginBottom: i < STEPS.length - 1 ? 0 : 0, position: "relative" }}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0, gap: 0 }}>
                  <div style={{ width: 40, height: 40, borderRadius: "50%", background: "#f59e0b18", border: "1px solid #f59e0b44", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'DM Mono', monospace", fontSize: 12, color: "#f59e0b", fontWeight: 700 }}>{s.n}</div>
                  {i < STEPS.length - 1 && <div style={{ width: 2, flex: 1, minHeight: 28, background: "#1e1e2a", marginTop: 6 }} />}
                </div>
                <div style={{ paddingTop: 8, paddingBottom: i < STEPS.length - 1 ? 36 : 0 }}>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "#e2e2ea", marginBottom: 6 }}>{s.title}</div>
                  <div style={{ fontSize: 14, color: "#555", lineHeight: 1.7 }}>{s.desc}</div>
                </div>
              </div>
            ))}
          </div>
          <div style={{ textAlign: "center", marginTop: 48 }}>
            <Link to="/forge" style={{ background: "#f59e0b", color: "#000", borderRadius: 10, padding: "13px 28px", fontSize: 14, fontWeight: 700, display: "inline-block", boxShadow: "0 0 24px #f59e0b33" }}>
              Try it now →
            </Link>
          </div>
        </div>
      </section>

      {/* Features grid */}
      <section style={{ maxWidth: 960, margin: "0 auto", padding: "72px 32px" }}>
        <div style={{ textAlign: "center", marginBottom: 48 }}>
          <div style={{ fontSize: 11, color: "#f59e0b", letterSpacing: "1.5px", textTransform: "uppercase", fontWeight: 700, marginBottom: 12 }}>Features</div>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 36, color: "#e2e2ea" }}>Everything you need. Nothing you don't.</h2>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
          {FEATURES.map(f => (
            <div key={f.title} style={{ background: "#0d0d14", border: "1px solid #1e1e2a", borderRadius: 14, padding: "24px 20px" }}>
              <span style={{ fontSize: 28, display: "block", marginBottom: 12 }}>{f.icon}</span>
              <div style={{ fontSize: 15, fontWeight: 600, color: "#e2e2ea", marginBottom: 6 }}>{f.title}</div>
              <div style={{ fontSize: 13, color: "#555", lineHeight: 1.7 }}>{f.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Comparison */}
      <section style={{ background: "#080810", borderTop: "1px solid #16161f", borderBottom: "1px solid #16161f", padding: "72px 32px" }}>
        <div style={{ maxWidth: 760, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 40 }}>
            <div style={{ fontSize: 11, color: "#f59e0b", letterSpacing: "1.5px", textTransform: "uppercase", fontWeight: 700, marginBottom: 12 }}>Why LLM Forge</div>
            <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 34, color: "#e2e2ea" }}>How it stacks up</h2>
          </div>
          <div style={{ background: "#0d0d14", border: "1px solid #1e1e2a", borderRadius: 14, overflow: "hidden" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", padding: "14px 20px", borderBottom: "1px solid #1e1e2a", background: "#13131a" }}>
              {["Feature", "LLM Forge", "HuggingFace AutoTrain", "MonsterAPI"].map((h, i) => (
                <div key={h} style={{ fontSize: 11, fontWeight: 700, color: i === 1 ? "#f59e0b" : "#444", textTransform: "uppercase", letterSpacing: "0.8px" }}>{h}</div>
              ))}
            </div>
            {COMPARISON.map((row, i) => (
              <div key={row.feature} style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", padding: "13px 20px", borderBottom: i < COMPARISON.length - 1 ? "1px solid #16161f" : "none" }}>
                <div style={{ fontSize: 13, color: "#888" }}>{row.feature}</div>
                {[row.forge, row.hf, row.monster].map((val, j) => (
                  <div key={j} style={{ fontSize: 13 }}>
                    {val === true ? <span style={{ color: "#10b981", fontWeight: 700 }}>✓</span>
                     : val === false ? <span style={{ color: "#ef4444" }}>✗</span>
                     : <span style={{ color: "#555" }}>{val}</span>}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ maxWidth: 700, margin: "0 auto", padding: "80px 32px", textAlign: "center" }}>
        <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 40, color: "#e2e2ea", marginBottom: 16 }}>Ready to forge your model?</h2>
        <p style={{ color: "#555", fontSize: 16, marginBottom: 36 }}>Free. Local. Yours.</p>
        <Link to="/forge" style={{ background: "#f59e0b", color: "#000", borderRadius: 10, padding: "15px 36px", fontSize: 15, fontWeight: 800, display: "inline-block", boxShadow: "0 0 40px #f59e0b44" }}>
          Launch LLM Forge →
        </Link>
        <div style={{ display: "flex", gap: 24, justifyContent: "center", marginTop: 20 }}>
          <Link to="/pricing" style={{ color: "#444", fontSize: 13 }}>View Pricing</Link>
          <Link to="/docs" style={{ color: "#444", fontSize: 13 }}>Read the Docs</Link>
        </div>
      </section>

      <footer style={{ borderTop: "1px solid #16161f", padding: "22px 48px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>🔬</span>
          <span style={{ fontFamily: "'Playfair Display', serif", color: "#e2e2ea", fontSize: 15 }}>LLM Forge</span>
        </div>
        <div style={{ display: "flex", gap: 24 }}>
          <Link to="/pricing" style={{ color: "#333", fontSize: 13 }}>Pricing</Link>
          <Link to="/docs" style={{ color: "#333", fontSize: 13 }}>Docs</Link>
          <a href="https://github.com" style={{ color: "#333", fontSize: 13 }}>GitHub</a>
        </div>
      </footer>
    </div>
  );
}
