import { Link } from "react-router-dom";

const PLANS = [
  {
    name: "Free",
    price: "$0",
    period: "forever",
    color: "#6366f1",
    highlight: false,
    features: [
      "5-step wizard",
      "All 4 base models",
      "6 export formats",
      "Hyperparameter tooltips",
      "Runs 100% locally",
      "MIT licensed",
    ],
    cta: "Start for free",
    ctaLink: "/forge",
  },
  {
    name: "Pro",
    price: "$12",
    period: "per month",
    color: "#f59e0b",
    highlight: true,
    badge: "Coming Soon",
    features: [
      "Everything in Free",
      "Saved projects (cloud sync)",
      "Training history & versioning",
      "One-click cloud training (RunPod)",
      "Model comparison side-by-side",
      "Priority support",
    ],
    cta: "Join the waitlist",
    ctaLink: "#waitlist",
  },
  {
    name: "Team",
    price: "$49",
    period: "per month",
    color: "#10b981",
    highlight: false,
    badge: "Coming Soon",
    features: [
      "Everything in Pro",
      "Up to 5 seats",
      "Shared project library",
      "Team training queue",
      "API access",
      "Dedicated support",
    ],
    cta: "Contact us",
    ctaLink: "mailto:hello@llmforge.dev",
  },
];

const FAQ = [
  { q: "Is it really free?", a: "Yes. The wizard, all models, and all export formats are free forever. We'll add optional paid features for teams and cloud training in the future, but the core tool stays free and open source." },
  { q: "Where does my data go?", a: "Nowhere. LLM Forge runs entirely in your browser and generates scripts that run on your own machine. Your training data is never uploaded anywhere." },
  { q: "What hardware do I need?", a: "GPT-2 runs on any laptop with just CPU. TinyLlama and Phi-2 need 4–8GB VRAM. For cloud training, we'll recommend RunPod or Lambda Labs ($0.50–1.50 per training run)." },
  { q: "Can I use the generated code commercially?", a: "Yes. All generated code is yours to use however you want. LLM Forge itself is MIT licensed." },
];

export default function Pricing() {
  return (
    <div style={{ background: "#0a0a10", minHeight: "100vh" }}>
      <section style={{ maxWidth: 960, margin: "0 auto", padding: "72px 32px 0" }}>
        <div style={{ textAlign: "center", marginBottom: 56 }}>
          <div style={{ fontSize: 11, color: "#f59e0b", letterSpacing: "1.5px", textTransform: "uppercase", fontWeight: 700, marginBottom: 12 }}>Pricing</div>
          <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 44, color: "#e2e2ea", marginBottom: 14 }}>Simple, honest pricing</h1>
          <p style={{ color: "#555", fontSize: 16, maxWidth: 420, margin: "0 auto" }}>Free to use. Paid plans add cloud features — but the core tool stays open source forever.</p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, alignItems: "start" }}>
          {PLANS.map(plan => (
            <div key={plan.name} style={{
              background: plan.highlight ? "#13131a" : "#0d0d14",
              border: `1px solid ${plan.highlight ? plan.color + "55" : "#1e1e2a"}`,
              borderRadius: 16,
              padding: "28px 24px",
              position: "relative",
              boxShadow: plan.highlight ? `0 0 40px ${plan.color}22` : "none",
            }}>
              {plan.badge && (
                <div style={{ position: "absolute", top: -12, left: "50%", transform: "translateX(-50%)", background: plan.color, color: plan.highlight ? "#000" : "#fff", borderRadius: 20, padding: "3px 12px", fontSize: 10, fontWeight: 800, letterSpacing: "0.5px", whiteSpace: "nowrap" }}>
                  {plan.badge}
                </div>
              )}
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 13, color: plan.color, fontWeight: 700, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.8px" }}>{plan.name}</div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                  <span style={{ fontFamily: "'Playfair Display', serif", fontSize: 40, color: "#e2e2ea" }}>{plan.price}</span>
                  <span style={{ color: "#444", fontSize: 13 }}>/{plan.period}</span>
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 24 }}>
                {plan.features.map(f => (
                  <div key={f} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                    <span style={{ color: plan.color, fontSize: 13, marginTop: 1, flexShrink: 0 }}>✓</span>
                    <span style={{ color: "#888", fontSize: 13, lineHeight: 1.5 }}>{f}</span>
                  </div>
                ))}
              </div>
              <Link to={plan.ctaLink} style={{
                display: "block", textAlign: "center",
                background: plan.highlight ? plan.color : "transparent",
                border: `1px solid ${plan.highlight ? "transparent" : plan.color + "55"}`,
                borderRadius: 9, padding: "11px",
                color: plan.highlight ? "#000" : plan.color,
                fontSize: 13, fontWeight: 700,
              }}>
                {plan.cta}
              </Link>
            </div>
          ))}
        </div>
      </section>

      {/* FAQ */}
      <section style={{ maxWidth: 680, margin: "72px auto 0", padding: "0 32px 80px" }}>
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 32, color: "#e2e2ea" }}>Frequently asked questions</h2>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {FAQ.map(item => (
            <details key={item.q} style={{ background: "#0d0d14", border: "1px solid #1e1e2a", borderRadius: 10, overflow: "hidden" }}>
              <summary style={{ padding: "16px 20px", cursor: "pointer", fontSize: 14, color: "#e2e2ea", fontWeight: 600, listStyle: "none", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                {item.q}
                <span style={{ color: "#444", fontSize: 18, flexShrink: 0, marginLeft: 12 }}>+</span>
              </summary>
              <div style={{ padding: "0 20px 16px", color: "#666", fontSize: 13, lineHeight: 1.75 }}>{item.a}</div>
            </details>
          ))}
        </div>
      </section>
    </div>
  );
}
