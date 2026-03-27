import { Link, useLocation } from "react-router-dom";

const LINKS = [
  { to: "/",        label: "Home" },
  { to: "/forge",   label: "Launch Forge" },
  { to: "/pricing", label: "Pricing" },
  { to: "/docs",    label: "Docs" },
];

export default function Nav() {
  const { pathname } = useLocation();

  return (
    <header style={{
      borderBottom: "1px solid #16161f",
      padding: "0 48px",
      height: 58,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      position: "sticky",
      top: 0,
      zIndex: 200,
      background: "#0a0a10",
    }}>
      {/* Logo */}
      <Link to="/" style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 20 }}>🔬</span>
        <span style={{ fontFamily: "'Playfair Display', serif", fontSize: 20, color: "#e2e2ea" }}>LLM Forge</span>
        <span style={{
          background: "#f59e0b18", color: "#f59e0b", border: "1px solid #f59e0b33",
          borderRadius: 20, padding: "2px 8px", fontSize: 10, fontWeight: 700,
          letterSpacing: "0.5px", marginLeft: 4,
        }}>BETA</span>
      </Link>

      {/* Nav links */}
      <nav style={{ display: "flex", alignItems: "center", gap: 4 }}>
        {LINKS.map(link => {
          const active = pathname === link.to;
          return (
            <Link key={link.to} to={link.to} style={{
              padding: "7px 14px",
              borderRadius: 8,
              fontSize: 13,
              fontWeight: active ? 600 : 400,
              color: active ? "#f59e0b" : "#555",
              background: active ? "#f59e0b0d" : "transparent",
              transition: "all 0.15s",
            }}>
              {link.label}
            </Link>
          );
        })}
      </nav>

      {/* CTA */}
      <Link to="/forge" style={{
        background: "#f59e0b",
        border: "none",
        borderRadius: 9,
        padding: "9px 20px",
        color: "#000",
        fontSize: 13,
        fontWeight: 700,
        cursor: "pointer",
        boxShadow: "0 0 20px #f59e0b33",
      }}>
        Start Building →
      </Link>
    </header>
  );
}
