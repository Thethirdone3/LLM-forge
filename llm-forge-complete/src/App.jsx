import { Routes, Route } from "react-router-dom";
import Nav from "./components/Nav";
import Landing from "./pages/Landing";
import Forge from "./pages/Forge";
import Pricing from "./pages/Pricing";
import Docs from "./pages/Docs";

export default function App() {
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0a0a10; font-family: 'DM Sans', sans-serif; color: #e2e2ea; min-height: 100vh; }
        input[type=range] { -webkit-appearance: none; height: 4px; background: #2a2a38; border-radius: 2px; display: block; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #f59e0b; cursor: pointer; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: #22222e; border-radius: 2px; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
        .fade-up { animation: fadeUp 0.35s ease forwards; }
        a { text-decoration: none; color: inherit; }
      `}</style>
      <Nav />
      <Routes>
        <Route path="/"        element={<Landing />} />
        <Route path="/forge"   element={<Forge />} />
        <Route path="/pricing" element={<Pricing />} />
        <Route path="/docs"    element={<Docs />} />
      </Routes>
    </>
  );
}
