
import logo from "../assets/LogoWatermarkX.png";

export const Sidebar = () => {
  const items = [
    { name: "Dashboard", icon: "▦" },
    { name: "Generate", icon: "✳" },
    { name: "Attacks", icon: "⌁" },
    { name: "Reports", icon: "▤" },
  ];

  return (
    <aside className="sidebar">

      <div className="sidebar-brand">
        <img src={logo} alt="WatermarkX logo" />

        <div>
          <h1>WatermarkX</h1>
          <p>LLM WATERMARK ROBUSTNESS</p>
        </div>
      </div>

      <div className="sidebar-menu">
        <p className="menu-label">WORKSPACE</p>

        <nav>
          {items.map((item) => (
            <button
              key={item.name}
              className={`sidebar-link ${
                item.name === "Generate" ? "selected" : ""
              }`}
              type="button"
            >
              <span className="sidebar-icon">{item.icon}</span>
              <span>{item.name}</span>
            </button>
          ))}
        </nav>
      </div>

      <div className="sidebar-footer">
        <div className="footer-line"></div>

        <span className="footer-label">CURRENT PROJECT</span>
        <p>WatermarkX</p>
        <span className="footer-description">
          LLM Watermark Analysis
        </span>

        <div className="version-label">
          <span>VERSION</span>
          <span>1.0.0</span>
        </div>
      </div>

    </aside>
  );
};
