
export const Navbar = () => {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="topbar-label">WORKSPACE</span>
        <span className="topbar-divider">/</span>
        <span className="topbar-current">Generate</span>
      </div>

      <div className="topbar-right">
        <div className="environment-label">
          <span className="environment-dot"></span>
          Local environment
        </div>

        <div className="profile">
          <div className="profile-avatar">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
            >
              <circle cx="12" cy="8" r="3.5" />
              <path d="M5 20c.8-3.7 3.2-5.5 7-5.5s6.2 1.8 7 5.5" />
            </svg>
          </div>
        </div>
      </div>
    </header>
  );
};
