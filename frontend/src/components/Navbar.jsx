import logo from "../assets/LogoWatermarkX.png";

export const Navbar = () => {
    return (
        <nav className="flex m-3 items-center justify-between rounded-2xl border-2 border-cyan-500/60 bg-black px-6 py-4 shadow-[0_0_20px_rgba(6,182,212,0.15)]">

            <div className="flex items-center gap-3">
                <img src={logo} alt="WatermarkX Logo" className="h-10 w-20" />
                <h1 className="text-2xl font-semibold text-slate-100">
                    WatermarkX
                </h1>
            </div>

            <div className="flex flex-col items-center">
                <p className="text-xs tracking-wider text-cyan-300">
                    USER
                </p>

                <div className="mt-1 flex h-10 w-10 items-center justify-center rounded-full border-2 border-cyan-500/60 bg-black">
                    <svg
                        viewBox="0 0 24 24"
                        className="h-6 w-6 text-slate-100"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                    >
                        <circle cx="12" cy="8" r="4" />
                        <path d="M4 20c2-4 14-4 16 0" />
                    </svg>
                </div>
            </div>

        </nav>
    );
};