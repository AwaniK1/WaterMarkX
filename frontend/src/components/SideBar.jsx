export const Sidebar = () => {
    return (
        <div className="flex flex-col m-3 w-60 p-4 rounded-3xl border-2 border-cyan-500/60 bg-black gap-4 shadow-[0_0_20px_rgba(6,182,212,0.15)]">

            <button className="cursor-pointer text-slate-100 text-left px-3 py-2 rounded-xl border-2 border-transparent hover:border-cyan-500/70 hover:shadow-[0_0_15px_rgba(6,182,212,0.6)] transition">
                Dashboard
            </button>

            <button className="cursor-pointer text-slate-100 text-left px-3 py-2 rounded-xl border-2 border-transparent hover:border-cyan-500/70 hover:shadow-[0_0_15px_rgba(6,182,212,0.6)] transition">
                Generate
            </button>

            <button className="cursor-pointer text-slate-100 text-left px-3 py-2 rounded-xl border-2 border-transparent hover:border-cyan-500/70 hover:shadow-[0_0_15px_rgba(6,182,212,0.6)] transition">
                Attacks
            </button>

            <button className="cursor-pointer text-slate-100 text-left px-3 py-2 rounded-xl border-2 border-transparent hover:border-cyan-500/70 hover:shadow-[0_0_15px_rgba(6,182,212,0.6)] transition">
                Reports
            </button>

        </div>
    );
};