
import { Navbar } from "./components/Navbar";
import { Sidebar } from "./components/SideBar";
import { InputPanel } from "./components/InputPanel";
import "./app.css";

function App() {
  return (
    <div className="app-shell">
      <Sidebar />

      <div className="app-main">
        <Navbar />

        <main className="workspace">
          <InputPanel />
        </main>
      </div>
    </div>
  );
}

export default App;
