import { Navbar } from "./components/Navbar";
import { Sidebar } from "./components/SideBar";
import { InputPanel } from "./components/InputPanel";

function App() {
  return (
    <div className="min-h-screen bg-[#071514] p-4">
      <Navbar />

      <div className="flex gap-4 mt-4">
        <Sidebar />
        <InputPanel />
      </div>
    </div>
  );
}

export default App;