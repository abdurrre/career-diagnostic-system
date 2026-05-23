import { BrowserRouter, Routes, Route } from "react-router-dom";

import Login from "./pages/Login";
import Register from "./pages/Register";

import UploadSection from "./components/UploadSection";
import ResultCard from "./components/ResultCard";

function App() {
  return (

    <BrowserRouter>

      <Routes>

        <Route path="/" element={<Login />} />

        <Route path="/register" element={<Register />} />

        <Route
          path="/dashboard"
          element={
            <div className="min-h-screen bg-[#f5f3ff] p-10">

              <UploadSection />

              <ResultCard />

            </div>
          }
        />

      </Routes>

    </BrowserRouter>
  );
}

export default App;