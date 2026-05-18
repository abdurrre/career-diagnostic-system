import { useState } from "react";
import UploadSection from "./components/UploadSection";
import ResultCard from "./components/ResultCard";

function App() {

  const [result, setResult] = useState(null);

  return (
    <div className="min-h-screen bg-[#f5f3ff] p-10">

      <UploadSection onAnalyze={setResult} />

      <ResultCard result={result} />

    </div>
  );
}

export default App;