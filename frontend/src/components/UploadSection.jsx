import { useState } from "react";
import api from "../services/api";

function UploadSection({ onAnalyze }) {

  const [file, setFile] = useState(null);
  const [profession, setProfession] = useState("");
  const [context, setContext] = useState("");

  const handleAnalyze = async () => {
    try {
      const response = await api.get("/");
      console.log(response.data);
      alert("API Connected!");
    } catch (error) {
      console.log(error);
      alert("API Failed!");
    }

    const result = {
      score: 78,
      profession,
      missingSkills: [
        "Docker",
        "TensorFlow",
        "REST API"
      ],
      recommendations: [
        "Learn Docker Basics",
        "Practice REST API",
        "Study TensorFlow"
      ]
    };

    onAnalyze(result);
  };

  return (
    <section className="max-w-4xl mx-auto bg-white p-8 rounded-2xl shadow">

      <div className="grid md:grid-cols-2 gap-8">

        <div className="border-2 border-dashed rounded-xl p-8 text-center">

          <p className="font-semibold">
            Upload Your CV
          </p>

          <input
            type="file"
            accept=".pdf"
            className="mt-4"
            onChange={(e) => setFile(e.target.files[0])}
          />

        </div>

        <div className="space-y-4">

          <select
            className="w-full border p-3 rounded-lg"
            value={profession}
            onChange={(e) => setProfession(e.target.value)}
          >
            <option value="">
              Select Profession
            </option>

            <option>Frontend Developer</option>
            <option>Backend Developer</option>
            <option>Data Scientist</option>

          </select>

          <textarea
            rows="5"
            className="w-full border p-3 rounded-lg"
            placeholder="Additional context..."
            value={context}
            onChange={(e) => setContext(e.target.value)}
          />

          <button
            onClick={handleAnalyze}
            className="bg-purple-700 text-white px-6 py-3 rounded-lg w-full"
          >
            Analyze My CV
          </button>

        </div>

      </div>

    </section>
  );
}

export default UploadSection;