function ResultCard({ result }) {

  if (!result) return null;

  return (
    <div className="bg-white p-10 mt-10 rounded-xl shadow">

      <h1 className="text-3xl font-bold text-purple-700">
        Analysis Result
      </h1>

      <p className="mt-6 text-xl">
        Match Score:
        <span className="font-bold">
          {" "} {result.score}%
        </span>
      </p>

      <div className="mt-8">

        <h2 className="text-2xl font-semibold mb-4">
          Missing Skills
        </h2>

        <div className="flex flex-wrap gap-3">

          {result.missingSkills.map((skill, index) => (
            <span
              key={index}
              className="bg-red-100 text-red-700 px-4 py-2 rounded-full"
            >
              {skill}
            </span>
          ))}

        </div>

      </div>

      <div className="mt-10">

        <h2 className="text-2xl font-semibold mb-4">
          Recommendations
        </h2>

        <ul className="space-y-2">

          {result.recommendations.map((item, index) => (
            <li key={index}>
              📚 {item}
            </li>
          ))}

        </ul>

      </div>

    </div>
  );
}

export default ResultCard;