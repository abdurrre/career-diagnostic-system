function Navbar() {
  return (
    <nav className="flex justify-between items-center px-8 py-4 border-b">
      <h1 className="text-2xl font-bold text-purple-700">
        SkillPath AI
      </h1>

      <ul className="flex gap-6 text-sm">
        <li>Home</li>
        <li>History</li>
        <li>About</li>
      </ul>

      <button className="bg-purple-700 text-white px-4 py-2 rounded-lg">
        Login
      </button>
    </nav>
  );
}

export default Navbar;