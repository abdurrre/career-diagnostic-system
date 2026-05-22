function Login() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#f5f3ff]">

      <div className="bg-white p-8 rounded-2xl shadow w-full max-w-md">

        <h1 className="text-3xl font-bold text-center mb-6">
          Login
          <p className="text-center mt-4">
  Don't have an account?

  <a
    href="/register"
    className="text-purple-700 font-semibold ml-1"
  >
    Register
  </a>
</p>
        </h1>

        <input
          type="email"
          placeholder="Email"
          className="w-full border p-3 rounded-lg mb-4"
        />

        <input
          type="password"
          placeholder="Password"
          className="w-full border p-3 rounded-lg mb-6"
        />

        <a
  href="/dashboard"
  className="block w-full bg-purple-700 text-white py-3 rounded-lg text-center"
>
  Login
</a>

      </div>

    </div>
  );
}

export default Login;