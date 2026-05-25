/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f4f6ff',
          100: '#e9ecff',
          200: '#d7dbff',
          300: '#b9c1ff',
          400: '#949eff',
          500: '#6d75ff',
          600: '#4f4bf5',
          700: '#3e37df',
          800: '#322cb7',
          900: '#2c2992',
          950: '#1a1854',
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        outfit: ['Outfit', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
