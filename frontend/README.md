# 💻 Frontend - SkillPath AI Dashboard

Dashboard web interaktif untuk menganalisis keselarasan karier dan memetakan kesenjangan keahlian (*skill gaps*) pengguna secara real-time. Didesain dengan estetika modern kelas premium (*premium aesthetics*) dan antarmuka Bahasa Indonesia yang ramah dan luwes.

---

## 🎨 Desain & Estetika Premium

Aplikasi klien ini dirancang dengan prinsip desain modern terbaik untuk memukau pengguna saat pertama kali membukanya:
* **Glassmorphic Cards:** Panel transparan dengan efek blur latar belakang (*backdrop blur*) yang halus.
* **Non-blocking Cognitive Loading:** Animasi ikon siluet kepala manusia tampak samping yang artistik dengan roda gigi tunggal yang berputar stabil pada poros sumbunya saat data sedang dianalisis oleh AI.
* **Glow & Pulsing FAB:** Tombol obrolan asisten AI chatbot melayang di pojok kanan bawah dengan efek berdenyut (*pulse*) dan bayangan lembut (*soft glowing shadow*).
* **Responsive Layout:** Grid layout modern Tailwind CSS v4 yang sepenuhnya responsif di perangkat seluler hingga layar desktop super lebar.
* **Branding Terintegrasi:** Integrasi favicon kustom (.svg & .ico) di seluruh halaman UI (Navbar, Footer, Login/Register, Reset Sandi, 404) untuk menyajikan tampilan yang solid dan profesional.

---

## ⚙️ Fitur Utama

1. **Upload CV Seret & Lepas (Drag & Drop):** Area pengunggahan berkas yang aman, membatasi format hanya untuk PDF dan membatasi ukuran file maksimal 5MB secara dinamis sebelum dikirim ke server.
2. **Riwayat Diagnostik Terintegrasi (`HistoryView`):** Halaman riwayat lengkap yang memuat log riwayat langsung dari database. Menekan tombol *"Lihat Hasil"* akan memuat seluruh daftar keahlian dan kesenjangan kompetensi asli dari database tanpa ada data kustom yang hilang.
3. **Pemuatan Laporan Terperinci (`ReportView`):** Laporan interaktif dengan ringkasan skor berupa diagram lingkaran SVG dinamis, pesan umpan balik cerdas, dan kartu kesenjangan keterampilan yang diwarnai berdasarkan prioritas (*tier*).
4. **Disclaimer Transparansi AI:** Papan spanduk formal (disclaimer) di dalam laporan hasil diagnosis yang memperjelas bahwa hasil analisis berbasis kecerdasan buatan bersifat estimasi pendukung keputusan dan tidak dijamin 100% akurat.
5. **Indikator Simpan Hasil Berputar (Loading Spinner):** Indikator loading berupa putaran lingkaran (`Loader2` beranimasi `animate-spin`) pada tombol "Simpan Hasil" saat proses penyimpanan log riwayat ke database sedang diproses.
6. **Autentikasi Aman & Visibilitas Kata Sandi:**
   * Kolom input kata sandi dilengkapi tombol ikon mata (*toggle visibility*) di layar pendaftaran, masuk, dan reset kata sandi baru.
   * Alur lupa kata sandi riil terintegrasi penuh lewat email SMTP Nodemailer. Komponen utama otomatis memindai parameter URL pada mount untuk memicu pengalihan alur reset kata sandi baru secara mulus.
7. **Widget AI Chatbot Cerdas (`ChatWidget`):** Kotak obrolan melayang di pojok kanan bawah yang otomatis menyinkronkan data kualifikasi CV Anda sebagai konteks saat mengobrol dengan asisten AI.

---

## 🚀 Teknologi yang Digunakan

* **Core:** React 19 (terbaru untuk performa optimal), HTML5, Vanilla JavaScript.
* **Styling:** Tailwind CSS v4 (menggunakan engine pengolah terbaru) & CSS kustom.
* **Animations:** Framer Motion v12 (untuk transisi halaman dan efek spring).
* **Icons:** Lucide React (ikon berkualitas tinggi yang minimalis).
* **Bundler:** Vite v8 (kecepatan kompilasi dan hot reload instan).

---

## ⚙️ Konfigurasi API

Aplikasi ini menggunakan konfigurasi API yang fleksibel di berkas `src/config/api.js`. Alamat URL base API diatur secara dinamis untuk mendeteksi apakah aplikasi dijalankan pada server lokal (*development*) atau server produksi (*production*):
```javascript
export const API_BASE_URL =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? "http://localhost:5000/api" // Endpoint lokal
    : "https://career-diagnostic-system-backend.vercel.app/api"; // Endpoint Vercel Live
```

---

## 🏃 Cara Menjalankan Aplikasi secara Lokal

1. Pastikan Anda berada di direktori `frontend/`:
   ```bash
   cd frontend
   ```
2. Instal semua dependensi Node.js:
   ```bash
   npm install
   ```
3. Jalankan server pengembangan Vite:
   ```bash
   npm run dev
   ```
4. Buka peramban (*browser*) Anda ke alamat `http://localhost:5173`.

---

## 🚀 Deployment (Vercel)

Aplikasi frontend ini siap di-deploy secara instan ke **Vercel**.

1. Masuk ke direktori `frontend/`:
   ```bash
   cd frontend
   ```
2. Jalankan Vercel CLI untuk pertama kali untuk menghubungkan project:
   ```bash
   vercel
   ```
3. Terapkan rilis production yang optimal:
   ```bash
   vercel --prod
   ```
4. Aplikasi akan live secara online (contoh: `https://skillpath-ai-delta.vercel.app`).
