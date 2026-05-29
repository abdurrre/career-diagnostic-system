# 🔌 Backend - Career Diagnostic API Gateway

Server backend utama yang bertindak sebagai gerbang API (*API Gateway*), penyedia lapisan persistensi database relasional, serta proxy penghubung yang menjembatani frontend dashboard dengan layanan AI Engine.

---

## 🔒 Fitur Utama & Keamanan

1. **Autentikasi Pengguna & Token JWT:**
   * Registrasi akun dan login aman menggunakan token enkripsi JWT (*JSON Web Token*) yang disimpan secara lokal di frontend.
2. **Mitigasi Serangan User Enumeration (Keamanan Akun):**
   * Menyeragamkan respons galat kegagalan masuk/login (`"Email atau password salah"`) dan status HTTP `401 Unauthorized` baik ketika email tidak ditemukan di database maupun ketika kata sandi salah, demi mencegah pelacakan dan pengumpulan daftar akun aktif oleh penyerang.
3. **Lupa & Reset Kata Sandi Riil (Nodemailer SMTP):**
   * Mengintegrasikan alur lupa kata sandi riil dengan membuat token stateless JWT dinamis yang ditandatangani menggunakan `JWT_SECRET` + `user.password` (hash sandi saat ini). Token otomatis kedaluwarsa setelah dipakai satu kali.
   * Menggunakan **Nodemailer** untuk mengirimkan email HTML formal langsung ke kotak masuk pengguna yang berisi tombol tautan pemulihan.
4. **Persistensi Riwayat yang Fleksibel (`getUserHistories`):**
   * Endpoint `/analysis/history` memuat seluruh log riwayat pencarian pengguna dengan memetakan asosiasi relasi Sequelize kompleks (`User`, `History`, `Profession`, `Skill`, `HistorySkill`).
5. **Pembuatan Otomatis Keahlian Baru (`saveHistory`):**
   * Menyelesaikan masalah hilangnya kesenjangan keterampilan (*gaps*) kustom yang dianalisis oleh AI. Jika hasil analisis AI menghasilkan nama keahlian baru yang belum ada di tabel database, server secara otomatis akan membuat record keahlian baru tersebut di tabel `skills` menggunakan logika pencarian/pembuatan dinamis sebelum menautkannya ke riwayat pencarian.
6. **Proxy Chatbot AI Engine (`chatWithAI`):**
   * Membuka rute aman `/analysis/chat` untuk menghubungkan *chat widget* frontend ke FastAPI AI Engine, meneruskan konteks CV, skor kesesuaian, dan menindaklanjuti batas aman permintaan (*rate-limiting errors*) secara transparan.

---

## 📦 Teknologi Inti

* **Runtime:** Node.js (v18+)
* **Framework:** Express.js (untuk routing dan middleware)
* **ORM Database:** Sequelize v6 (pemetaan objek relasional untuk database)
* **Database Driver:** MySQL (menggunakan `mysql2` dengan enkripsi SSL untuk kesiapan cloud)
* **Email Service:** Nodemailer SMTP Email (mendukung integrasi email riil via TLS/STARTTLS)
* **HTTP Client:** Axios (untuk komunikasi internal antarservis ke AI Engine)
* **Utilitas:** bcryptjs (enkripsi kata sandi) & jsonwebtoken (autentikasi token), pdf-parse (untuk ekstraksi teks PDF dengan kompatibilitas serverless)

---

## ⚙️ Variabel Lingkungan (.env)

Buat berkas `.env` di direktori `backend/` dengan parameter lengkap berikut:
```env
PORT=5000
JWT_SECRET=rahasia_sangat_kuat_2026

# Konfigurasi Database (MySQL)
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASS=
DB_NAME=capstone_project
DB_DIALECT=mysql

# Layanan AI Engine (Hugging Face / FastAPI)
AI_SCAN_SERVICE_URL=https://rizsd21-career-diagnostic-ai-engine.hf.space/api/diagnose
AI_CHAT_SERVICE_URL=https://rizsd21-career-diagnostic-ai-engine.hf.space/api/chat
GROQ_API_KEY=your_groq_api_key_here
AI_ENGINE_API_KEY=your_secure_ai_engine_secret_here

# Konfigurasi SMTP untuk Pengiriman Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USER=email_pengirim@gmail.com
SMTP_PASS=kata_sandi_aplikasi_google_16_karakter
FRONTEND_URL=http://localhost:5173
```

---

## 🏃 Cara Menjalankan Server secara Lokal

1. Masuk ke direktori `backend/`:
   ```bash
   cd backend
   ```
2. Instal semua pustaka dependensi:
   ```bash
   npm install
   ```
3. Jalankan server dalam mode pengembangan:
   ```bash
   npm run dev
   ```
4. Server akan aktif pada alamat `http://localhost:5000` dan secara otomatis melakukan sinkronisasi database lokal.

---

## 🚀 Deployment (Serverless Vercel)

Backend ini dirancang dengan struktur serverless sehingga dapat di-deploy ke **Vercel** dengan sangat efisien.

1. **Persiapan:**
   * Pastikan berkas `vercel.json` dan ekspor `module.exports = app` di `server.js` tetap terjaga.
   * Pastikan database cloud (seperti Aiven atau TiDB Cloud MySQL) telah dibuat dan mendukung koneksi aman SSL.
2. **Langkah Deploy:**
   * Jalankan `vercel` di dalam folder `backend/` untuk menghubungkan project.
   * Masukkan seluruh Environment Variables di atas ke halaman Settings Vercel Dashboard.
   * Terapkan rilis production dengan menjalankan:
     ```bash
     vercel --prod
     ```