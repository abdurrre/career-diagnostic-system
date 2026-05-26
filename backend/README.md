# 🔌 Backend - Career Diagnostic API Gateway

Server backend utama yang bertindak sebagai gerbang API (*API Gateway*), penyedia lapisan persistensi database relasional, serta proxy penghubung yang menjembatani frontend dashboard dengan layanan AI Engine.

---

## 🔒 Fitur Utama & Keamanan

1. **Autentikasi Pengguna & Token JWT:**
   * Registrasi akun dan login aman menggunakan token enkripsi JWT (*JSON Web Token*) yang disimpan secara lokal di frontend.
2. **Validasi Pendaftaran Duplikat:**
   * Dilengkapi pemeriksaan integritas database sebelum mendaftarkan akun baru. Jika alamat email yang sama sudah terdaftar, server akan menolak pendaftaran dengan mengirimkan status `400 Bad Request` beserta pesan kesalahan terjemahan Indonesia yang ramah untuk ditampilkan di antarmuka.
3. **Persistensi Riwayat yang Fleksibel (`getUserHistories`):**
   * Endpoint `/analysis/history` memuat seluruh log riwayat pencarian pengguna dengan memetakan asosiasi relasi Sequelize kompleks (`User`, `History`, `Profession`, `Skill`, `HistorySkill`).
4. **Pembuatan Otomatis Keahlian Baru (`saveHistory`):**
   * Menyelesaikan masalah hilangnya kesenjangan keterampilan (*gaps*) kustom yang dianalisis oleh AI. Jika hasil analisis AI menghasilkan nama keahlian baru yang belum ada di tabel database, server secara otomatis akan membuat record keahlian baru tersebut di tabel `skills` menggunakan logika pencarian/pembuatan dinamis sebelum menautkannya ke riwayat pencarian.
5. **Proxy Chatbot AI Engine (`chatWithAI`):**
   * Membuka rute aman `/analysis/chat` untuk menghubungkan *chat widget* frontend ke FastAPI AI Engine, meneruskan konteks CV, skor kesesuaian, dan menindaklanjuti batas aman permintaan (*rate-limiting errors*) secara transparan.

---

## 📦 Teknologi Inti

* **Runtime:** Node.js (v18+)
* **Framework:** Express.js (untuk routing dan middleware)
* **ORM Database:** Sequelize (pemetaan objek relasional untuk database)
* **Database Driver:** SQLite (pengembangan lokal) / PostgreSQL (siap produksi)
* **HTTP Client:** Axios (untuk komunikasi internal antarservis ke AI Engine)
* **Utilitas:** bcryptjs (enkripsi kata sandi) & jsonwebtoken (autentikasi token)

---

## ⚙️ Variabel Lingkungan (.env)

Buat berkas `.env` di direktori `backend/` dengan parameter berikut:
```env
PORT=5000
JWT_SECRET=rahasia_sangat_kuat_2026
AI_SCAN_SERVICE_URL=http://localhost:8000/api/scan
AI_CHAT_SERVICE_URL=http://localhost:8000/api/chat
```

---

## 🏃 Cara Menjalankan Server

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
4. Server akan aktif pada alamat `http://localhost:5000` dan secara otomatis melakukan sinkronisasi tabel database.