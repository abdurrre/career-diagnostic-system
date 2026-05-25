# Career Diagnostic System — Backend

REST API untuk sistem analisis CV berbasis profesi, dibangun dengan **Node.js**, **Express**, dan **Sequelize** (MySQL).

---

## Prasyarat

Pastikan sudah terinstall di mesinmu:

- [Node.js](https://nodejs.org/) >= 18
- MySQL (aktif dan berjalan)

---

## Setup & Instalasi

### 1. Clone repository

```bash
git clone <url-repository>
cd career-diagnostic-system/backend
```

### 2. Install dependencies

```bash
npm install
```

### 3. Konfigurasi environment

Buat file `.env` di folder `backend/` berdasarkan contoh berikut:

```env
PORT=5000
DB_HOST=localhost
DB_USER=root
DB_PASS=password_kamu
DB_NAME=career_diagnostic
DB_DIALECT=mysql

JWT_SECRET=rahasia

AI_SERVICE_URL=http://localhost:8000/api/ai/analyze
```

### 4. Buat database

Buat database MySQL secara manual (nama harus sama dengan `DB_NAME` di `.env`):

```sql
CREATE DATABASE career_diagnostic;
```

### 5. Jalankan server (auto-sync tabel)

Server menggunakan `sequelize.sync({ alter: true })`, sehingga tabel akan otomatis dibuat saat server pertama kali dijalankan:

```bash
npm run dev
```

Tunggu hingga muncul:

```
Database terhubung dan tabel berhasil disinkronkan
Server berjalan pada http://localhost:5000
```

Lalu **hentikan server** (`Ctrl + C`) sebelum menjalankan seeder.

### 6. Jalankan seeder

```bash
npx sequelize-cli db:seed:all
```

Seeder akan mengisi tabel `professions`, `skills`, dan `profession_skills` dengan data awal. Aman dijalankan berulang kali (skip jika data sudah ada).

### 7. Jalankan server kembali

```bash
npm run dev
```

---

## Struktur Folder

```
backend/
├── config/
│   ├── database.js          # Koneksi Sequelize untuk aplikasi
│   └── sequelize-cli.js     # Konfigurasi khusus Sequelize CLI
├── controllers/             # Logic handler tiap endpoint
├── middleware/              # Auth middleware (JWT)
├── models/                  # Definisi model Sequelize
├── routes/                  # Definisi routing Express
├── scripts/                 # Utility & generator scripts
│   └── generate_seeder.js   # Script untuk generate seeder dari CSV
├── seeders/                 # Data awal (professions, skills, profession_skills)
├── .env                     # Environment variables (buat sendiri)
├── .sequelizerc             # Konfigurasi path Sequelize CLI
├── package.json
└── server.js                # Entry point aplikasi
```

---

## API Endpoints

### Auth
| Method | Endpoint | Keterangan |
|--------|----------|------------|
| POST | `/api/auth/register` | Daftar akun baru |
| POST | `/api/auth/login` | Login, mendapat token JWT |

### Professions
| Method | Endpoint | Keterangan |
|--------|----------|------------|
| GET | `/api/professions` | Ambil semua profesi |
| POST | `/api/professions` | Tambah profesi baru |
| GET | `/api/professions/:id/skills` | Ambil skill dari profesi tertentu |

### Analysis *(butuh login untuk beberapa endpoint)*
| Method | Endpoint | Auth | Keterangan |
|--------|----------|------|------------|
| POST | `/api/analysis/scan` | ❌ | Scan CV (upload PDF) |
| POST | `/api/analysis/save` | ✅ | Simpan hasil analisis ke history |
| GET | `/api/analysis/history` | ✅ | Ambil riwayat analisis user |

> Untuk endpoint yang butuh auth, sertakan header: `Authorization: Bearer <token>`

---

## Integrasi Data & Regenerasi Seeder

Aplikasi ini menggunakan data profesi dan skill yang terintegrasi dari tim Data Science (`data_science/data/knowledge_base_skills.csv`). Agar database backend selalu selaras dengan pembaruan dataset, kami menyediakan script generator seeder otomatis.

### Meng-generate Ulang Seeder
Jika ada pembaruan data pada CSV di folder `data_science`, jalankan perintah berikut untuk meng-generate kembali file seeder `20260523073208-professions-skills.js`:

```bash
node scripts/generate_seeder.js
```

Script ini akan otomatis:
1. Membaca dataset terbaru dari `../data_science/data/knowledge_base_skills.csv`.
2. Melakukan ekstraksi profesi, skill unik, serta relasi pemetaannya.
3. Menghasilkan deskripsi otomatis berbahasa Indonesia untuk setiap skill berdasarkan kamus bawaan dan pola nama skill.
4. Menulis file seeder baru yang siap dieksekusi di `seeders/`.

---

## Perintah Berguna

```bash
# Jalankan dalam mode development (auto-restart)
npm run dev

# Jalankan migration untuk membuat tabel di database
npx sequelize-cli db:migrate

# Generate file seeder dari CSV Data Science
node scripts/generate_seeder.js

# Jalankan seeder (mengisi database dengan profesi dan skill)
npx sequelize-cli db:seed:all

# Undo semua seeder yang telah masuk ke database
npx sequelize-cli db:seed:undo:all

# Undo seeder profesi & skill secara spesifik
npx sequelize-cli db:seed:undo --seed 20260523073208-professions-skills.js
```