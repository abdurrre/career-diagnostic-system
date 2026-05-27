---
title: Career Diagnostic AI Engine
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# 🧠 AI Engine - Career Diagnostic Brain

Layanan API berbasis kecerdasan buatan (*AI Engine*) yang bertugas memproses data masukan teks, mengekstraksi informasi kualifikasi CV menggunakan algoritma Natural Language Processing (NLP), mencocokkannya dengan pedoman standar industri untuk profesi tertentu, serta menjalankan agen chatbot interaktif yang peka konteks.

---

## ⚡ Fitur Utama

1. **CV Text Parsing & Matching (`/api/scan`):**
   * Menerima ekstraksi data teks mentah dari file PDF CV.
   * Mencocokkan deskripsi keahlian pengguna dengan kriteria kebutuhan utama (*critical*), penting (*important*), dan pelengkap (*supplementary*) dari target profesi yang dipilih.
   * Menghitung nilai skor kecocokan (*match score*) secara prediktif dan mengembalikan daftar keterampilan teridentifikasi serta peta kesenjangan kompetensi secara dinamis.
2. **Context-Aware Conversational Agent (`/api/chat`):**
   * Mengoperasikan agen asisten obrolan chatbot interaktif.
   * Mengintegrasikan data kualifikasi laporan CV pengguna secara langsung ke dalam memori sesi chatbot, sehingga AI dapat menjawab pertanyaan mengenai karir Anda secara relevan, spesifik, dan taktis.
3. **Optimasi Kinerja FastAPI:**
   * Dibangun menggunakan struktur asinkronus Python berkinerja sangat tinggi, ramah penggunaan memori, dan siap menangani konkurensi tinggi.

---

## 🛠️ Spesifikasi Teknologi

* **Framework:** FastAPI (Python)
* **ASGI Server:** Uvicorn (kecepatan server asinkronus ultra cepat)
* **NLP & Processing:** Pustaka pemrosesan teks Python & integrasi API model NLP
* **Validasi Data:** Pydantic (menjamin tipe data masukan dan keluaran API aman)

---

## 🏃 Cara Menjalankan Layanan

1. Masuk ke direktori `ai_engine/`:
   ```bash
   cd ai_engine
   ```
2. Buat lingkungan virtual (*virtual environment*) Python (sangat disarankan):
   ```bash
   python -m venv venv
   # Aktifkan virtual environment
   ./venv/Scripts/activate # Windows (Command Prompt/PowerShell)
   source venv/bin/activate # macOS/Linux
   ```
3. Instal semua dependensi pustaka Python:
   ```bash
   pip install -r requirements.txt
   ```
4. Masuk ke sub-direktori kode sumber dan jalankan server menggunakan Uvicorn:
   ```bash
   cd src
   uvicorn api:app --reload
   ```
5. Layar dokumentasi interaktif OpenAPI (Swagger UI) dapat diakses pada alamat `http://localhost:8000/docs`.
