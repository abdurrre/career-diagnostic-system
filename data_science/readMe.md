# 📊 Data Science - Riset, Klasifikasi, & Pemetaan Keterampilan

Repositori ini menyimpan ruang kerja riset (*research workspace*) divisi Data Science untuk merancang, menguji, dan memvalidasi model pemrosesan bahasa alami (NLP) serta database klasifikasi kata kunci keterampilan (*skills classification database*).

---

## 🔬 Cakupan Kerja & Penelitian

1. **Exploratory Data Analysis (EDA):**
   * Menganalisis variasi kata kunci (*keywords*) keahlian yang sering muncul pada CV bidang teknologi dan ilmu komputer.
   * Melakukan pembersihan data (*data cleaning*), penghapusan kata tidak penting (*stop words*), dan stemming bahasa.
2. **Perancangan Database Klasifikasi Keterampilan:**
   * Menyusun pemetaan klasifikasi hubungan antara keterampilan dan tingkat urgensi profesi di industri:
     * **Critical:** Keahlian inti mutlak yang wajib dikuasai untuk bisa melamar.
     * **Important:** Keahlian teknis pendukung utama yang sangat dicari perekrut.
     * **Supplementary:** Keahlian pelengkap/sampingan yang memberikan nilai tambah daya saing.
3. **Validasi Model Prediktif:**
   * Merancang formula penghitungan skor kecocokan (*match score calculation algorithm*) yang adil, konsisten, dan realistis untuk mencerminkan keselarasan kualifikasi CV dengan standar industri teknologi terkini.

---

## 📂 Struktur Direktori

* **`notebooks/`**: Berisi file Jupyter Notebook (`.ipynb`) untuk eksperimen ekstraksi teks, klasifikasi kategori keterampilan, dan pemodelan statistik.
* **`datasets/`**: Kumpulan dataset benih (*seed data*) profesi, profil kompetensi minimal, dan daftar pustaka kosakata keahlian bidang IT.
* **`scripts/`**: Skrip otomatisasi Python untuk menyinkronkan data hasil riset ke database inti server backend atau FastAPI AI Engine.

---

## 🛠️ Persyaratan Lingkungan Riset

Untuk menjalankan ruang kerja riset ini, pasang paket pustaka analisis Python berikut:

```bash
cd data_science
pip install jupyter pandas numpy scikit-learn nltk
```

Jalankan Jupyter Notebook untuk memulai eksperimen:
```bash
jupyter notebook
```
