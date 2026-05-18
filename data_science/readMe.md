# penjelasan isi notebook
### 1. `Data_augmented_part_1.ipynb`
* **Tujuan:** Melakukan integrasi awal antara dataset utama (*Data Science Job Postings & Skills 2024*) dengan data hasil *web scraping* platform Jobstreet Indonesia.
* **Proses:** Melakukan pembersihan teks (*text cleaning*), standardisasi kolom, penentuan fungsionalitas penargetan profesi, serta filtrasi murni untuk 7 profesi IT target.
* **Evaluasi & Kendala:** Setelah dilakukan penggabungan, analisis distribusi data menunjukkan adanya *class imbalance* (ketidakseimbangan kelas) yang sangat ekstrem pada rumpun *Software Engineering / Web Development* (Frontend Developer hanya memiliki 36 data unik dan Fullstack Developer hanya 138 data unik). Jumlah sampel ini dinilai tidak mencukupi untuk melatih model NLP secara adil.
**Data_augmented_part_1.ipynb menghasilkan file dataset 7role.csv**

### 2. `Data_augmented_part_2.ipynb`
* **Tujuan:** Mengatasi masalah ketimpangan data (*class imbalance*) yang ditemukan pada Part 1 secara organik menggunakan strategi *Data Augmentation* (Augmentasi Data).
* **Proses:** * Mengintegrasikan dataset sekunder pertama (*Job Descriptions 2025*) dan menerapkan strategi *Cap Sampling* (membatasi penambahan maksimal 800 baris per profesi secara acak dengan `random_state=42`) agar volume data Backend Developer tidak mendominasi sistem.
  * Mengintegrasikan dataset sekunder kedua (*LinkedIn Software Engineering Jobs* oleh asaniczka) khusus untuk memperkuat representasi sampel pada profesi Frontend Developer.
  * Melakukan sinkronisasi indeks data yang lompat menggunakan fungsi `.reset_index(drop=True)`.
* **Hasil Akhir:** Berhasil mengunci dataset final yang bersih, murni (tanpa kolom duplikat `Unnamed: 0`), dan siap latih (*MLOps Ready*) pada angka **8.644 baris data unik organik**.
**Data_augmented_part_2.ipynb menghasilkan file dataset Bismillah_fix_dataset.csv** yang digunakan untuk training model


# link-link dataset
- https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills (main dataset) + scrapping web jobstreet path (dataset/dataset_jobstreet.csv)
- https://www.kaggle.com/datasets/adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles (support dataset)
- https://www.kaggle.com/datasets/asaniczka/software-engineer-job-postings-linkedin?select=postings.csv (support dataset)

