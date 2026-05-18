### 1. Data_augmented_part_1.ipynb
Notebook ini fokus pada proses **penggabungan dan preprocessing dataset utama**, mulai dari text cleaning, standardisasi kolom, hingga filtering untuk 7 profesi IT target.

Dari proses ini **dihasilkan dataset awal 7role.csv.**
Namun setelah dianalisis, **distribusi data masih sangat tidak seimbang**, terutama pada role Software Engineering seperti Frontend dan Fullstack Developer yang jumlah datanya masih terlalu sedikit untuk training model NLP.

### 2. Data_augmented_part_2.ipynb
Notebook ini dibuat untuk **mengatasi class imbalance dari Part 1** dengan menambahkan support dataset secara terukur menggunakan teknik cap sampling. 
Selain itu dilakukan sinkronisasi indeks dan pembersihan akhir dataset.

Hasil akhirnya adalah dataset final **Bismillah_fix_dataset.csv** yang berisi 8.644 data unik organik dan **sudah siap digunakan** untuk training model NLP.

## EDA
Pada tahap EDA, dataset final bismillah_fix_dataset.csv dianalisis untuk menjawab **enam business questions** yang sudah ditentukan sebelumnya. 
Sebelum proses analisis dilakukan, setiap job title terlebih dahulu dipetakan ke dalam **7 kategori profesi** menggunakan pendekatan rule-based keyword matching. Pendekatan ini dipilih karena dataset belum memiliki label kategori bawaan, sehingga proses mapping dilakukan berdasarkan pola keyword yang paling sering muncul pada job title di data.
