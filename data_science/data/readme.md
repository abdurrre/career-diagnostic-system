# Project ini menggunakan dua jenis raw dataset:
### Main Dataset 
→ dataset utama yang dipakai sebagai fondasi awal sistem dan berisi data kualifikasi berbagai profesi teknologi.

### Support Dataset 
→ dataset tambahan yang digunakan untuk membantu menyeimbangkan distribusi data, terutama untuk role Software Engineering yang jumlah datanya masih kurang di Main Dataset.

# Data hasil EDA
### knowledge_base_skills.csv
merupakan output utama dari proses EDA dan digunakan sebagai knowledge base sistem. File ini menjadi penghubung antara hasil analisis Data Scientist dengan proses scoring yang dikembangkan oleh AI Engineer. (opsional di gunakan oleh AI engineer (klo sudah ada))

Isi file terdiri dari beberapa informasi utama:
- **job_category**	(Nama profesi atau role)
- **skill**	(Nama skill)
- **frequency**	(Jumlah kemunculan skill pada job posting role terkait)
- **rank_in_role**	(Ranking skill berdasarkan frekuensi kemunculan)
- **tier**	(Tingkat prioritas skill)

# link-link public dataset
- https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills (main dataset) + scrapping web jobstreet path (dataset/dataset_jobstreet.csv)
- https://www.kaggle.com/datasets/adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles (support dataset)
- https://www.kaggle.com/datasets/asaniczka/software-engineer-job-postings-linkedin?select=postings.csv (support dataset)

# link drive dataset
https://drive.google.com/drive/folders/1G65uT53DOE_SvyaDLYrMGtliDcUnUMro?usp=share_link **(support_dataset)**
https://drive.google.com/drive/folders/1G65uT53DOE_SvyaDLYrMGtliDcUnUMro?usp=share_link **(main_dataset)**
