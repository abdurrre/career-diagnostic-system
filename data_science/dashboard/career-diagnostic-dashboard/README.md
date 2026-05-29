## Cara Menjalankan
1. Install dependencies
`pip install -r requirements.txt`

2. Jalankan dashboard
`streamlit run dashboard.py`

Dashboard akan terbuka otomatis di browser: `http://localhost:8501`

## Fitur Dashboard
### Halaman 1 - Overview Industri
- **4 KPI Cards**: Total Job Postings, Top Role, Unique Skills, Indonesia Market Share
- **Donut Chart**: Distribusi demand per 7 kategori role/profesi
- **Role Distribution within Each Market (%)**: Untuk mengetahui dominasi suatu role di market masing masing
- **Top 15 Skills Bar Chart**: Skill paling banyak dicari secara keseluruhan
- **Search Skill**: Cari skill spesifik dan lihat seberapa laku di tiap role

### Halaman 2 - Role Analysis
- **Dropdown Role Selector**: Pilih profesi untuk dianalisis
- **Horizontal Bar Chart**: Top 10 skills dengan color-coding tier (Critical/Important/Supplementary)
- **Violin Plot**: Distribusi jumlah skill yang diminta per job posting + statistik mean/median/max
- **Skill Table**: Tabel lengkap skill + tier + frekuensi dengan color-coding

### Halaman 3 - Skill Intelligence
- **Skill Frequency Heatmap**: membantu membandingkan kebutuhan skill antar role, melihat specialization pattern, memahami overlap skill antar profesi
- **Skill Co-occurrence Network**: Menunjukkan skill apa yang sering muncul bersamaan dalam satu job posting
- **Normalized Skill Demand: Global vs Indonesia**: Menunjukkan skill yang paling banyak diminta industri secara keseluruhan