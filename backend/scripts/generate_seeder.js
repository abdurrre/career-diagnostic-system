/**
 * generate_seeder.js
 * Script untuk generate file seeder dari knowledge_base_skills.csv
 * Menghasilkan deskripsi otomatis dalam Bahasa Indonesia untuk setiap skill.
 *
 * Usage: node generate_seeder.js
 * Output: ../seeders/20260523073208-professions-skills.js
 */

const fs = require("fs");
const path = require("path");
const readline = require("readline");

// ─── Descriptions dictionary ────────────────────────────────────────────────
// Deskripsi khusus untuk skill yang paling umum / butuh penjelasan akurat
const customDescriptions = {
  // Languages
  sql: "Bahasa query standar untuk mengelola dan mengambil data dari database relasional.",
  python:
    "Bahasa pemrograman serbaguna yang banyak digunakan dalam data science, AI, dan pengembangan backend.",
  javascript:
    "Bahasa pemrograman utama web yang digunakan untuk membangun antarmuka interaktif dan logika aplikasi.",
  typescript:
    "Superset JavaScript dengan sistem tipe statis untuk pengembangan aplikasi skala besar.",
  java: "Bahasa pemrograman berbasis objek yang banyak digunakan untuk aplikasi enterprise dan backend.",
  "c++":
    "Bahasa pemrograman performa tinggi untuk sistem tertanam, game, dan komputasi ilmiah.",
  "c#": "Bahasa pemrograman modern dari Microsoft untuk aplikasi Windows, web, dan game berbasis .NET.",
  go: "Bahasa pemrograman dari Google yang efisien dan cepat untuk sistem dan layanan backend.",
  ruby: "Bahasa pemrograman dinamis yang dikenal dengan sintaksis bersih dan framework Ruby on Rails.",
  php: "Bahasa skrip server-side yang banyak digunakan untuk pengembangan web dinamis.",
  scala:
    "Bahasa pemrograman yang menggabungkan paradigma OOP dan fungsional, populer di ekosistem big data.",
  kotlin:
    "Bahasa pemrograman modern untuk Android dan backend, interoperable dengan Java.",
  swift:
    "Bahasa pemrograman dari Apple untuk pengembangan aplikasi iOS dan macOS.",
  rust: "Bahasa pemrograman sistem dengan jaminan keamanan memori tanpa garbage collector.",
  r: "Bahasa pemrograman dan lingkungan statistik untuk analisis data dan visualisasi ilmiah.",
  matlab:
    "Lingkungan komputasi numerik dan bahasa pemrograman untuk analisis teknik dan sains.",
  perl: "Bahasa pemrograman untuk pemrosesan teks, administrasi sistem, dan scripting.",
  shell: "Skrip baris perintah untuk otomasi tugas pada sistem Unix/Linux.",
  bash: "Shell Unix dan bahasa skrip untuk otomasi tugas dan administrasi sistem Linux.",
  powershell:
    "Shell scripting dan bahasa otomasi dari Microsoft untuk administrasi Windows.",
  lua: "Bahasa skrip ringan yang sering digunakan untuk scripting game dan aplikasi tertanam.",
  dart: "Bahasa pemrograman dari Google untuk membangun aplikasi mobile dan web menggunakan Flutter.",
  elixir:
    "Bahasa fungsional berbasis Erlang VM untuk sistem terdistribusi dan fault-tolerant.",
  haskell:
    "Bahasa pemrograman fungsional murni dengan sistem tipe statis yang kuat.",
  clojure:
    "Dialek Lisp modern di atas JVM, dirancang untuk pemrograman konkuren dan fungsional.",
  groovy:
    "Bahasa dinamis di atas JVM yang kompatibel dengan Java dan populer dalam CI/CD (Groovy/Jenkins).",

  // Cloud & Infrastructure
  aws: "Platform cloud dari Amazon yang menyediakan layanan infrastruktur, komputasi, dan penyimpanan data.",
  azure:
    "Platform cloud dari Microsoft untuk membangun, mengelola, dan mendeploy aplikasi dan layanan.",
  gcp: "Google Cloud Platform — layanan cloud dari Google untuk komputasi, penyimpanan, dan machine learning.",
  "google cloud":
    "Platform cloud dari Google yang menyediakan layanan komputasi, penyimpanan, dan AI.",
  docker:
    "Platform containerisasi untuk mengemas aplikasi beserta dependensinya agar mudah dijalankan di mana saja.",
  kubernetes:
    "Platform orkestrasi container untuk mengotomatiskan deployment, scaling, dan pengelolaan aplikasi.",
  terraform:
    "Alat Infrastructure as Code (IaC) untuk mendefinisikan dan menyediakan infrastruktur cloud secara deklaratif.",
  ansible:
    "Alat otomasi IT untuk konfigurasi, deployment, dan orkestrasi infrastruktur tanpa agen.",
  jenkins:
    "Server otomasi open-source untuk membangun pipeline CI/CD dan mengotomatiskan proses pengembangan.",
  git: "Sistem kontrol versi terdistribusi untuk melacak perubahan kode dan kolaborasi tim.",
  github:
    "Platform hosting kode berbasis Git dengan fitur kolaborasi, CI/CD, dan manajemen proyek.",
  gitlab:
    "Platform DevOps lengkap berbasis Git dengan CI/CD terintegrasi untuk siklus pengembangan.",
  "version control":
    "Praktik dan alat untuk melacak dan mengelola perubahan kode sumber dari waktu ke waktu.",
  linux:
    "Sistem operasi open-source berbasis Unix yang banyak digunakan untuk server dan pengembangan.",
  unix: "Sistem operasi multi-tasking dan multi-user yang menjadi dasar banyak sistem modern.",
  ci: "Continuous Integration — praktik mengintegrasikan kode secara otomatis dan sering ke repository bersama.",
  cd: "Continuous Delivery/Deployment — otomasi pengiriman kode ke lingkungan staging atau produksi.",
  "ci/cd":
    "Continuous Integration & Continuous Deployment — praktik otomasi build, test, dan deployment perangkat lunak.",

  // Data Engineering
  spark:
    "Framework pemrosesan data terdistribusi berkecepatan tinggi untuk analitik big data dalam skala besar.",
  hadoop:
    "Framework open-source untuk pemrosesan dan penyimpanan dataset besar secara terdistribusi.",
  etl: "Proses Extract, Transform, Load untuk memindahkan dan mengolah data antar sistem penyimpanan.",
  kafka:
    "Platform streaming data terdistribusi dari Apache untuk membangun pipeline data real-time.",
  airflow:
    "Platform orkestrasi workflow dari Apache untuk menjadwalkan dan memantau pipeline data.",
  "apache airflow":
    "Platform orkestrasi workflow dari Apache untuk menjadwalkan dan memantau pipeline data.",
  "apache spark":
    "Framework komputasi terdistribusi dari Apache untuk pemrosesan data berskala besar.",
  "apache kafka":
    "Platform streaming event terdistribusi dari Apache untuk pipeline data real-time.",
  flink:
    "Framework pemrosesan stream data real-time dari Apache untuk analitik event berkecepatan tinggi.",
  hive: "Data warehouse berbasis Hadoop untuk kueri dan analisis data berstruktur dalam skala besar.",
  "data modeling":
    "Proses merancang struktur data untuk merepresentasikan entitas dan hubungannya secara efisien.",
  "data warehouse":
    "Sistem penyimpanan data terpusat yang dirancang untuk analisis dan pelaporan bisnis.",
  "data lake":
    "Repositori penyimpanan data besar dalam format mentah untuk analisis fleksibel di masa depan.",
  "data pipeline":
    "Rangkaian proses otomatis untuk mengumpulkan, mengolah, dan mengirimkan data dari sumber ke tujuan.",
  "data engineering":
    "Disiplin membangun sistem dan infrastruktur untuk pengumpulan, penyimpanan, dan pemrosesan data.",
  "data governance":
    "Kerangka kebijakan dan proses untuk memastikan kualitas, keamanan, dan kepatuhan penggunaan data.",
  "data quality":
    "Praktik dan alat untuk memastikan akurasi, kelengkapan, dan konsistensi data dalam sistem.",
  dbt: "Alat transformasi data yang memungkinkan analis dan engineer menulis transformasi SQL yang dapat diuji.",
  fivetran:
    "Platform ELT managed yang mengotomatiskan pipeline data dari berbagai sumber ke data warehouse.",
  snowflake:
    "Platform data cloud yang menyediakan data warehouse, data lake, dan berbagi data secara terpadu.",
  redshift:
    "Data warehouse berbasis cloud dari AWS yang dioptimalkan untuk analitik skala besar.",
  bigquery:
    "Data warehouse serverless dari Google Cloud untuk analitik data berskala petabyte.",
  databricks:
    "Platform data dan AI berbasis cloud yang mengintegrasikan Apache Spark dengan tools kolaborasi.",

  // Databases
  postgresql:
    "Sistem database relasional open-source yang kuat dengan dukungan fitur SQL tingkat lanjut.",
  mysql:
    "Sistem manajemen database relasional open-source yang populer untuk aplikasi web.",
  mongodb:
    "Database NoSQL berbasis dokumen yang fleksibel untuk menyimpan data tidak terstruktur.",
  redis:
    "Database in-memory berkecepatan tinggi untuk caching, session management, dan pub/sub.",
  elasticsearch:
    "Mesin pencari dan analitik terdistribusi berbasis Lucene untuk pencarian teks lengkap dan log.",
  "sql server":
    "Sistem manajemen database relasional dari Microsoft untuk aplikasi enterprise.",
  oracle:
    "Sistem database relasional enterprise dari Oracle Corporation dengan fitur keamanan tingkat tinggi.",
  cassandra:
    "Database NoSQL terdistribusi yang dirancang untuk ketersediaan tinggi dan skalabilitas horizontal.",
  dynamodb:
    "Database NoSQL managed dari AWS yang fully serverless dengan performa konsisten.",
  sqlite:
    "Database relasional ringan berbasis file yang ideal untuk pengembangan dan aplikasi mobile.",
  neo4j: "Database graf untuk menyimpan dan mengkueri data yang memiliki relasi kompleks antar entitas.",

  // ML & AI
  "machine learning":
    "Cabang AI yang memungkinkan sistem belajar dari data tanpa pemrograman eksplisit.",
  "deep learning":
    "Subfield machine learning menggunakan jaringan saraf berlapis untuk mengenali pola kompleks dalam data.",
  tensorflow:
    "Framework open-source dari Google untuk membangun dan melatih model machine learning dan deep learning.",
  pytorch:
    "Framework deep learning dari Meta yang populer untuk riset dan pengembangan model AI.",
  "scikit-learn":
    "Library machine learning Python untuk klasifikasi, regresi, clustering, dan preprocessing data.",
  keras:
    "API deep learning tingkat tinggi yang berjalan di atas TensorFlow untuk prototyping cepat.",
  nlp: "Natural Language Processing — cabang AI untuk memahami, memproses, dan menghasilkan bahasa manusia.",
  "natural language processing":
    "Bidang AI yang berfokus pada kemampuan komputer untuk memahami dan berinteraksi dengan bahasa manusia.",
  "computer vision":
    "Bidang AI yang memungkinkan komputer menginterpretasikan dan memahami konten visual dari gambar/video.",
  ai: "Bidang ilmu komputer yang berfokus pada pembuatan sistem yang mampu meniru kecerdasan manusia.",
  "artificial intelligence":
    "Teknologi yang memungkinkan mesin melakukan tugas yang biasanya membutuhkan kecerdasan manusia.",
  llm: "Large Language Model — model AI berskala besar yang dilatih untuk memahami dan menghasilkan teks.",
  mlops:
    "Praktik DevOps untuk machine learning: otomasi training, deployment, dan monitoring model ML di produksi.",
  "feature engineering":
    "Proses membuat dan memilih fitur input yang relevan untuk meningkatkan performa model machine learning.",
  "model deployment":
    "Proses mengintegrasikan model machine learning ke dalam sistem produksi untuk penggunaan nyata.",
  "reinforcement learning":
    "Paradigma machine learning di mana agen belajar melalui interaksi dan umpan balik reward dari lingkungan.",
  "generative ai":
    "Jenis AI yang mampu menghasilkan konten baru (teks, gambar, kode) berdasarkan pola yang dipelajari.",
  "transfer learning":
    "Teknik machine learning yang memanfaatkan model yang sudah dilatih untuk tugas baru yang terkait.",
  "model training":
    "Proses mengoptimalkan parameter model machine learning menggunakan dataset latih.",

  // Data Analysis & Visualization
  tableau:
    "Platform visualisasi data interaktif untuk membuat dashboard dan laporan analitik.",
  "power bi":
    "Alat business intelligence Microsoft untuk membuat laporan dan dashboard data secara interaktif.",
  excel:
    "Aplikasi spreadsheet Microsoft untuk analisis data, pembuatan formula, dan visualisasi dasar.",
  looker:
    "Platform business intelligence berbasis SQL untuk eksplorasi data dan pembuatan dashboard.",
  "data visualization":
    "Proses menyajikan data dalam bentuk grafis (chart, grafik) agar mudah dipahami dan dianalisis.",
  "data analysis":
    "Proses memeriksa, membersihkan, dan memodelkan data untuk mengungkap informasi berguna dan insight.",
  statistics:
    "Ilmu pengumpulan, analisis, interpretasi, dan penyajian data untuk pengambilan keputusan.",
  pandas:
    "Library Python untuk manipulasi dan analisis data tabular menggunakan struktur DataFrame.",
  numpy:
    "Library Python untuk komputasi numerik dengan dukungan array multidimensi dan fungsi matematika.",
  matplotlib:
    "Library visualisasi data Python untuk membuat grafik statis, animasi, dan interaktif.",
  seaborn:
    "Library visualisasi statistik Python berbasis Matplotlib dengan API yang lebih sederhana.",
  plotly:
    "Library grafik interaktif Python/JavaScript untuk membuat visualisasi data yang kaya fitur.",

  // Web Development
  react:
    "Library JavaScript dari Meta untuk membangun antarmuka pengguna berbasis komponen.",
  angular:
    "Framework frontend berbasis TypeScript dari Google untuk membangun aplikasi web skala enterprise.",
  "vue.js":
    "Framework JavaScript progresif untuk membangun antarmuka pengguna yang ringan dan fleksibel.",
  vuejs:
    "Framework JavaScript progresif untuk membangun antarmuka pengguna yang ringan dan fleksibel.",
  vue: "Framework JavaScript progresif untuk membangun antarmuka pengguna yang ringan dan fleksibel.",
  "next.js":
    "Framework React untuk produksi dengan SSR, SSG, dan routing berbasis file system.",
  nextjs:
    "Framework React untuk produksi dengan rendering sisi server dan generasi halaman statis.",
  "node.js":
    "Runtime JavaScript di sisi server berbasis V8 untuk membangun aplikasi backend yang cepat.",
  nodejs:
    "Runtime JavaScript di sisi server berbasis V8 untuk membangun aplikasi backend yang cepat.",
  express:
    "Framework web minimalis untuk Node.js yang menyederhanakan pembuatan API dan server HTTP.",
  "express.js":
    "Framework web minimalis untuk Node.js yang menyederhanakan pembuatan API dan server HTTP.",
  django:
    "Framework web Python berfitur lengkap dengan prinsip 'batteries included' untuk pengembangan cepat.",
  flask:
    "Microframework web Python yang ringan dan fleksibel untuk membangun API dan aplikasi web.",
  "rest api":
    "Arsitektur antarmuka pemrograman berbasis HTTP yang menggunakan prinsip REST untuk komunikasi layanan.",
  graphql:
    "Bahasa query untuk API yang memungkinkan klien meminta data sesuai kebutuhan secara presisi.",
  html: "Bahasa markup standar untuk menyusun struktur dan konten halaman web.",
  css: "Bahasa stylesheet untuk mengatur tampilan dan layout halaman web.",
  "css/scss/sass/less":
    "Bahasa stylesheet dan preprocessornya untuk mengatur tampilan halaman web dengan fitur lebih canggih.",
  scss:
    "Preprocessor CSS yang menambahkan variabel, nesting, dan mixin untuk penulisan stylesheet yang lebih efisien.",
  sass: "Preprocessor CSS yang memperkenalkan fitur seperti variabel, nesting, dan fungsi untuk pengelolaan style.",
  tailwind:
    "Framework CSS utility-first untuk membangun antarmuka kustom dengan cepat langsung di markup.",
  "tailwind css":
    "Framework CSS utility-first untuk membangun antarmuka kustom dengan cepat langsung di markup.",
  bootstrap:
    "Framework CSS populer dengan komponen UI siap pakai untuk pengembangan web responsif.",
  webpack:
    "Modul bundler JavaScript untuk mengemas aset, kode, dan dependensi aplikasi web.",
  vite: "Build tool frontend generasi berikutnya yang sangat cepat dengan Hot Module Replacement.",
  "frontend development":
    "Praktik pengembangan sisi klien yang mencakup UI, UX, performa, dan aksesibilitas web.",

  // DevOps & Tools
  agile:
    "Metodologi pengembangan perangkat lunak iteratif yang menekankan kolaborasi tim dan adaptasi cepat.",
  scrum:
    "Framework agile berbasis sprint pendek dengan peran Product Owner, Scrum Master, dan tim pengembang.",
  kanban:
    "Metode manajemen alur kerja visual yang mengoptimalkan proses melalui batasan pekerjaan dalam proses.",
  jira: "Alat manajemen proyek dan pelacakan isu yang populer untuk tim agile dan DevOps.",
  confluence:
    "Platform kolaborasi dan dokumentasi tim dari Atlassian untuk berbagi pengetahuan.",
  trello:
    "Alat manajemen proyek visual berbasis papan Kanban untuk mengorganisir tugas tim.",
  sdlc: "Software Development Life Cycle — siklus hidup pengembangan perangkat lunak dari perencanaan hingga pemeliharaan.",
  "code review":
    "Proses evaluasi kode oleh rekan tim untuk memastikan kualitas, keamanan, dan konsistensi standar.",
  testing:
    "Proses memverifikasi bahwa perangkat lunak berfungsi sesuai spesifikasi dan bebas dari bug.",
  "unit testing":
    "Pengujian individual unit kode (fungsi/metode) secara terisolasi untuk memastikan kebenaran logika.",
  "integration testing":
    "Pengujian interaksi antar komponen atau layanan untuk memastikan sistem bekerja bersama dengan benar.",
  "test-driven development":
    "Metodologi pengembangan yang menulis tes sebelum kode implementasi untuk memastikan kualitas.",
  tdd: "Test-Driven Development — praktik menulis tes otomatis sebelum menulis kode implementasi.",
  devops:
    "Praktik budaya dan teknik untuk mengintegrasikan pengembangan software dan operasi IT.",
  "microservices":
    "Arsitektur perangkat lunak yang membagi aplikasi menjadi layanan kecil independen yang dapat di-deploy sendiri.",

  // Soft skills & general
  "problem solving":
    "Kemampuan mengidentifikasi, menganalisis, dan menyelesaikan masalah secara efektif dan efisien.",
  communication:
    "Kemampuan menyampaikan ide dan informasi secara jelas dan efektif kepada berbagai pihak.",
  leadership:
    "Kemampuan menginspirasi, memotivasi, dan mengarahkan tim menuju tujuan bersama.",
  teamwork:
    "Kemampuan bekerja sama secara efektif dalam tim untuk mencapai tujuan bersama.",
  collaboration:
    "Kemampuan bekerja sama dengan anggota tim dan pemangku kepentingan lintas fungsi.",
  "project management":
    "Proses merencanakan, mengeksekusi, dan mengendalikan proyek agar selesai tepat waktu dan sesuai anggaran.",
  "time management":
    "Kemampuan mengatur dan memprioritaskan tugas untuk memaksimalkan produktivitas dalam batas waktu.",
  mentoring:
    "Proses membimbing dan berbagi pengetahuan dengan anggota tim yang lebih junior untuk pengembangan mereka.",
  documentation:
    "Praktik menulis dan memelihara dokumentasi teknis yang jelas untuk produk dan proses.",

  // Security
  cybersecurity:
    "Praktik melindungi sistem, jaringan, dan data dari serangan digital dan akses tidak sah.",
  security:
    "Disiplin mengamankan sistem informasi dari ancaman, kerentanan, dan pelanggaran data.",
  "network security":
    "Praktik melindungi infrastruktur jaringan dari ancaman, intrusi, dan serangan siber.",
  oauth:
    "Protokol otorisasi open standard yang memungkinkan akses aman ke sumber daya tanpa berbagi kata sandi.",
  "oauth/openid":
    "Protokol otorisasi (OAuth) dan autentikasi (OpenID Connect) untuk keamanan akses aplikasi web.",
  jwt: "JSON Web Token — standar terbuka untuk transmisi informasi identitas secara aman antar pihak.",
  ssl: "Secure Sockets Layer — protokol kriptografi untuk mengamankan komunikasi jaringan.",
  tls: "Transport Layer Security — protokol kriptografi untuk mengamankan komunikasi melalui jaringan komputer.",
  https:
    "HTTP Secure — versi aman dari HTTP menggunakan enkripsi TLS untuk komunikasi web.",

  // Mobile
  android:
    "Platform mobile dari Google untuk pengembangan aplikasi pada perangkat berbasis Android.",
  ios: "Sistem operasi mobile Apple untuk iPhone dan iPad, target utama pengembangan aplikasi native.",
  flutter:
    "Framework UI cross-platform dari Google menggunakan bahasa Dart untuk aplikasi mobile dan web.",
  "react native":
    "Framework dari Meta untuk membangun aplikasi mobile native menggunakan JavaScript dan React.",

  // Misc commonly seen
  "data science":
    "Bidang interdisiplin yang menggunakan metode ilmiah, algoritma, dan sistem untuk mengekstrak pengetahuan dari data.",
  "business intelligence":
    "Proses dan teknologi menganalisis data bisnis untuk mendukung pengambilan keputusan strategis.",
  "object-oriented programming":
    "Paradigma pemrograman yang mengorganisir kode ke dalam objek yang memiliki atribut dan perilaku.",
  oop: "Object-Oriented Programming — paradigma pemrograman berbasis objek dengan enkapsulasi, pewarisan, dan polimorfisme.",
  "functional programming":
    "Paradigma pemrograman yang memperlakukan komputasi sebagai evaluasi fungsi matematika tanpa efek samping.",
  "design patterns":
    "Solusi yang dapat digunakan kembali untuk masalah umum dalam desain perangkat lunak.",
  "software design patterns":
    "Template solusi yang dapat digunakan kembali untuk masalah desain yang sering muncul dalam pengembangan software.",
  "system design":
    "Proses mendefinisikan arsitektur, komponen, dan antarmuka sistem untuk memenuhi persyaratan tertentu.",
  "api development":
    "Proses merancang dan membangun Application Programming Interface untuk komunikasi antar sistem.",
  "restful api":
    "API yang mengikuti prinsip arsitektur REST untuk komunikasi stateless antar klien dan server.",
  microservices:
    "Arsitektur perangkat lunak yang membagi aplikasi menjadi layanan kecil independen yang dapat di-deploy mandiri.",
  "cloud computing":
    "Model pengiriman layanan komputasi (server, storage, database) melalui internet secara on-demand.",
  "agile methodology":
    "Pendekatan pengembangan perangkat lunak iteratif yang menekankan fleksibilitas, kolaborasi, dan pengiriman berkelanjutan.",
  "software architecture":
    "Struktur tingkat tinggi suatu sistem perangkat lunak dan prinsip-prinsip yang memandu desainnya.",
  devops:
    "Budaya dan praktik yang menyatukan pengembangan software dan operasi IT untuk siklus delivery lebih cepat.",
  "site reliability engineering":
    "Disiplin yang menerapkan rekayasa perangkat lunak pada masalah operasi IT untuk meningkatkan keandalan sistem.",
  sre: "Site Reliability Engineering — praktik menerapkan teknik software untuk meningkatkan keandalan dan skalabilitas sistem.",
};

// ─── Generate description ────────────────────────────────────────────────────
function generateDescription(skillName) {
  const lower = skillName.toLowerCase().trim();

  // Check custom descriptions first
  if (customDescriptions[lower]) {
    return customDescriptions[lower];
  }

  // Pattern-based generation
  // Cloud & Infrastructure patterns
  if (lower.includes("aws") && lower !== "aws")
    return `Layanan ${skillName} dari Amazon Web Services untuk infrastruktur dan komputasi cloud.`;
  if (lower.includes("azure") && lower !== "azure")
    return `Layanan ${skillName} dari Microsoft Azure untuk kebutuhan cloud enterprise.`;
  if (lower.includes("google cloud") || lower.includes("gcp"))
    return `Layanan ${skillName} dari Google Cloud Platform untuk komputasi dan analitik data.`;

  // Framework/Library patterns
  if (
    lower.includes("react") ||
    lower.includes("angular") ||
    lower.includes("vue")
  ) {
    return `Framework/library frontend JavaScript ${skillName} untuk membangun antarmuka pengguna modern.`;
  }

  // Database patterns
  if (
    lower.includes("database") ||
    lower.includes("db") ||
    lower.includes("sql") ||
    lower.includes("nosql")
  ) {
    return `Sistem atau teknik pengelolaan database ${skillName} untuk penyimpanan dan pengambilan data.`;
  }

  // API patterns
  if (lower.includes("api")) {
    return `Teknik atau protokol ${skillName} untuk membangun dan mengintegrasikan antarmuka pemrograman aplikasi.`;
  }

  // Testing patterns
  if (lower.includes("test") || lower.includes("testing")) {
    return `Praktik dan metodologi ${skillName} untuk memastikan kualitas dan keandalan perangkat lunak.`;
  }

  // CI/CD patterns
  if (lower.includes("ci/cd") || lower.includes("pipeline")) {
    return `Alat atau konsep ${skillName} dalam pipeline otomasi build, integrasi, dan deployment perangkat lunak.`;
  }

  // Machine learning patterns
  if (
    lower.includes("learning") ||
    lower.includes("model") ||
    lower.includes("neural") ||
    lower.includes("prediction")
  ) {
    return `Teknik atau konsep ${skillName} dalam bidang machine learning dan kecerdasan buatan.`;
  }

  // Data patterns
  if (
    lower.includes("data") ||
    lower.includes("analytics") ||
    lower.includes("analysis")
  ) {
    return `Alat atau metodologi ${skillName} untuk pengelolaan, analisis, dan pengolahan data.`;
  }

  // Security patterns
  if (
    lower.includes("security") ||
    lower.includes("auth") ||
    lower.includes("encrypt") ||
    lower.includes("ssl") ||
    lower.includes("tls")
  ) {
    return `Mekanisme atau protokol ${skillName} untuk mengamankan sistem dan data dari ancaman siber.`;
  }

  // Container/orchestration patterns
  if (
    lower.includes("container") ||
    lower.includes("docker") ||
    lower.includes("kubernetes") ||
    lower.includes("k8s")
  ) {
    return `Teknologi ${skillName} untuk containerisasi dan orkestrasi aplikasi dalam lingkungan cloud-native.`;
  }

  // Mobile patterns
  if (
    lower.includes("ios") ||
    lower.includes("android") ||
    lower.includes("mobile")
  ) {
    return `Teknologi atau framework ${skillName} untuk pengembangan aplikasi mobile.`;
  }

  // Frontend patterns
  if (
    lower.includes("frontend") ||
    lower.includes("front-end") ||
    lower.includes("front end") ||
    lower.includes("ui") ||
    lower.includes("ux")
  ) {
    return `Teknologi atau praktik ${skillName} dalam pengembangan antarmuka pengguna web.`;
  }

  // Backend patterns
  if (
    lower.includes("backend") ||
    lower.includes("back-end") ||
    lower.includes("server")
  ) {
    return `Teknologi atau konsep ${skillName} dalam pengembangan logika dan layanan sisi server.`;
  }

  // Management patterns
  if (lower.includes("management") || lower.includes("agile")) {
    return `Metodologi atau praktik ${skillName} untuk mengelola proyek dan tim pengembangan secara efektif.`;
  }

  // Development patterns
  if (lower.includes("development") || lower.includes("programming")) {
    return `Keterampilan atau metodologi ${skillName} dalam proses pengembangan perangkat lunak.`;
  }

  // Design patterns
  if (lower.includes("design") || lower.includes("architecture")) {
    return `Pendekatan atau pola ${skillName} dalam perancangan sistem dan arsitektur perangkat lunak.`;
  }

  // Deployment/DevOps patterns
  if (
    lower.includes("deploy") ||
    lower.includes("devops") ||
    lower.includes("automation")
  ) {
    return `Proses atau alat ${skillName} dalam otomasi dan pengoperasian sistem perangkat lunak.`;
  }

  // Performance patterns
  if (lower.includes("performance") || lower.includes("optimization")) {
    return `Teknik ${skillName} untuk meningkatkan performa dan efisiensi aplikasi dan sistem.`;
  }

  // Communication/collaboration
  if (
    lower.includes("communication") ||
    lower.includes("collaboration") ||
    lower.includes("team")
  ) {
    return `Kemampuan ${skillName} yang penting untuk bekerja secara efektif dalam tim pengembangan.`;
  }

  // Specific tools recognition
  const toolsMap = {
    jira: "Alat manajemen proyek dan pelacakan isu untuk tim agile.",
    confluence: "Platform kolaborasi dan dokumentasi tim dari Atlassian.",
    slack:
      "Platform komunikasi tim berbasis pesan untuk kolaborasi di tempat kerja.",
    figma:
      "Alat desain UI/UX berbasis cloud untuk kolaborasi tim desain produk.",
    postman:
      "Platform pengembangan API untuk pengujian, dokumentasi, dan kolaborasi.",
    nginx:
      "Server web dan reverse proxy berkinerja tinggi untuk melayani aplikasi web.",
    apache:
      "Server web open-source paling populer untuk melayani konten web dan aplikasi.",
    rabbitmq:
      "Message broker open-source untuk komunikasi asinkron antar layanan.",
    celery:
      "Task queue terdistribusi untuk Python yang memproses tugas secara asinkron.",
    grafana:
      "Platform visualisasi dan monitoring open-source untuk data metrik dan log.",
    prometheus:
      "Sistem monitoring dan alerting open-source berbasis time-series.",
    sonarqube:
      "Platform analisis kualitas kode untuk mendeteksi bug, kerentanan, dan code smell.",
    splunk:
      "Platform untuk mencari, memantau, dan menganalisis data machine-generated.",
    datadog:
      "Platform monitoring cloud untuk infrastruktur, aplikasi, dan log.",
    kibana:
      "Antarmuka visualisasi data untuk Elasticsearch, digunakan untuk analitik log.",
    logstash:
      "Pipeline pemrosesan data open-source untuk mengumpulkan dan mentransformasi log.",
    zookeeper:
      "Layanan koordinasi terdistribusi untuk sinkronisasi dan konfigurasi cluster.",
    yarn: "Package manager JavaScript alternatif npm yang lebih cepat dan deterministik.",
    npm: "Node Package Manager — manajer paket default untuk ekosistem JavaScript/Node.js.",
    gradle:
      "Sistem build otomasi fleksibel untuk proyek Java, Android, dan bahasa lainnya.",
    maven:
      "Alat build dan manajemen dependensi untuk proyek Java berbasis konfigurasi XML.",
    junit:
      "Framework testing unit populer untuk aplikasi Java yang mendukung TDD.",
    selenium:
      "Framework pengujian otomasi browser untuk memvalidasi aplikasi web secara end-to-end.",
    cypress:
      "Framework pengujian end-to-end modern untuk aplikasi web dengan debugging interaktif.",
    jest: "Framework testing JavaScript yang ringan dengan fokus pada kesederhanaan dan kecepatan.",
    mocha:
      "Framework testing JavaScript fleksibel yang berjalan di Node.js dan browser.",
    redux:
      "Library manajemen state yang dapat diprediksi untuk aplikasi JavaScript.",
    mobx: "Library manajemen state reaktif untuk aplikasi JavaScript menggunakan observable.",
    graphql:
      "Bahasa query untuk API yang memungkinkan klien meminta data sesuai kebutuhan secara presisi.",
    grpc: "Framework RPC modern dari Google untuk komunikasi antar layanan yang efisien.",
    protobuf:
      "Format serialisasi data biner dari Google yang efisien untuk komunikasi antar layanan.",
    websocket:
      "Protokol komunikasi dua arah real-time antara klien dan server melalui satu koneksi.",
    oauth2:
      "Versi terbaru protokol otorisasi OAuth untuk keamanan akses aplikasi modern.",
    jwt: "JSON Web Token — standar untuk transmisi informasi identitas secara aman antar pihak.",
    markdown:
      "Bahasa markup ringan untuk memformat teks dengan sintaksis yang mudah dibaca.",
    latex:
      "Sistem persiapan dokumen ilmiah berkualitas tinggi, populer di akademia dan penelitian.",
    xml: "eXtensible Markup Language — format data terstruktur yang dapat dibaca manusia dan mesin.",
    json: "JavaScript Object Notation — format pertukaran data ringan berbasis teks yang mudah dibaca.",
    yaml: "YAML Ain't Markup Language — format serialisasi data yang mudah dibaca manusia untuk konfigurasi.",
    regex:
      "Regular Expression — pola pencarian teks yang kuat untuk pencocokan dan manipulasi string.",
    "machine learning":
      "Cabang AI yang memungkinkan sistem belajar dari data tanpa pemrograman eksplisit.",
  };

  if (toolsMap[lower]) return toolsMap[lower];

  // Default fallback — generate a generic but reasonable description
  const capitalized = skillName
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
  return `${capitalized} — kemampuan atau teknologi yang digunakan dalam pengembangan software dan sistem informasi modern.`;
}

// ─── Main ────────────────────────────────────────────────────────────────────
async function main() {
  const csvPath = path.join(
    __dirname,
    "..",
    "..",
    "data_science",
    "data",
    "knowledge_base_skills.csv"
  );
  const outputPath = path.join(
    __dirname,
    "..",
    "seeders",
    "20260523073208-professions-skills.js"
  );

  console.log("Reading CSV:", csvPath);

  const rl = readline.createInterface({
    input: fs.createReadStream(csvPath),
    crlfDelay: Infinity,
  });

  const professionSet = new Set();
  const skillSet = new Map(); // name -> description
  const professionSkills = []; // [profName, skillName, tier]
  const seenProfSkill = new Set();

  let lineNumber = 0;
  let headers = [];

  for await (const line of rl) {
    lineNumber++;
    if (lineNumber === 1) {
      headers = line.split(";").map((h) => h.trim().replace(/\r/g, ""));
      console.log("Headers:", headers);
      continue;
    }

    const parts = line.split(";");
    if (parts.length < 6) continue;

    const jobCategory = parts[0].trim().replace(/\r/g, "");
    const skill = parts[1].trim().replace(/\r/g, "");
    const tier = parts[5].trim().replace(/\r/g, "");

    if (!jobCategory || !skill) continue;

    // Add profession
    professionSet.add(jobCategory);

    // Add skill with generated description
    if (!skillSet.has(skill.toLowerCase())) {
      skillSet.set(skill.toLowerCase(), {
        name: skill,
        description: generateDescription(skill),
      });
    }

    // Add profession-skill mapping (avoid duplicates)
    const key = `${jobCategory}|${skill.toLowerCase()}`;
    if (!seenProfSkill.has(key)) {
      seenProfSkill.add(key);
      // Normalize tier
      let normalizedTier = "supplementary";
      const tierLower = tier.toLowerCase();
      if (tierLower === "critical") normalizedTier = "critical";
      else if (tierLower === "important") normalizedTier = "important";

      professionSkills.push([jobCategory, skill, normalizedTier]);
    }
  }

  console.log(`Professions found: ${professionSet.size}`);
  console.log(`Unique skills found: ${skillSet.size}`);
  console.log(`Profession-skill mappings: ${professionSkills.length}`);

  // Build professions array
  const professionDescriptions = {
    "Data Engineer":
      "Merancang, membangun, dan memelihara pipeline data serta infrastruktur untuk mengolah data dalam skala besar.",
    "Data Analyst":
      "Menganalisis data untuk menghasilkan insight bisnis melalui visualisasi, pelaporan, dan interpretasi statistik.",
    "Backend Developer":
      "Membangun dan memelihara logika server, API, dan integrasi database yang mendukung aplikasi.",
    "AI / Machine Learning Engineer":
      "Merancang dan mengimplementasikan model machine learning serta sistem AI untuk produksi.",
    "Data Scientist":
      "Menerapkan metode statistik dan machine learning untuk mengekstrak insight dari data kompleks.",
    "Fullstack Developer":
      "Mengembangkan aplikasi end-to-end, mencakup antarmuka pengguna hingga server dan database.",
    "Frontend Developer":
      "Membangun antarmuka pengguna yang interaktif dan responsif menggunakan teknologi web modern.",
  };

  const professionsArr = Array.from(professionSet).map((name) => ({
    name,
    description:
      professionDescriptions[name] ||
      `${name} — profesional yang membangun dan mengoptimalkan sistem di bidang teknologi informasi.`,
  }));

  // Build skills array (sorted alphabetically)
  const skillsArr = Array.from(skillSet.values()).sort((a, b) =>
    a.name.localeCompare(b.name)
  );

  // Generate JS content
  console.log("Generating seeder file...");

  let output = `"use strict";\n\n`;

  // Professions
  output += `const professions = [\n`;
  for (const p of professionsArr) {
    const desc = p.description.replace(/"/g, '\\"');
    output += `  {\n    name: "${p.name}",\n    description: "${desc}",\n  },\n`;
  }
  output += `];\n\n`;

  // Skills
  output += `const skills = [\n`;
  for (const s of skillsArr) {
    const name = s.name.replace(/"/g, '\\"');
    const desc = s.description.replace(/"/g, '\\"');
    output += `  {\n    name: "${name}",\n    description: "${desc}",\n  },\n`;
  }
  output += `];\n\n`;

  // Profession-skills mapping
  output += `const professionSkills = [\n`;
  let currentProf = "";
  for (const [prof, skill, tier] of professionSkills) {
    if (prof !== currentProf) {
      output += `\n  // ${prof}\n`;
      currentProf = prof;
    }
    const profEsc = prof.replace(/"/g, '\\"');
    const skillEsc = skill.replace(/"/g, '\\"');
    output += `  ["${profEsc}", "${skillEsc}", "${tier}"],\n`;
  }
  output += `];\n\n`;

  // Seeder logic (preserved from original)
  output += `/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up(queryInterface, Sequelize) {
    const now = new Date();

    // masukkan professions (skip jika sudah ada)
    for (const prof of professions) {
      const [existing] = await queryInterface.sequelize.query(
        \`SELECT id FROM professions WHERE name = :name LIMIT 1\`,
        {
          replacements: { name: prof.name },
          type: Sequelize.QueryTypes.SELECT,
        },
      );
      if (!existing) {
        await queryInterface.bulkInsert("professions", [
          {
            name: prof.name,
            description: prof.description,
            created_at: now,
            updated_at: now,
          },
        ]);
      }
    }

    // masukkan skills (skip jika sudah ada)
    for (const skill of skills) {
      const [existing] = await queryInterface.sequelize.query(
        \`SELECT id FROM skills WHERE name = :name LIMIT 1\`,
        {
          replacements: { name: skill.name },
          type: Sequelize.QueryTypes.SELECT,
        },
      );
      if (!existing) {
        await queryInterface.bulkInsert("skills", [
          {
            name: skill.name,
            description: skill.description,
            created_at: now,
            updated_at: now,
          },
        ]);
      }
    }

    // masukkan profession_skills
    for (const [profName, skillName, category] of professionSkills) {
      const [profession] = await queryInterface.sequelize.query(
        \`SELECT id FROM professions WHERE name = :name LIMIT 1\`,
        { replacements: { name: profName }, type: Sequelize.QueryTypes.SELECT },
      );
      const [skill] = await queryInterface.sequelize.query(
        \`SELECT id FROM skills WHERE name = :name LIMIT 1\`,
        {
          replacements: { name: skillName },
          type: Sequelize.QueryTypes.SELECT,
        },
      );

      if (!profession || !skill) continue;

      const [existing] = await queryInterface.sequelize.query(
        \`SELECT id FROM profession_skills WHERE id_profession = :pid AND id_skill = :sid LIMIT 1\`,
        {
          replacements: { pid: profession.id, sid: skill.id },
          type: Sequelize.QueryTypes.SELECT,
        },
      );
      if (!existing) {
        await queryInterface.bulkInsert("profession_skills", [
          {
            id_profession: profession.id,
            id_skill: skill.id,
            category,
            created_at: now,
            updated_at: now,
          },
        ]);
      }
    }
  },

  async down(queryInterface, Sequelize) {
    await queryInterface.bulkDelete("profession_skills", null, {});
    await queryInterface.bulkDelete("skills", null, {});
    await queryInterface.bulkDelete("professions", null, {});
  },
};
`;

  fs.writeFileSync(outputPath, output, "utf8");
  console.log(`\\nSeeder berhasil ditulis ke: ${outputPath}`);
  console.log(`Total professions: ${professionsArr.length}`);
  console.log(`Total skills: ${skillsArr.length}`);
  console.log(`Total profession-skill mappings: ${professionSkills.length}`);
}

main().catch(console.error);
