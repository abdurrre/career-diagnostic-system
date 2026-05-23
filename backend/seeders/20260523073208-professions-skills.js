"use strict";

const professions = [
  {
    name: "Data Engineer",
    description:
      "Merancang, membangun, dan memelihara pipeline data serta infrastruktur untuk mengolah data dalam skala besar.",
  },
  {
    name: "Data Analyst",
    description:
      "Menganalisis data untuk menghasilkan insight bisnis melalui visualisasi, pelaporan, dan interpretasi statistik.",
  },
  {
    name: "Backend Developer",
    description:
      "Membangun dan memelihara logika server, API, dan integrasi database yang mendukung aplikasi.",
  },
  {
    name: "AI / Machine Learning Engineer",
    description:
      "Merancang dan mengimplementasikan model machine learning serta sistem AI untuk produksi.",
  },
  {
    name: "Data Scientist",
    description:
      "Menerapkan metode statistik dan machine learning untuk mengekstrak insight dari data kompleks.",
  },
  {
    name: "Fullstack Developer",
    description:
      "Mengembangkan aplikasi end-to-end, mencakup antarmuka pengguna hingga server dan database.",
  },
  {
    name: "Frontend Developer",
    description:
      "Membangun antarmuka pengguna yang interaktif dan responsif menggunakan teknologi web modern.",
  },
];

const skills = [
  {
    name: "sql",
    description:
      "Bahasa query standar untuk mengelola dan mengambil data dari database relasional.",
  },
  {
    name: "python",
    description:
      "Bahasa pemrograman serbaguna yang banyak digunakan dalam data science, AI, dan pengembangan backend.",
  },
  {
    name: "aws",
    description:
      "Platform cloud dari Amazon yang menyediakan layanan infrastruktur, komputasi, dan penyimpanan data.",
  },
  {
    name: "spark",
    description:
      "Framework pemrosesan data terdistribusi berkecepatan tinggi untuk analitik big data dalam skala besar.",
  },
  {
    name: "agile",
    description:
      "Metodologi pengembangan perangkat lunak iteratif yang menekankan kolaborasi tim dan adaptasi cepat.",
  },
  {
    name: "azure",
    description:
      "Platform cloud dari Microsoft untuk membangun, mengelola, dan mendeploy aplikasi dan layanan.",
  },
  {
    name: "etl",
    description:
      "Proses Extract, Transform, Load untuk memindahkan dan mengolah data antar sistem penyimpanan.",
  },
  {
    name: "java",
    description:
      "Bahasa pemrograman berbasis objek yang banyak digunakan untuk aplikasi enterprise dan backend.",
  },
  {
    name: "data modeling",
    description:
      "Proses merancang struktur data untuk merepresentasikan entitas dan hubungannya secara efisien.",
  },
  {
    name: "hadoop",
    description:
      "Framework open-source untuk pemrosesan dan penyimpanan dataset besar secara terdistribusi.",
  },
  {
    name: "tableau",
    description:
      "Platform visualisasi data interaktif untuk membuat dashboard dan laporan analitik.",
  },
  {
    name: "excel",
    description:
      "Aplikasi spreadsheet Microsoft untuk analisis data, pembuatan formula, dan visualisasi dasar.",
  },
  {
    name: "power bi",
    description:
      "Alat business intelligence Microsoft untuk membuat laporan dan dashboard data secara interaktif.",
  },
  {
    name: "machine learning",
    description:
      "Cabang AI yang memungkinkan sistem belajar dari data tanpa pemrograman eksplisit.",
  },
  {
    name: "sql server",
    description:
      "Sistem manajemen database relasional dari Microsoft untuk aplikasi enterprise.",
  },
  {
    name: "git",
    description:
      "Sistem kontrol versi terdistribusi untuk melacak perubahan kode dan kolaborasi tim.",
  },
  {
    name: "kubernetes",
    description:
      "Platform orkestrasi container untuk mengotomatiskan deployment, scaling, dan pengelolaan aplikasi.",
  },
  {
    name: "javascript",
    description:
      "Bahasa pemrograman utama web yang digunakan untuk membangun antarmuka interaktif dan logika aplikasi.",
  },
  {
    name: "docker",
    description:
      "Platform containerisasi untuk mengemas aplikasi beserta dependensinya agar mudah dijalankan di mana saja.",
  },
  {
    name: "c++",
    description:
      "Bahasa pemrograman performa tinggi yang digunakan dalam sistem tertanam, game, dan komputasi ilmiah.",
  },
  {
    name: "tensorflow",
    description:
      "Framework open-source dari Google untuk membangun dan melatih model machine learning dan deep learning.",
  },
  {
    name: "deep learning",
    description:
      "Subfield machine learning menggunakan jaringan saraf berlapis untuk mengenali pola kompleks dalam data.",
  },
  {
    name: "pytorch",
    description:
      "Framework deep learning dari Meta yang populer untuk riset dan pengembangan model AI.",
  },
  {
    name: "ai",
    description:
      "Bidang ilmu komputer yang berfokus pada pembuatan sistem yang mampu meniru kecerdasan manusia.",
  },
  {
    name: "nlp",
    description:
      "Natural Language Processing — cabang AI untuk memahami, memproses, dan menghasilkan bahasa manusia.",
  },
  {
    name: "artificial intelligence",
    description:
      "Teknologi yang memungkinkan mesin melakukan tugas yang biasanya membutuhkan kecerdasan manusia.",
  },
  {
    name: "typescript",
    description:
      "Superset JavaScript dengan sistem tipe statis untuk pengembangan aplikasi skala besar.",
  },
  {
    name: "react",
    description:
      "Library JavaScript dari Meta untuk membangun antarmuka pengguna berbasis komponen.",
  },
  {
    name: "go",
    description:
      "Bahasa pemrograman dari Google yang efisien dan berkecepatan tinggi untuk sistem dan layanan backend.",
  },
  {
    name: "css",
    description:
      "Bahasa stylesheet untuk mengatur tampilan dan layout halaman web.",
  },
  {
    name: "html",
    description:
      "Bahasa markup standar untuk menyusun struktur dan konten halaman web.",
  },
  {
    name: "angular",
    description:
      "Framework frontend berbasis TypeScript dari Google untuk membangun aplikasi web skala enterprise.",
  },
  {
    name: "frontend development",
    description:
      "Praktik pengembangan sisi klien yang mencakup UI, UX, performa, dan aksesibilitas web.",
  },
];

const professionSkills = [
  // Data Engineer
  ["Data Engineer", "sql", "critical"],
  ["Data Engineer", "python", "critical"],
  ["Data Engineer", "aws", "critical"],
  ["Data Engineer", "spark", "important"],
  ["Data Engineer", "agile", "important"],
  ["Data Engineer", "azure", "important"],
  ["Data Engineer", "etl", "supplementary"],
  ["Data Engineer", "java", "supplementary"],
  ["Data Engineer", "data modeling", "supplementary"],
  ["Data Engineer", "hadoop", "supplementary"],

  // Data Analyst
  ["Data Analyst", "sql", "critical"],
  ["Data Analyst", "python", "critical"],
  ["Data Analyst", "tableau", "critical"],
  ["Data Analyst", "excel", "important"],
  ["Data Analyst", "power bi", "important"],
  ["Data Analyst", "machine learning", "important"],
  ["Data Analyst", "data modeling", "supplementary"],
  ["Data Analyst", "etl", "supplementary"],
  ["Data Analyst", "agile", "supplementary"],
  ["Data Analyst", "sql server", "supplementary"],

  // Backend Developer
  ["Backend Developer", "python", "critical"],
  ["Backend Developer", "java", "critical"],
  ["Backend Developer", "aws", "critical"],
  ["Backend Developer", "git", "important"],
  ["Backend Developer", "sql", "important"],
  ["Backend Developer", "kubernetes", "important"],
  ["Backend Developer", "agile", "supplementary"],
  ["Backend Developer", "javascript", "supplementary"],
  ["Backend Developer", "docker", "supplementary"],
  ["Backend Developer", "c++", "supplementary"],

  // AI / Machine Learning Engineer
  ["AI / Machine Learning Engineer", "machine learning", "critical"],
  ["AI / Machine Learning Engineer", "python", "critical"],
  ["AI / Machine Learning Engineer", "tensorflow", "critical"],
  ["AI / Machine Learning Engineer", "deep learning", "important"],
  ["AI / Machine Learning Engineer", "pytorch", "important"],
  ["AI / Machine Learning Engineer", "aws", "important"],
  ["AI / Machine Learning Engineer", "ai", "supplementary"],
  ["AI / Machine Learning Engineer", "agile", "supplementary"],
  ["AI / Machine Learning Engineer", "azure", "supplementary"],
  ["AI / Machine Learning Engineer", "nlp", "supplementary"],

  // Data Scientist
  ["Data Scientist", "python", "critical"],
  ["Data Scientist", "machine learning", "critical"],
  ["Data Scientist", "sql", "critical"],
  ["Data Scientist", "spark", "important"],
  ["Data Scientist", "aws", "important"],
  ["Data Scientist", "tableau", "important"],
  ["Data Scientist", "artificial intelligence", "supplementary"],
  ["Data Scientist", "ai", "supplementary"],
  ["Data Scientist", "tensorflow", "supplementary"],
  ["Data Scientist", "hadoop", "supplementary"],

  // Fullstack Developer
  ["Fullstack Developer", "javascript", "critical"],
  ["Fullstack Developer", "aws", "critical"],
  ["Fullstack Developer", "python", "critical"],
  ["Fullstack Developer", "typescript", "important"],
  ["Fullstack Developer", "java", "important"],
  ["Fullstack Developer", "sql", "important"],
  ["Fullstack Developer", "agile", "supplementary"],
  ["Fullstack Developer", "react", "supplementary"],
  ["Fullstack Developer", "go", "supplementary"],
  ["Fullstack Developer", "docker", "supplementary"],

  // Frontend Developer
  ["Frontend Developer", "javascript", "critical"],
  ["Frontend Developer", "react", "critical"],
  ["Frontend Developer", "css", "critical"],
  ["Frontend Developer", "html", "important"],
  ["Frontend Developer", "typescript", "important"],
  ["Frontend Developer", "git", "important"],
  ["Frontend Developer", "angular", "supplementary"],
  ["Frontend Developer", "agile", "supplementary"],
  ["Frontend Developer", "aws", "supplementary"],
  ["Frontend Developer", "frontend development", "supplementary"],
];

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up(queryInterface, Sequelize) {
    const now = new Date();

    // masukkan professions (skip jika sudah ada)
    for (const prof of professions) {
      const [existing] = await queryInterface.sequelize.query(
        `SELECT id FROM professions WHERE name = :name LIMIT 1`,
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
        `SELECT id FROM skills WHERE name = :name LIMIT 1`,
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
        `SELECT id FROM professions WHERE name = :name LIMIT 1`,
        { replacements: { name: profName }, type: Sequelize.QueryTypes.SELECT },
      );
      const [skill] = await queryInterface.sequelize.query(
        `SELECT id FROM skills WHERE name = :name LIMIT 1`,
        {
          replacements: { name: skillName },
          type: Sequelize.QueryTypes.SELECT,
        },
      );

      if (!profession || !skill) continue;

      const [existing] = await queryInterface.sequelize.query(
        `SELECT id FROM profession_skills WHERE id_profession = :pid AND id_skill = :sid LIMIT 1`,
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
