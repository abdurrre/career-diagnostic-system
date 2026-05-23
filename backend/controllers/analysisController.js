const { History, Skill, HistorySkill, Profession } = require("../models/index");
const axios = require("axios");
const { PDFParse } = require("pdf-parse");
require("dotenv").config();

// scan cv tanpa login
exports.scanCV = async (req, res) => {
  try {
    const { profession_name, additional_text } = req.body;

    if (!req.file) {
      // response
      return res
        .status(400)
        .json({ message: "File CV berbentuk PDF wajib diupload" });
    }

    const profession = await Profession.findOne({
      where: { name: profession_name },
    });

    if (!profession) {
      // response
      return res
        .status(404)
        .json({ message: `Profesi ${profession_name} tidak ditemukan` });
    }

    // ekstraksi teks pdf
    let pdfText = "";
    try {
      const uint8ArrayData = new Uint8Array(req.file.buffer);
      const pdfData = await new PDFParse(uint8ArrayData).getText();
      pdfText = pdfData.text;
    } catch (pdfError) {
      console.error(pdfError);
      // response
      return res.status(422).json({ message: "Gagal membaca file PDF" });
    }

    const finalRawTextInput = `${pdfText}\n\nKonteks Tambahan\n${additional_text || ""}`;

    // ============ MENUNGGU SERVICE AI ============
    // let aiResponse;
    // try {
    //   aiResponse = await axios.post(process.env.AI_SERVICE_URL, {
    //     profession_name,
    //     text_input: finalRawTextInput,
    //   });
    // } catch (aiError) {
    //   return res.status(502).json({
    //     message: "Gagal mendapatkan respon dari AI Engine",
    //     error: aiError.message,
    //   });
    // }

    // const { final_score, skill_analysis } = aiResponse.data;

    const mockFinalScore = 85.5;
    const mockSkillAnalysis = [
      // critical
      {
        name: "machine learning",
        status: "match",
        category: "critical",
        description:
          "Cabang AI yang memungkinkan sistem belajar dari data tanpa pemrograman eksplisit.",
      },
      {
        name: "python",
        status: "match",
        category: "critical",
        description:
          "Bahasa pemrograman serbaguna yang banyak digunakan dalam data science, AI, dan pengembangan backend.",
      },
      {
        name: "tensorflow",
        status: "gap",
        category: "critical",
        description:
          "Framework open-source dari Google untuk membangun dan melatih model machine learning dan deep learning.",
      },

      // important
      {
        name: "deep learning",
        status: "gap",
        category: "important",
        description:
          "Subfield machine learning menggunakan jaringan saraf berlapis untuk mengenali pola kompleks dalam data.",
      },
      {
        name: "pytorch",
        status: "gap",
        category: "important",
        description:
          "Framework deep learning dari Meta yang populer untuk riset dan pengembangan model AI.",
      },
      {
        name: "aws",
        status: "match",
        category: "important",
        description:
          "Platform cloud dari Amazon yang menyediakan layanan infrastruktur, komputasi, dan penyimpanan data.",
      },

      // supplementary
      {
        name: "ai",
        status: "match",
        category: "supplementary",
        description:
          "Bidang ilmu komputer yang berfokus pada pembuatan sistem yang mampu meniru kecerdasan manusia.",
      },
      {
        name: "agile",
        status: "gap",
        category: "supplementary",
        description:
          "Metodologi pengembangan perangkat lunak iteratif yang menekankan kolaborasi tim dan adaptasi cepat.",
      },
      {
        name: "azure",
        status: "gap",
        category: "supplementary",
        description:
          "Platform cloud dari Microsoft untuk membangun, mengelola, dan mendeploy aplikasi dan layanan.",
      },
      {
        name: "nlp",
        status: "gap",
        category: "supplementary",
        description:
          "Natural Language Processing — cabang AI untuk memahami, memproses, dan menghasilkan bahasa manusia.",
      },
    ];

    // response
    res.status(200).json({
      message: "Analisis CV berhasil diproses",
      extracted_text_preview: finalRawTextInput,
      profession_name: profession.name,
      score: mockFinalScore,
      // score: final_score, -> tunggu service ai
      analysis: mockSkillAnalysis,
      // analysis: skill_analysis, -> tunggu service ai
      id_profession: profession.id,
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// simpan history
exports.saveHistory = async (req, res) => {
  try {
    const { score, id_profession, skill_analysis } = req.body;
    // contoh isi dari skill_analysis
    // {
    //   "score": 85.50,
    //   "id_profession": 1,
    //   "skill_analysis": [
    //     { "name": "python", "status": "match", "category": "critical" },
    //     { "name": "sql", "status": "match", "category": "important" },
    //     { "name": "communication", "status": "gap", "category": "supplementary" }
    //   ]
    // }
    const id_user = req.user.id;

    const newHistory = await History.create({
      score: parseFloat(score) || 0.0,
      id_profession,
      id_user,
    });

    const historySkillData = [];

    if (!skill_analysis || !Array.isArray(skill_analysis)) {
      // response
      return res
        .status(400)
        .json({ message: "Data analisis skill tidak valid" });
    }

    for (const item of skill_analysis) {
      const skillData = await Skill.findOne({ where: { name: item.name } });

      if (skillData) {
        historySkillData.push({
          id_history: newHistory.id,
          id_skill: skillData.id,
          status: item.status,
          category: item.category,
        });
      }
    }

    if (historySkillData.length > 0) {
      await HistorySkill.bulkCreate(historySkillData);
    }

    // response
    res.status(201).json({
      message: "History berhasil disimpan",
      historyId: newHistory.id,
    });
  } catch (error) {
    // response
    res.status(500).json({ error: error.message });
  }
};

// ambil history user
exports.getUserHistories = async (req, res) => {
  try {
    const id_user = req.user.id;
    const histories = await History.findAll({
      where: { id_user },
      include: [
        { model: Profession, attributes: ["name"] },
        { model: Skill, through: { attributes: ["status"] } },
      ],
      order: [["created_at", "DESC"]],
    });
    // response
    res.json(histories);
  } catch (error) {
    // response
    res.status(500).json({ error: error.message });
  }
};
