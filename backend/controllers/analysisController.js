const { History, Skill, HistorySkill, Profession } = require("../models/index");
const axios = require("axios");
const { PDFParse } = require("pdf-parse");
require("dotenv").config();

// scan cv tanpa login
exports.scanCV = async (req, res) => {
  try {
    const { profession_name, additional_text } = req.body;

    if (!req.file) {
      return res
        .status(400)
        .json({ message: "File CV berbentuk PDF wajib diupload" });
    }

    const profession = await Profession.findOne({
      where: { name: profession_name },
    });

    if (!profession) {
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

    const mockFinalScore = 85.50;
    const mockSkillAnalysis = [
      { name: "Python", status: "match" },
      { name: "SQL", status: "match" },
      { name: "Data Visualization", status: "match" },
      { name: "Machine Learning Basics", status: "match" },
      { name: "Pandas", status: "match" },
      { name: "Communication", status: "gap" },
      { name: "Project Management", status: "gap" },
      { name: "Agile Methodologies", status: "gap" },
    ];

    res.status(200).json({
      message: "Analisis CV berhasil diproses",
      extracted_text_preview: finalRawTextInput,
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
    const id_user = req.user.id;

    const newHistory = await History.create({
      score: parseFloat(score) || 0.0,
      id_profession,
      id_user,
    });

    const historySkillData = [];

    if (!skill_analysis || !Array.isArray(skill_analysis)) {
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
        });
      }
    }

    if (historySkillData.length > 0) {
      await HistorySkill.bulkCreate(historySkillData);
    }

    res.status(201).json({
      message: "History berhasil disimpan",
      historyId: newHistory.id,
    });
  } catch (error) {
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
    res.json(histories);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
