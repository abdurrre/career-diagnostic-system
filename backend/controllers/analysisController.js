const { History, Skill, HistorySkill, Profession } = require("../models/index");
const axios = require("axios");
const { CanvasFactory } = require("pdf-parse/worker");
const { PDFParse } = require("pdf-parse");
require("dotenv").config();

// scan cv tanpa login
exports.scanCV = async (req, res) => {
  try {
    const { target_profession, additional_text } = req.body;

    if (!req.file) {
      // response
      return res
        .status(400)
        .json({ message: "File CV berbentuk PDF wajib diupload" });
    }

    const profession = await Profession.findOne({
      where: { name: target_profession },
    });

    if (!profession) {
      // response
      return res
        .status(404)
        .json({ message: `Profesi ${target_profession} tidak ditemukan` });
    }

    // ekstraksi teks pdf
    let pdfText = "";
    try {
      const uint8ArrayData = new Uint8Array(req.file.buffer);
      const pdfData = await new PDFParse(uint8ArrayData, {
        CanvasFactory,
      }).getText();
      pdfText = pdfData.text;
    } catch (pdfError) {
      console.error(pdfError);
      // response
      return res.status(422).json({ message: "Gagal membaca file PDF" });
    }

    const finalRawTextInput = `${pdfText}\n\nKonteks Tambahan\n${additional_text || ""}`;

    // service ai
    let aiResponse;
    try {
      aiResponse = await axios.post(
        process.env.AI_SCAN_SERVICE_URL,
        {
          raw_text: finalRawTextInput,
          target_profession,
        },
        {
          headers: {
            "X-API-Key": process.env.AI_ENGINE_API_KEY,
          },
        },
      );
    } catch (aiError) {
      return res.status(502).json({
        message: "Gagal mendapatkan respon dari AI Engine",
        error: aiError.message,
      });
    }

    // const { score, skill_analysis } = aiResponse.data;
    const { score, skill_analysis } = aiResponse.data;

    // mencari description di database untuk memasukkannya ke dalam response ke frontend
    for (const item of skill_analysis) {
      const skillData = await Skill.findOne({ where: { name: item.name } });
      if (skillData && skillData.description) {
        item.description = skillData.description;
      }
    }

    // response
    res.status(200).json({
      message: "Analisis CV berhasil diproses",
      extracted_text_preview: finalRawTextInput,
      id_profession: profession.id,
      profession_name: profession.name,
      skill_analysis: skill_analysis, // data dari ai service
      score: score, // data dari ai service
      // score_percentage: mockFinalScore,
      // skill_analysis: mockSkillAnalysis,
      // id_profession: profession.id,
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// simpan history
exports.saveHistory = async (req, res) => {
  try {
    const { score, id_profession, profession_name, skill_analysis } = req.body;
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

    for (const skillItem of skill_analysis) {
      const cleanName = skillItem.name.toLowerCase().trim();
      let skillData = await Skill.findOne({ where: { name: cleanName } });

      if (!skillData) {
        skillData = await Skill.create({
          name: cleanName,
          description:
            skillItem.description ||
            "Kesenjangan kemampuan keahlian yang terpetakan berdasarkan kebutuhan standar industri.",
        });
      }

      historySkillData.push({
        id_history: newHistory.id,
        id_skill: skillData.id,
        status: skillItem.status,
        category: skillItem.category,
      });
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

exports.deleteHistory = async (req, res) => {
  try {
    const id_user = req.user.id;
    const { id } = req.params;

    const history = await History.findOne({
      where: { id, id_user },
    });

    if (!history) {
      return res.status(404).json({
        message:
          "Riwayat tidak ditemukan atau Anda tidak memiliki akses untuk menghapus data ini",
      });
    }

    await HistorySkill.destroy({
      where: { id_history: id },
    });

    await history.destroy();

    res.status(200).json({ message: "Riwayat analisis berhasil dihapus" });
  } catch (error) {
    res
      .status(500)
      .json({ message: "Gagal menghapus riwayat", error: error.message });
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
        { model: Skill, through: { attributes: ["status", "category"] } },
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

// chatbot dengan AI Engine
exports.chatWithAI = async (req, res) => {
  try {
    const { message, profession_name, score, skill_analysis } = req.body;

    if (!message) {
      return res.status(400).json({ message: "Pesan tidak boleh kosong" });
    }

    const chatServiceUrl = process.env.AI_CHAT_SERVICE_URL;

    let aiResponse;
    try {
      aiResponse = await axios.post(
        chatServiceUrl,
        {
          message,
          profession_name,
          score,
          skill_analysis,
        },
        {
          headers: {
            "X-API-Key": process.env.AI_ENGINE_API_KEY,
          },
        },
      );
    } catch (aiError) {
      console.error(aiError);

      // Jika AI Engine mengembalikan error spesifik (misal dari slowapi atau validation error)
      if (aiError.response) {
        return res.status(aiError.response.status).json(aiError.response.data);
      }

      return res.status(502).json({
        message: "Gagal mendapatkan respon dari Chatbot AI Engine",
        error: aiError.message,
      });
    }

    res.status(200).json(aiResponse.data);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
