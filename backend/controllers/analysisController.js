const { History, Skill, HistorySkill, Profession } = require("../models/index");
const axios = require("axios");
require("dotenv").config();

exports.scanCV = async (req, res) => {
  try {
    const { id_profession, raw_text_input } = req.body;
    const id_user = req.user.id;

    const profession = await Profession.findByPk(id_profession, {
      include: { model: skill },
    });
    if (!profession)
      return res
        .status(404)
        .json({ message: "Profesi target tidak ditemukan" });

    // Kirim data teks CV dan target profesi ke Service Python (AI)
    // Sesuaikan payload body-nya dengan format yang diminta oleh tim AI
    let aiResponse;
    try {
      aiResponse = await axios.post(process.env.AI_SERVICE_URL, {
        profession_name: profession.name,
        text_input: raw_text_input,
      });
    } catch (aiError) {
      console.error("Error dari AI Service", aiError.message);
      return res.status(502).json({
        message: "Gagal mendapatkan respon dari model AI Engine",
        error: aiError.message,
      });
    }

    const { final_score, skill_analysis } = aiResponse.data;
    /* Contoh struktur 'skill_analysis' yang diharapkan dari Python:
      [
        { name: "Python", status: "match" },
        { name: "Docker", status: "gap" }
      ]
    */

    const newHistory = await History.create({
      score: final_score,
      id_profession,
      id_user,
    });

    const historySkillData = [];

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
      message: "Analisis CV berhasil diproses dan disimpan",
      historyId: newHistory.id,
      score: final_score,
      analysis: skill_analysis,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

exports.getUserHistories = async (req, res) => {
  try {
    const id_user = req.user.id;
    const histories = await History.findAll({
      where: { id_user },
      include: [
        { model: Profession, attributes: ["name"] },
        { model: Skill, through: { attributes: ["status"] } },
      ],
      order: [["created_at"], "DESC"],
    });
    res.json(histories);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
