const { Profession, Skill } = require("../models/index");

exports.getAllProfession = async (req, res) => {
  try {
    const professions = await Profession.findAll();
    res.json(professions);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

exports.getProfessionSkills = async (req, res) => {
  try {
    const { id } = req.params;
    const profession = await Profession.findByPk(id, {
      include: { model: Skill, through: { attributes: [] } },
    });

    if (!profession)
      return res.status(404).json({ message: "Profesi tidak ditemukan" });
    res.json(profession);
  } catch (error) {
    res.status(500).json({error: error.message});
  }
};

exports.createProfession = async (req, res) => {
  try {
    const { name, description } = req.body;
    if (!name) return res.status(400).json({ message: "Nama profesi wajib diisi" });

    const newProfession = await Profession.create({ name, description });
    res.status(201).json(newProfession);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
