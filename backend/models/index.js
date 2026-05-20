const sequelize = require("../config/database");

const User = require("./userModel");
const Verification = require("./verificationModel");
const Profession = require("./professionModel");
const Skill = require("./skillModel");
const ProfessionSkill = require("./professionSkill");
const History = require("./historyModel");
const HistorySkill = require("./historySkill");

// users <-> verifications
User.hasMany(Verification, { foreignKey: "id_user", onDelete: "CASCADE" });
Verification.belongsTo(User, { foreignKey: "id_user" });

// users <-> histories
User.hasMany(History, { foreignKey: "id_user", onDelete: "CASCADE" });
History.belongsTo(User, { foreignKey: "id_user" });

// professions <-> histories
Profession.hasMany(History, {
  foreignKey: "id_profession",
  onDelete: "RESTRICT",
});
History.belongsTo(Profession, { foreignKey: "id_profession" });

// professions <-> skills
Profession.belongsToMany(Skill, {
  through: ProfessionSkill,
  foreignKey: "id_profession",
  onDelete: "CASCADE",
});
Skill.belongsToMany(Profession, {
  through: ProfessionSkill,
  foreignKey: "id_skill",
  onDelete: "CASCADE",
});

// histories <-> skills
HistorySkill.belongsToMany(Skill, {
  through: HistorySkill,
  foreignKey: "id_history",
  onDelete: "CASCADE",
});
Skill.belongsToMany(History, {
  through: HistorySkill,
  foreignKey: "id_skill",
  onDelete: "RESTRICT",
});

module.exports = {
  sequelize,
  User,
  Verification,
  Profession,
  Skill,
  ProfessionSkill,
  History,
  HistorySkill,
};
