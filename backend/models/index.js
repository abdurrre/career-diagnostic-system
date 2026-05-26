const sequelize = require("../config/database");

const User = require("./userModel");
const Profession = require("./professionModel");
const Skill = require("./skillModel");
const History = require("./historyModel");
const ProfessionSkill = require("./professionSkill");
const HistorySkill = require("./historySkill");

// Users <-> Histories
User.hasMany(History, { foreignKey: "id_user", onDelete: "CASCADE" });
History.belongsTo(User, { foreignKey: "id_user" });

// Professions <-> Histories
Profession.hasMany(History, { foreignKey: "id_profession", onDelete: "RESTRICT" });
History.belongsTo(Profession, { foreignKey: "id_profession" });

// Professions <-> Skills (Many-to-Many lewat ProfessionSkill)
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

// Histories <-> Skills (Many-to-Many lewat HistorySkill)
History.belongsToMany(Skill, {
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
  Profession,
  Skill,
  ProfessionSkill,
  History,
  HistorySkill,
};