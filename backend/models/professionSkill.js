const { DataTypes } = require("sequelize");
const sequelize = require("../config/database");

const ProfessionSkill = sequelize.define(
  "ProfessionSkill",
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
  },
  {
    tableName: "Profession_Skills",
    timestamps: true,
  },
);

module.exports = ProfessionSkill;