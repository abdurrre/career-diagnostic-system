const { DataTypes, DATE } = require("sequelize");
const sequelize = require("../config/database");

const Skill = sequelize.define(
  "Skill",
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    name: { type: DataTypes.STRING(100), allowNull: false },
    description: { type: DataTypes.TEXT, allowNull: true },
  },
  {
    tableName: "Skills",
    timestamps: true,
  },
);

module.exports = Skill;