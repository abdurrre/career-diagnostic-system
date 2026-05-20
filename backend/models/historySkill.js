const { DataTypes } = require("sequelize");
const sequelize = require("../config/database");

const HistorySkill = sequelize.define(
  "HistorySkill",
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    id_history: {
      type: DataTypes.INTEGER,
      allowNull: false,
      references: { model: "Histories", key: "id" },
    },
    id_skill: {
      type: DataTypes.INTEGER,
      allowNull: false,
      references: { model: "Skills", key: "id" },
    },
    status: { type: DataTypes.ENUM("match", "gap"), allowNull: false },
  },
  {
    tableName: "History_Skills",
    timestamps: true,
  },
);

module.exports = HistorySkill;
