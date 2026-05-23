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
    category: {
      type: DataTypes.ENUM("critical", "important", "supplementary"),
      allowNull: true
    },
    created_at: {
      type: DataTypes.DATE,
      allowNull: false,
      defaultValue: DataTypes.NOW,
    },
    updated_at: {
      type: DataTypes.DATE,
      allowNull: false,
      defaultValue: DataTypes.NOW, 
    },
  },
  {
    tableName: "history_skills",
    timestamps: true,
    underscored: true
  },
);

module.exports = HistorySkill;
