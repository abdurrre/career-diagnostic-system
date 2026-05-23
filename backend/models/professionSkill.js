const { DataTypes } = require("sequelize");
const sequelize = require("../config/database");

const ProfessionSkill = sequelize.define(
  "ProfessionSkill",
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    id_profession: { type: DataTypes.INTEGER, allowNull: false },
    id_skill: { type: DataTypes.INTEGER, allowNull: false },
    category: {
      type: DataTypes.ENUM("critical", "important", "supplementary"),
      allowNull: false,
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
    tableName: "profession_skills",
    timestamps: true,
    underscored: true,
  },
);

module.exports = ProfessionSkill;
