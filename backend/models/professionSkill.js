const { DataTypes } = require("sequelize");
const sequelize = require("../config/database");

const ProfessionSkill = sequelize.define(
  "ProfessionSkill",
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
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
    underscored: true
  },
);

module.exports = ProfessionSkill;