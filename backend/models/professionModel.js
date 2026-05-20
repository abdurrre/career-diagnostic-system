const { DataTypes } = require("sequelize");
const sequelize = require("../config/database");

const Profession = sequelize.define(
  "Profession",
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    name: { type: DataTypes.VARCHAR(100), allowNull: false },
    description: { type: DataTypes.TEXT, allowNull: true },
  },
  {
    tableName: "Professions",
    timestamps: true,
  },
);

module.exports = Profession;