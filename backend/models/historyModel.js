const { DataTypes } = require("sequelize");
const sequelize = require("../config/database");

const History = sequelize.define(
  "History",
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    score: { type: DataTypes.DECIMAL(5, 2), allowNull: false },
  },
  {
    tableName: "Histories",
    timestamps: true,
  },
);

module.exports = History;