const { DataTypes } = require("sequelize");
const sequelize = require("../config/database");

const History = sequelize.define(
  "History",
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    score: { type: DataTypes.DECIMAL(5, 2), allowNull: false },
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
    tableName: "histories",
    timestamps: false
  },
);

module.exports = History;
