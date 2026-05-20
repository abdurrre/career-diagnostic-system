const { DataTypes } = require("sequelize");
const sequelize = require("../config/database");

const HistorySkill = sequelize.define(
  "HistorySkill",
  {
    id: {type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true},
    status: {type: DataTypes.ENUM("match", "gap"), allowNull: false}
  },
  {
    tableName: "History_Skills",
    timestamps: true
  }
);