const { DataTypes } = require("sequelize");
const sequelize = require("../config/database");

const Verification = sequelize.define(
  "Verification",
  {
    id: {type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true},
    token: {type: DataTypes.VARCHAR(255), allowNull: false},
    expires_at: {type: DataTypes.DATE, allowNull: false},
  },
  {
    tableName: "Verifications",
    timestamps: true,
  }
);

module.exports = Verification;