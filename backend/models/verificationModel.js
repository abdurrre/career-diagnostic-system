const { DataTypes } = require("sequelize");
const sequelize = require("../config/database");

const Verification = sequelize.define(
  "Verification",
  {
    id: {type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true},
    token: {type: DataTypes.STRING(255), allowNull: false},
    expires_at: {type: DataTypes.DATE, allowNull: false},
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
    tableName: "verifications",
    timestamps: true,
    underscored: true
  }
);

module.exports = Verification;