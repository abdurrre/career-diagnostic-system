const { DataTypes } = require("sequelize");
const sequelize = require("../config/database");

const User = sequelize.define(
  "User",
  {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    email: { type: DataTypes.STRING(255), unique: true, allowNull: false },
    password: { type: DataTypes.STRING(255), allowNull: false },
    status: {
      type: DataTypes.ENUM("inactive", "active"),
      defaultValue: "inactive",
      allowNull: false,
    },
  },
  {
    tableName: "Users",
    timestamps: true,
  },
);

module.exports = User;
