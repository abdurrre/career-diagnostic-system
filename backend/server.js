const express = require("express");
const cors = require("cors");
const { sequelize } = require("./models/index");
const authRoutes = require("./routes/authRoutes");
const professionRoutes = require("./routes/professionRoutes");
const analysisRoutes = require("./routes/analysisRoutes");

require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 5000;

// middleware global
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// routing
app.use("/api/auth", authRoutes);
app.use("/api/professions", professionRoutes);
app.use("/api/analysis", analysisRoutes);

// fallback route
app.use((req, res) => {
  res.status(404).json({ message: "Endpoint tidak ditemukan" });
});

// koneksi database
sequelize
  .sync()
  .then(() => {
    console.log("Database terhubung dan tabel berhasil disinkronkan");
    app.listen(PORT, () => {
      console.log(`Server berjalan pada http://localhost:${PORT}`);
    });
  })
  .catch((err) => {
    console.log("Gagal terhubung ke database: ", err);
  });
