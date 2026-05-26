const express = require("express");
const router = express.Router();
const analysisController = require("../controllers/analysisController");
const authMiddleware = require("../middleware/authMiddleware");

const multer = require("multer");
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === "application/pdf") {
      cb(null, true);
    } else {
      cb(new Error("Hanya file PDF yang diperbolehkan!"), false);
    }
  },
}).single("cv_file");

const uploadMiddleware = (req, res, next) => {
  upload(req, res, (err) => {
    if (err) {
      return res.status(400).json({
        message: "Gagal mengupload file",
        error: err.message,
      });
    }
    next();
  });
};

router.post("/scan", uploadMiddleware, analysisController.scanCV);

router.post("/save", authMiddleware, analysisController.saveHistory);

router.get("/history", authMiddleware, analysisController.getUserHistories);

router.post("/chat", analysisController.chatWithAI);

module.exports = router;
