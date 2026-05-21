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
});

router.post(
  "/scan",
  upload.single("cv_file"),
  analysisController.scanCV,
);

router.post("/save", authMiddleware, analysisController.saveHistory);

router.get("/history", authMiddleware, analysisController.getUserHistories);

module.exports = router;
