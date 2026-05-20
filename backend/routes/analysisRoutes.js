const express = require("express");
const router = express.Router();
const analysisController = require("../controllers/analysisController");
const authMiddleware = require("../middleware/authMiddleware");

router.post("/scan", authMiddleware, analysisController.scanCV);
router.get("/history", authMiddleware, analysisController.getUserHistories);
