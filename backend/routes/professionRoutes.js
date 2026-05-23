const express = require("express");
const router = express.Router();
const professionController = require("../controllers/professionController");

router.get("/", professionController.getAllProfession);
router.post("/", professionController.createProfession);
router.get("/:id/skills", professionController.getProfessionSkills);

module.exports = router;