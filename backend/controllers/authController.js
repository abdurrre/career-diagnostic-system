const { User } = require("../models/index");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const nodemailer = require("nodemailer");

exports.register = async (req, res) => {
  try {
    const { email, password } = req.body;

    // Cek apakah email sudah terdaftar
    const existingUser = await User.findOne({ where: { email } });
    if (existingUser) {
      return res.status(400).json({
        message: "Email sudah terdaftar. Silakan gunakan email lain.",
      });
    }

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    const newUser = await User.create({
      email,
      password: hashedPassword,
      status: "active",
    });

    res
      .status(201)
      .json({ message: "User berdasarkan didaftarkan", userId: newUser.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

exports.login = async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ where: { email } });
    if (!user) return res.status(404).json({ message: "User tidak ditemukan" });

    const validPass = await bcrypt.compare(password, user.password);
    if (!validPass) return res.status(400).json({ message: "Password salah" });

    const token = jwt.sign({ id: user.id }, process.env.JWT_SECRET, {
      expiresIn: "1d",
    });

    res.json({ message: "Login berhasil", token });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

exports.forgotPassword = async (req, res) => {
  try {
    const { email } = req.body;

    const user = await User.findOne({ where: { email } });
    if (!user) {
      return res.status(404).json({ message: "Alamat email tidak terdaftar." });
    }

    const secret = process.env.JWT_SECRET + user.password;

    const token = jwt.sign({ id: user.id, email: user.email }, secret, {
      expiresIn: "15m",
    });

    const recoveryUrl = `${process.env.FRONTEND_URL}/?view=reset-password-new&token=${token}&email=${user.email}`;

    const transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port: parseInt(process.env.SMTP_PORT),
      secure: false,
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS,
      },
    });

    const mailOptions = {
      from: `"SKillPath AI Support" <${process.env.SMTP_USER}>`,
      to: user.email,
      subject: "[SkillPath AI] Atur Ulang Kata Sandi Anda",
      html: `
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; margin: 0 auto; padding: 30px; background-color: #f9fafb; border: 1px solid #f3f4f6; border-radius: 16px;">
          <h2 style="color: #4f4bf5; font-size: 24px; font-weight: bold; margin-bottom: 20px;">Atur Ulang Kata Sandi Anda</h2>
          <p style="color: #4b5563; font-size: 14px; line-height: 1.6;">Halo,</p>
          <p style="color: #4b5563; font-size: 14px; line-height: 1.6;">Kami menerima permintaan untuk mengatur ulang kata sandi akun SkillPath AI Anda. Silakan klik tombol di bawah ini untuk melanjutkan:</p>
          
          <div style="margin: 30px 0; text-align: center;">
            <a href="${recoveryUrl}" style="background-color: #4f4bf5; color: white; padding: 12px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 14px; display: inline-block; box-shadow: 0 4px 6px -1px rgba(79, 75, 245, 0.2);">
              Atur Ulang Kata Sandi
            </a>
          </div>
          
          <p style="color: #9ca3af; font-size: 11px; line-height: 1.6;">*Tautan ini hanya berlaku selama 15 menit. Jika Anda tidak merasa meminta perubahan ini, abaikan saja email ini.</p>
          <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 20px 0;" />
          <p style="color: #9ca3af; font-size: 11px; text-align: center;">SkillPath AI Dashboard © 2026</p>
        </div>
      `,
    };

    await transporter.sendMail(mailOptions);

    res.json({
      message: "Link pemulihan kata sandi telah dikirim ke email Anda.",
      recoveryUrl,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

exports.resetPassword = async (req, res) => {
  try {
    const { email, token, newPassword } = req.body;

    const user = await User.findOne({ where: { email } });
    if (!user) {
      return res.status(404).json({ message: "User tidak ditemukan." });
    }

    const secret = process.env.JWT_SECRET + user.password;
    try {
      jwt.verify(token, secret);
    } catch (error) {
      return res
        .status(400)
        .json({ message: "Link tidak valid atau sudah kedaluarsa." });
    }

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(newPassword, salt);

    user.password = hashedPassword;
    await user.save();

    res.json({ message: "Kata sandi Anda berhasil diperbarui" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
