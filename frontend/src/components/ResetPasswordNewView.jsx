import { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, User, Lock, AlertCircle, Loader2, ArrowRight, ArrowLeft, Eye, EyeOff } from 'lucide-react';
import authVisualImg from '../assets/auth_visual.png';

export default function ResetPasswordNewView({
  currentView,
  setCurrentView,
  handleTryAnother
}) {
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  if (currentView !== 'reset-password-new') {
    return null;
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');

    if (!newPassword) {
      setError("Kata sandi baru wajib diisi.");
      return;
    }
    if (newPassword.length < 6) {
      setError("Kata sandi minimal harus 6 karakter.");
      return;
    }
    if (newPassword !== confirmPassword) {
      setError("Kata sandi tidak cocok.");
      return;
    }

    setLoading(true);
    // Simulate updating password
    setTimeout(() => {
      setLoading(false);
      setCurrentView('reset-password-success');
    }, 1200);
  };

  return (
    <motion.main
      key="reset-password-new"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.35 }}
      className="flex-grow w-full flex items-center justify-center px-4 py-8 sm:py-16 md:py-24 bg-[#f4f6fc]"
    >
      <div className="w-full max-w-[1000px] bg-white rounded-3xl overflow-hidden shadow-[0_15px_45px_rgba(0,0,0,0.04)] border border-slate-100/60 flex flex-col md:flex-row min-h-[580px] md:min-h-[620px]">
        
        {/* LEFT PANEL: Visual quote */}
        <div 
          style={{ backgroundImage: `url(${authVisualImg})` }}
          className="hidden md:flex md:w-1/2 bg-slate-900 bg-cover bg-center p-10 flex-col justify-between relative text-white border-r border-slate-100/10"
        >
          <div onClick={handleTryAnother} className="flex items-center gap-2.5 cursor-pointer z-10">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-500 to-brand-400 flex items-center justify-center shadow-lg shadow-brand-500/25">
              <Sparkles className="w-4.5 h-4.5 text-white" />
            </div>
            <span className="text-lg font-bold font-outfit text-white">SkillPath AI</span>
          </div>

          <div className="absolute inset-0 bg-slate-950/20 backdrop-blur-[1px] pointer-events-none" />

          {/* Quote */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white/10 backdrop-blur-md border border-white/15 rounded-2xl p-6 z-10 shadow-lg space-y-4"
          >
            <p className="text-sm font-sans font-medium text-slate-100 leading-relaxed">
              &ldquo;Re-create yourself. Your potential is not fixed by your past, but shaped by your daily discipline.&rdquo;
            </p>
            <div className="flex items-center gap-3 pt-1">
              <div className="w-9 h-9 rounded-full bg-brand-500/20 text-brand-300 flex items-center justify-center flex-shrink-0">
                <User className="w-4.5 h-4.5" />
              </div>
              <div>
                <h5 className="font-bold font-outfit text-xs !text-white">Robert Greene</h5>
                <p className="text-[10px] !text-slate-300 font-sans mt-0.5">Author of The 50th Law</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* RIGHT PANEL: Recovery Inputs */}
        <div className="w-full md:w-1/2 p-8 sm:p-12 md:p-14 flex flex-col justify-between space-y-8 bg-white">
          
          <div className="flex md:hidden items-center gap-2 cursor-pointer pb-2 border-b border-slate-50" onClick={handleTryAnother}>
            <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <span className="text-base font-bold font-outfit bg-gradient-to-r from-brand-900 to-brand-700 bg-clip-text text-transparent">
              SkillPath AI
            </span>
          </div>

          {/* Titles */}
          <div className="space-y-2">
            <h2 className="text-2xl sm:text-3xl font-bold font-outfit text-slate-900 tracking-tight">
              Reset Kata Sandi
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 font-sans leading-relaxed">
              Masukkan kata sandi baru Anda di bawah untuk memperbarui kredensial masuk Anda.
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* New Password */}
            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-700 font-outfit tracking-wide block">
                Kata Sandi Baru
              </label>
              <div className="relative">
                <input
                  type={showNewPassword ? "text" : "password"}
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="••••••••"
                  className="premium-input pl-11 pr-11 text-sm font-sans"
                  required
                />
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">
                  <Lock className="w-4.5 h-4.5" />
                </div>
                <button
                  type="button"
                  onClick={() => setShowNewPassword(!showNewPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors focus:outline-none flex items-center justify-center cursor-pointer"
                  aria-label={showNewPassword ? "Sembunyikan kata sandi baru" : "Tampilkan kata sandi baru"}
                >
                  {showNewPassword ? (
                    <EyeOff className="w-4.5 h-4.5" />
                  ) : (
                    <Eye className="w-4.5 h-4.5" />
                  )}
                </button>
              </div>
            </div>

            {/* Confirm New Password */}
            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-700 font-outfit tracking-wide block">
                Konfirmasi Kata Sandi Baru
              </label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? "text" : "password"}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="••••••••"
                  className="premium-input pl-11 pr-11 text-sm font-sans"
                  required
                />
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">
                  <Lock className="w-4.5 h-4.5" />
                </div>
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors focus:outline-none flex items-center justify-center cursor-pointer"
                  aria-label={showConfirmPassword ? "Sembunyikan konfirmasi kata sandi" : "Tampilkan konfirmasi kata sandi"}
                >
                  {showConfirmPassword ? (
                    <EyeOff className="w-4.5 h-4.5" />
                  ) : (
                    <Eye className="w-4.5 h-4.5" />
                  )}
                </button>
              </div>
            </div>

            {error && (
              <motion.div 
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-rose-50 border border-rose-100 text-rose-600 rounded-xl px-4 py-2.5 text-xs font-medium flex items-center gap-2"
              >
                <AlertCircle className="w-4 h-4" />
                <span>{error}</span>
              </motion.div>
            )}

            {/* Submit */}
            <div className="pt-2">
              <button
                type="submit"
                disabled={loading}
                className={`w-full py-3.5 rounded-xl font-bold font-outfit text-white text-sm shadow-md flex items-center justify-center gap-2 transition-all active:scale-[0.98] ${
                  loading 
                    ? 'bg-brand-500 cursor-not-allowed shadow-inner' 
                    : 'bg-brand-600 hover:bg-brand-700 hover:shadow-lg hover:shadow-brand-600/15'
                }`}
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4.5 h-4.5 animate-spin" />
                    <span>Menyimpan Kata Sandi...</span>
                  </>
                ) : (
                  <>
                    <span>Simpan Kata Sandi Baru</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Footer Back */}
          <div className="text-center space-y-4 border-t border-slate-50 pt-6">
            <div>
              <button
                onClick={handleTryAnother}
                className="inline-flex items-center gap-1 text-xs text-slate-400 hover:text-slate-600 hover:underline transition-all"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                Batal & Kembali ke Beranda
              </button>
            </div>
          </div>

        </div>

      </div>
    </motion.main>
  );
}
