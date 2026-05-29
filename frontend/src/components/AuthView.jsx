import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Sparkles, 
  User, 
  Mail, 
  Lock, 
  AlertCircle, 
  Loader2, 
  ArrowRight, 
  ArrowLeft,
  CheckCircle2,
  Eye,
  EyeOff
} from 'lucide-react';
import authVisualImg from '../assets/auth_visual.png';

export default function AuthView({
  currentView,
  setCurrentView,
  authEmail,
  setAuthEmail,
  authPassword,
  setAuthPassword,
  authConfirmPassword,
  setAuthConfirmPassword,
  authError,
  authLoading,
  registerSuccess,
  handleLoginSubmit,
  handleRegisterSubmit,
  handleTryAnother,
  triggerLoginView,
  triggerRegisterView
}) {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // Reset visibility states when switching views (e.g. login <-> register)
  useEffect(() => {
    setShowPassword(false);
    setShowConfirmPassword(false);
  }, [currentView]);

  if (currentView !== 'login' && currentView !== 'register') {
    return null;
  }

  return (
    <motion.main
      key="auth-view"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.35 }}
      className="flex-grow w-full flex items-center justify-center px-4 py-8 sm:py-16 md:py-24 bg-[#f4f6fc]"
    >
      {/* The Unified Auth Split Container Card */}
      <div className="w-full max-w-[1000px] bg-white rounded-3xl overflow-hidden shadow-[0_15px_45px_rgba(0,0,0,0.04)] border border-slate-100/60 flex flex-col md:flex-row min-h-[580px] md:min-h-[620px]">
        
        {/* LEFT SPLIT PANEL: Visual Artwork (Logo, Abstract Background & Quotes) */}
        <div 
          style={{ backgroundImage: `url(${authVisualImg})` }}
          className="hidden md:flex md:w-1/2 bg-slate-900 bg-cover bg-center p-10 flex-col justify-between relative text-white border-r border-slate-100/10"
        >
          {/* Top Left: Logo */}
          <div 
            onClick={handleTryAnother}
            className="flex items-center gap-2.5 cursor-pointer z-10"
          >
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-500 to-brand-400 flex items-center justify-center shadow-lg shadow-brand-500/25">
              <Sparkles className="w-4.5 h-4.5 text-white" />
            </div>
            <span className="text-lg font-bold font-outfit text-white">
              SkillPath AI
            </span>
          </div>

          {/* Ambient graphic glow backdrops */}
          <div className="absolute inset-0 bg-slate-950/20 backdrop-blur-[1px] pointer-events-none" />

          {/* Bottom Overlay Quote Panel */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white/10 backdrop-blur-md border border-white/15 rounded-2xl p-6 z-10 shadow-lg space-y-4"
          >
            <p className="text-sm font-sans font-medium text-slate-100 leading-relaxed">
              &ldquo;The future belongs to those who learn more skills and combine them in creative ways.&rdquo;
            </p>
            
            <div className="flex items-center gap-3 pt-1">
              <div className="w-9 h-9 rounded-full bg-brand-500/20 text-brand-300 flex items-center justify-center flex-shrink-0">
                <User className="w-4.5 h-4.5" />
              </div>
              <div>
                <h5 className="font-bold font-outfit text-xs !text-white">Robert Greene</h5>
                <p className="text-[10px] !text-slate-300 font-sans mt-0.5">American Author and Strategist</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* RIGHT SPLIT PANEL: Dynamic Form Fields (Login vs. Register) */}
        <div className="w-full md:w-1/2 p-8 sm:p-12 md:p-14 flex flex-col justify-between space-y-8 bg-white">
          
          {/* Logo top anchor for mobile responsive view */}
          <div className="flex md:hidden items-center gap-2 cursor-pointer pb-2 border-b border-slate-50" onClick={handleTryAnother}>
            <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <span className="text-base font-bold font-outfit bg-gradient-to-r from-brand-900 to-brand-700 bg-clip-text text-transparent">
              SkillPath AI
            </span>
          </div>

          {/* 1. Welcomes Title Block */}
          <div className="space-y-2">
            <h2 className="text-2xl sm:text-3xl font-bold font-outfit text-slate-900 tracking-tight">
              {currentView === 'login' ? 'Selamat Datang' : 'Buat Akun'}
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 font-sans leading-relaxed">
              {currentView === 'login' 
                ? 'Masuk untuk mengakses peta jalan keahlian khusus Anda.' 
                : 'Daftar untuk memulai diagnosis dan memetakan perkembangan keahlian Anda.'}
            </p>
          </div>

          {/* 2. Interactive Inputs Form */}
          <form 
            onSubmit={currentView === 'login' ? handleLoginSubmit : handleRegisterSubmit}
            className="space-y-5"
          >
            {/* Email Address Input */}
            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-700 font-outfit tracking-wide block">
                Alamat Email
              </label>
              <div className="relative">
                <input
                  type="email"
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  placeholder="anda@perusahaan.com"
                  className="premium-input pl-11 text-sm font-sans"
                  required
                />
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">
                  <Mail className="w-4.5 h-4.5" />
                </div>
              </div>
            </div>

            {/* Password Input */}
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <label className="text-xs font-bold text-slate-700 font-outfit tracking-wide block">
                  Kata Sandi
                </label>
                {currentView === 'login' && (
                  <button 
                    type="button"
                    onClick={() => setCurrentView('forgot-password')}
                    className="text-xs font-semibold text-brand-600 hover:text-brand-700 hover:underline cursor-pointer"
                  >
                    Lupa kata sandi?
                  </button>
                )}
              </div>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  placeholder="••••••••"
                  className="premium-input pl-11 pr-11 text-sm font-sans"
                  required
                />
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">
                  <Lock className="w-4.5 h-4.5" />
                </div>
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors focus:outline-none flex items-center justify-center cursor-pointer"
                  aria-label={showPassword ? "Sembunyikan kata sandi" : "Tampilkan kata sandi"}
                >
                  {showPassword ? (
                    <EyeOff className="w-4.5 h-4.5" />
                  ) : (
                    <Eye className="w-4.5 h-4.5" />
                  )}
                </button>
              </div>
            </div>

            {/* Confirm Password Input (Register Only) */}
            {currentView === 'register' && (
              <motion.div 
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="space-y-2"
              >
                <label className="text-xs font-bold text-slate-700 font-outfit tracking-wide block">
                  Konfirmasi Kata Sandi
                </label>
                <div className="relative">
                  <input
                    type={showConfirmPassword ? "text" : "password"}
                    value={authConfirmPassword}
                    onChange={(e) => setAuthConfirmPassword(e.target.value)}
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
              </motion.div>
            )}

            {/* Dynamic Auth Error alerts banner */}
            {authError && (
              <motion.div 
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-rose-50 border border-rose-100 text-rose-600 rounded-xl px-4 py-2.5 text-xs font-medium flex items-center gap-2"
              >
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{authError}</span>
              </motion.div>
            )}

            {/* Primary CTA Buttons */}
            <div className="pt-2 space-y-4">
              <button
                type="submit"
                disabled={authLoading}
                className={`w-full py-3.5 rounded-xl font-bold font-outfit text-white text-sm shadow-md flex items-center justify-center gap-2 transition-all active:scale-[0.98] ${
                  authLoading 
                    ? 'bg-brand-500 cursor-not-allowed shadow-inner' 
                    : 'bg-brand-600 hover:bg-brand-700 hover:shadow-lg hover:shadow-brand-600/15'
                }`}
              >
                {authLoading ? (
                  <>
                    <Loader2 className="w-4.5 h-4.5 animate-spin" />
                    <span>Memproses...</span>
                  </>
                ) : currentView === 'login' ? (
                  <>
                    <span>Masuk</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                ) : (
                  <span>Buat Akun</span>
                )}
              </button>

              {/* Dynamic Successful Registration Notification (Register Only) */}
              <AnimatePresence>
                {currentView === 'register' && registerSuccess && (
                  <motion.div 
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    className="bg-emerald-50 border border-emerald-100 text-emerald-800 rounded-xl p-4 text-left flex items-start gap-3 shadow-sm mt-3"
                  >
                    <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
                    <div className="space-y-1 w-full">
                      <h5 className="font-bold font-outfit text-emerald-950 text-xs sm:text-sm">Registrasi Berhasil!</h5>
                      <p className="text-emerald-700 leading-relaxed font-sans text-xs">
                        Akun Anda berhasil terdaftar. Silakan masuk (Sign In) menggunakan email dan kata sandi Anda.
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </form>

          {/* 3. Footer Links Switching */}
          <div className="text-center space-y-4 border-t border-slate-50 pt-6">
            {currentView === 'login' ? (
              <p className="text-xs sm:text-sm text-slate-500 font-sans">
                Belum punya akun?{' '}
                <button 
                  onClick={triggerRegisterView}
                  className="font-bold text-brand-600 hover:text-brand-700 hover:underline transition-colors"
                >
                  Buat Akun
                </button>
              </p>
            ) : (
              <p className="text-xs sm:text-sm text-slate-500 font-sans">
                Sudah punya akun?{' '}
                <button 
                  onClick={triggerLoginView}
                  className="font-bold text-brand-600 hover:text-brand-700 hover:underline transition-colors"
                >
                  Masuk
                </button>
              </p>
            )}

            {/* Return back home toggle link */}
            <div>
              <button
                onClick={handleTryAnother}
                className="inline-flex items-center gap-1 text-xs text-slate-400 hover:text-slate-600 hover:underline transition-all"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                Kembali ke Beranda
              </button>
            </div>
          </div>

        </div>

      </div>
    </motion.main>
  );
}
