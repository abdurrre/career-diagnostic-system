import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, User, Mail, AlertCircle, Loader2, ArrowRight, ArrowLeft, CheckCircle2 } from 'lucide-react';
import authVisualImg from '../assets/auth_visual.png';
import { API_BASE_URL } from '../config/api';

export default function ForgotPasswordView({
  currentView,
  setCurrentView,
  handleTryAnother,
  triggerLoginView
}) {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  if (currentView !== 'forgot-password') {
    return null;
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');
    setSuccess(false);

    if (!email) {
      setError("Alamat email wajib diisi.");
      return;
    }

    setLoading(true);

    fetch(`${API_BASE_URL}/auth/forgot-password`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ email })
    })
    .then(async (res) => {
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.message || 'Gagal memproses permintaan lupa kata sandi.');
      }
      setLoading(false);
      setSuccess(true);
      if (data.recoveryUrl) {
        localStorage.setItem('mock_recovery_url', data.recoveryUrl);
      }
    })
    .catch(err => {
      setLoading(false);
      setError(err.message || 'Terjadi kesalahan jaringan.');
    });
  };

  return (
    <motion.main
      key="forgot-password"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.35 }}
      className="flex-grow w-full flex items-center justify-center px-4 py-8 sm:py-16 md:py-24 bg-[#f4f6fc]"
    >
      <div className="w-full max-w-[1000px] bg-white rounded-3xl overflow-hidden shadow-[0_15px_45px_rgba(0,0,0,0.04)] border border-slate-100/60 flex flex-col md:flex-row min-h-[580px] md:min-h-[620px]">
        
        {/* LEFT PANEL: Abstract background & Quote */}
        <div 
          style={{ backgroundImage: `url(${authVisualImg})` }}
          className="hidden md:flex md:w-1/2 bg-slate-900 bg-cover bg-center p-10 flex-col justify-between relative text-white border-r border-slate-100/10"
        >
          {/* Brand Logo Link */}
          <div onClick={handleTryAnother} className="flex items-center gap-2.5 cursor-pointer z-10">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-500 to-brand-400 flex items-center justify-center shadow-lg shadow-brand-500/25">
              <Sparkles className="w-4.5 h-4.5 text-white" />
            </div>
            <span className="text-lg font-bold font-outfit text-white">SkillPath AI</span>
          </div>

          <div className="absolute inset-0 bg-slate-950/20 backdrop-blur-[1px] pointer-events-none" />

          {/* Quote Panel */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white/10 backdrop-blur-md border border-white/15 rounded-2xl p-6 z-10 shadow-lg space-y-4"
          >
            <p className="text-sm font-sans font-medium text-slate-100 leading-relaxed">
              &ldquo;Security is an illusion. The only true safety lies in constant growth and adaptability.&rdquo;
            </p>
            <div className="flex items-center gap-3 pt-1">
              <div className="w-9 h-9 rounded-full bg-brand-500/20 text-brand-300 flex items-center justify-center flex-shrink-0">
                <User className="w-4.5 h-4.5" />
              </div>
              <div>
                <h5 className="font-bold font-outfit text-xs !text-white">Robert Greene</h5>
                <p className="text-[10px] !text-slate-300 font-sans mt-0.5">Author of The 48 Laws of Power</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* RIGHT PANEL: Forgot Password request form */}
        <div className="w-full md:w-1/2 p-8 sm:p-12 md:p-14 flex flex-col justify-between space-y-8 bg-white">
          
          {/* Logo anchor for responsive view */}
          <div className="flex md:hidden items-center gap-2 cursor-pointer pb-2 border-b border-slate-50" onClick={handleTryAnother}>
            <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <span className="text-base font-bold font-outfit bg-gradient-to-r from-brand-900 to-brand-700 bg-clip-text text-transparent">
              SkillPath AI
            </span>
          </div>

          {/* Titles block */}
          <div className="space-y-2">
            <h2 className="text-2xl sm:text-3xl font-bold font-outfit text-slate-900 tracking-tight">
              Lupa Kata Sandi
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 font-sans leading-relaxed">
              Masukkan alamat email Anda di bawah. Kami akan mengirimkan tautan pemulihan kata sandi untuk memulihkan akses akun Anda.
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-700 font-outfit tracking-wide block">
                Alamat Email
              </label>
              <div className="relative">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  className="premium-input pl-11 text-sm font-sans"
                  required
                  disabled={success}
                />
                <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">
                  <Mail className="w-4.5 h-4.5" />
                </div>
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

            {/* Submit & Simulate links dispatched banner */}
            <div className="pt-2 space-y-4">
              {!success && (
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
                      <span>Mengirimkan Tautan...</span>
                    </>
                  ) : (
                    <>
                      <span>Kirim Tautan Pemulihan</span>
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              )}

              {/* Dynamic Alert Banner with Simulation Testing Buttons */}
              <AnimatePresence>
                {success && (
                  <motion.div 
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    className="bg-emerald-50 border border-emerald-100 text-emerald-800 rounded-xl p-4 text-left flex items-start gap-3 shadow-sm mt-3"
                  >
                    <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
                    <div className="space-y-1 w-full">
                      <h5 className="font-bold font-outfit text-emerald-950 text-xs sm:text-sm">Link Dikirimkan!</h5>
                      <p className="text-emerald-700 leading-relaxed font-sans text-xs">
                        Tautan pemulihan kata sandi telah dikirim. Mohon cek email Anda.
                      </p>

                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </form>

          {/* Footer links switching back to Login */}
          <div className="text-center space-y-4 border-t border-slate-50 pt-6">
            <p className="text-xs sm:text-sm text-slate-500 font-sans">
              Ingat kata sandi Anda?{' '}
              <button 
                onClick={triggerLoginView}
                className="font-bold text-brand-600 hover:text-brand-700 hover:underline transition-colors"
              >
                Masuk
              </button>
            </p>
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
