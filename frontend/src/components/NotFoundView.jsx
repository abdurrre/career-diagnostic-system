import { motion } from 'framer-motion';
import { HelpCircle, ArrowLeft, Sparkles } from 'lucide-react';

export default function NotFoundView({ currentView, handleTryAnother }) {
  if (currentView !== 'not-found') {
    return null;
  }

  return (
    <motion.main
      key="not-found"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.35 }}
      className="flex-grow w-full flex items-center justify-center px-4 py-16 md:py-24 bg-[#f4f6fc]"
    >
      {/* Centered Glass 404 Card */}
      <div className="w-full max-w-[480px] bg-white rounded-3xl border border-slate-100/60 shadow-[0_20px_50px_rgba(0,0,0,0.04)] relative overflow-hidden p-8 sm:p-10 text-center space-y-8">
        
        {/* Top ribbon */}
        <div className="absolute top-0 left-0 right-0 h-[5px] bg-gradient-to-r from-slate-400 via-slate-500 to-indigo-500" />

        {/* Logo */}
        <div 
          onClick={handleTryAnother}
          className="flex items-center justify-center gap-2 cursor-pointer opacity-80 hover:opacity-100 transition-opacity"
        >
          <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-brand-600 to-indigo-50 flex items-center justify-center">
            <Sparkles className="w-3.5 h-3.5 text-white" />
          </div>
          <span className="text-sm font-bold font-outfit bg-gradient-to-r from-brand-900 to-brand-700 bg-clip-text text-transparent">
            SkillPath AI
          </span>
        </div>

        {/* 404 Warning icon */}
        <div className="relative w-28 h-28 mx-auto flex items-center justify-center">
          <motion.div
            animate={{ scale: [1, 1.12, 1] }}
            transition={{ repeat: Infinity, duration: 3, ease: "easeInOut" }}
            className="absolute inset-2 bg-slate-500/10 rounded-full blur-xl"
          />
          
          <motion.div 
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="z-10 bg-slate-50 w-20 h-20 rounded-full flex items-center justify-center shadow-inner border border-slate-200/60 text-slate-500"
          >
            <HelpCircle className="w-10 h-10 stroke-[2.2]" />
          </motion.div>
        </div>

        {/* 404 titles */}
        <div className="space-y-3">
          <span className="text-brand-600 font-extrabold text-sm tracking-wider uppercase font-outfit">
            Error 404
          </span>
          <h2 className="text-2xl sm:text-3xl font-bold font-outfit text-slate-900 tracking-tight leading-tight">
            Halaman Tidak Ditemukan
          </h2>
          <p className="text-xs sm:text-sm text-slate-400 font-sans leading-relaxed max-w-xs mx-auto">
            Jalur atau halaman URL yang Anda cari tidak ada atau telah dipindahkan oleh sistem kami.
          </p>
        </div>

        {/* Primary Proceed CTA Button */}
        <div className="pt-2">
          <button
            onClick={handleTryAnother}
            className="w-full py-3.5 rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-bold font-outfit text-sm shadow-md shadow-brand-600/10 hover:shadow-lg hover:shadow-brand-600/20 transition-all flex items-center justify-center gap-2 active:scale-[0.98]"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Kembali ke Beranda</span>
          </button>
        </div>

      </div>
    </motion.main>
  );
}
