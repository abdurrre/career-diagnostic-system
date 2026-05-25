import { motion } from 'framer-motion';
import { CheckCircle2, ArrowRight, Sparkles } from 'lucide-react';

export default function VerificationSuccessView({ currentView, triggerLoginView, handleTryAnother }) {
  if (currentView !== 'verification-success') {
    return null;
  }

  return (
    <motion.main
      key="verification-success"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.35 }}
      className="flex-grow w-full flex items-center justify-center px-4 py-16 md:py-24 bg-[#f4f6fc]"
    >
      {/* Centered Premium Glassmorphic Success Card */}
      <div className="w-full max-w-[480px] bg-white rounded-3xl border border-slate-100/60 shadow-[0_20px_50px_rgba(0,0,0,0.04)] relative overflow-hidden p-8 sm:p-10 text-center space-y-8">
        
        {/* Top Decorative Gradient Ribbon */}
        <div className="absolute top-0 left-0 right-0 h-[5px] bg-gradient-to-r from-emerald-400 via-emerald-500 to-teal-500" />

        {/* Brand Logo Header */}
        <div 
          onClick={handleTryAnother}
          className="flex items-center justify-center gap-2 cursor-pointer opacity-80 hover:opacity-100 transition-opacity"
        >
          <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-brand-600 to-indigo-500 flex items-center justify-center shadow-md shadow-brand-500/20">
            <Sparkles className="w-3.5 h-3.5 text-white" />
          </div>
          <span className="text-sm font-bold font-outfit bg-gradient-to-r from-brand-900 to-brand-700 bg-clip-text text-transparent">
            SkillPath AI
          </span>
        </div>

        {/* Animated Verification Success Glowing Icon */}
        <div className="relative w-28 h-28 mx-auto flex items-center justify-center">
          {/* Pulsing Green Background Glow */}
          <motion.div
            animate={{ scale: [1, 1.15, 1] }}
            transition={{ repeat: Infinity, duration: 2.5, ease: "easeInOut" }}
            className="absolute inset-2 bg-emerald-500/10 rounded-full blur-xl"
          />
          
          <motion.div 
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 15 }}
            className="z-10 bg-emerald-50 w-20 h-20 rounded-full flex items-center justify-center shadow-inner border border-emerald-100/60 text-emerald-600"
          >
            <CheckCircle2 className="w-10 h-10 stroke-[2.2]" />
          </motion.div>
        </div>

        {/* Successful Verification Headings */}
        <div className="space-y-3">
          <h2 className="text-2xl sm:text-3xl font-bold font-outfit text-slate-900 tracking-tight leading-tight">
            Email Verified Successfully
          </h2>
          <p className="text-xs sm:text-sm text-slate-400 font-sans leading-relaxed max-w-xs mx-auto">
            Your profile email address has been verified. You can now sign in to start diagnostic skill mapping.
          </p>
        </div>

        {/* Primary Proceed CTA Button */}
        <div className="pt-2">
          <button
            onClick={triggerLoginView}
            className="w-full py-3.5 rounded-xl bg-brand-600 hover:bg-brand-700 text-white font-bold font-outfit text-sm shadow-md shadow-brand-600/10 hover:shadow-lg hover:shadow-brand-600/20 transition-all flex items-center justify-center gap-2 active:scale-[0.98]"
          >
            <span>Proceed to Sign In</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>

        {/* Supportive Footer Note */}
        <div className="text-[11px] text-slate-400 font-sans border-t border-slate-50 pt-6">
          Need support? Reach out to our team at{' '}
          <a href="#contact" className="font-semibold text-brand-600 hover:underline">
            support@skillpath.ai
          </a>
        </div>

      </div>
    </motion.main>
  );
}
