import { motion, AnimatePresence } from 'framer-motion';
import { Loader2 } from 'lucide-react';

export default function LoadingScreen({ isAnalyzing, loadingStep }) {
  return (
    <AnimatePresence>
      {isAnalyzing && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-[#f7f9ff]/95 backdrop-blur-md"
        >
          {/* Centered Loading Card */}
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={{ type: 'spring', duration: 0.5 }}
            className="w-full max-w-[480px] bg-white rounded-3xl border border-slate-100/60 shadow-[0_20px_50px_rgba(0,0,0,0.06)] relative overflow-hidden p-8 md:p-10 text-center space-y-8 mx-4"
          >
            {/* Top Gradient Ribbon */}
            <div className="absolute top-0 left-0 right-0 h-[5px] bg-gradient-to-r from-brand-500 via-brand-600 to-indigo-600" />
            
            {/* Spinner & Brain-Gear Silhouette Icon Container */}
            <div className="relative w-36 h-36 mx-auto flex items-center justify-center">
              {/* Custom rotating thick circle spinner */}
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                className="absolute inset-0 rounded-full border-[5px] border-slate-100 border-t-brand-600"
              />
              
              {/* Cognitive/Brain Gear AI Icon Inside */}
              <div className="z-10 bg-brand-50/50 w-24 h-24 rounded-full flex items-center justify-center shadow-inner relative">
                <style>{`
                  @keyframes spin-cw {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                  }
                  .gear-cw {
                    transform-origin: 12px 10px;
                    animation: spin-cw 4s linear infinite;
                  }
                `}</style>
                <svg className="w-12 h-12 text-brand-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  {/* Head Contour (Left Side Profile - Highly recognizable) */}
                  <path 
                    d="M15.5 21.5c0-1.5.5-3 1.5-4s1.8-3 1.8-6c0-5-3.3-8-7.3-8-4 0-5 2.5-5 4s.2 1.7.2 2c0 .3-.3.5-.5.8-0.4.5-1.9 1.2-1.9 1.7s1.5.7 2 .9c0.4.2-.1.6-.1.9s.3.7-.2 1.2c-0.5.5-.8 1-.5 1.5.3.5 2 1.3 4 1.5s.5 2 .5 3.5"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  {/* Single Centered Gear in Brain (Smaller & delicate) */}
                  <g className="gear-cw">
                    <circle cx="12" cy="10" r="2.2" stroke="currentColor" strokeWidth="1.5" fill="none" />
                    <circle cx="12" cy="10" r="0.6" fill="currentColor" />
                    <path d="M12 7v.8M12 12.2v.8M9 10h.8M14.2 10h.8M9.84 7.84l.6.6M13.56 11.56l.6.6M9.84 12.16l.6-.6M13.56 8.44l.6-.6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                  </g>
                </svg>
              </div>
            </div>

            {/* Headings */}
            <div className="space-y-2">
              <h2 className="text-2xl font-bold font-outfit text-slate-800">Menganalisis Profil</h2>
              <p className="text-sm text-slate-400 font-sans leading-relaxed max-w-xs mx-auto">
                Mohon tunggu sebentar, AI kami sedang memetakan keahlian Anda.
              </p>
            </div>

            {/* Sequential Steps Checklist */}
            <div className="bg-slate-50 border border-slate-100/60 rounded-2xl p-6 text-left space-y-4 max-w-sm mx-auto">
              {/* Step 1 */}
              <div className="flex items-center gap-3.5">
                {loadingStep > 1 ? (
                  <div className="w-5 h-5 rounded-full bg-brand-600 text-white flex items-center justify-center flex-shrink-0">
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                ) : loadingStep === 1 ? (
                  <Loader2 className="w-5 h-5 text-brand-600 animate-spin flex-shrink-0" />
                ) : (
                  <div className="w-5 h-5 rounded-full border-2 border-slate-200 flex-shrink-0" />
                )}
                <span className={`text-sm font-sans transition-all duration-200 ${
                  loadingStep === 1 
                    ? 'text-brand-700 font-bold' 
                    : loadingStep > 1 
                      ? 'text-slate-400 font-medium' 
                      : 'text-slate-300'
                }`}>
                  Mengekstrak keahlian dari CV Anda...
                </span>
              </div>

              {/* Step 2 */}
              <div className="flex items-center gap-3.5">
                {loadingStep > 2 ? (
                  <div className="w-5 h-5 rounded-full bg-brand-600 text-white flex items-center justify-center flex-shrink-0">
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                ) : loadingStep === 2 ? (
                  <Loader2 className="w-5 h-5 text-brand-600 animate-spin flex-shrink-0" />
                ) : (
                  <div className="w-5 h-5 rounded-full border-2 border-slate-200 flex-shrink-0" />
                )}
                <span className={`text-sm font-sans transition-all duration-200 ${
                  loadingStep === 2 
                    ? 'text-brand-700 font-bold' 
                    : loadingStep > 2 
                      ? 'text-slate-400 font-medium' 
                      : 'text-slate-300'
                }`}>
                  Menganalisis standar kebutuhan industri...
                </span>
              </div>

              {/* Step 3 */}
              <div className="flex items-center gap-3.5">
                {loadingStep > 3 ? (
                  <div className="w-5 h-5 rounded-full bg-brand-600 text-white flex items-center justify-center flex-shrink-0">
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                ) : loadingStep === 3 ? (
                  <Loader2 className="w-5 h-5 text-brand-600 animate-spin flex-shrink-0" />
                ) : (
                  <div className="w-5 h-5 rounded-full border-2 border-slate-200 flex-shrink-0" />
                )}
                <span className={`text-sm font-sans transition-all duration-200 ${
                  loadingStep === 3 
                    ? 'text-brand-700 font-bold' 
                    : loadingStep > 3 
                      ? 'text-slate-400 font-medium' 
                      : 'text-slate-300'
                }`}>
                  Menghitung skor kecocokan Anda...
                </span>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
