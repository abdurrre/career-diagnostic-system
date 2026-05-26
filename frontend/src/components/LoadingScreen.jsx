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
              <div className="z-10 bg-brand-50/50 w-24 h-24 rounded-full flex items-center justify-center shadow-inner">
                <svg className="w-11 h-11 text-brand-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 4.5c-3.59 0-6.5 2.91-6.5 6.5 0 1.82.75 3.47 1.95 4.65L7 19.5c-.1.3.1.5.4.5h8.2c.3 0 .5-.2.4-.5l-.45-3.85c1.2-1.18 1.95-2.83 1.95-4.65 0-3.59-2.91-6.5-6.5-6.5Z" />
                  <circle cx="12" cy="11" r="2.5" />
                  <path d="M12 8v1M12 13v1M9 11h1M14 11h1M9.9 8.9l.7.7M13.4 12.4l.7.7M9.9 13.1l.7-.7M13.4 9.6l.7-.7" strokeWidth="2" />
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
