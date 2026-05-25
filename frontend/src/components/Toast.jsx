import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, X } from 'lucide-react';

export default function Toast({ showToast, setShowToast, toastMessage }) {
  return (
    <AnimatePresence>
      {showToast && (
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 20, scale: 0.9 }}
          className="fixed bottom-6 right-6 z-55 bg-slate-900 border border-slate-800 text-slate-100 rounded-2xl px-5 py-4 shadow-2xl flex items-center gap-3.5 max-w-sm"
        >
          <div className="w-8 h-8 rounded-full bg-brand-600 flex items-center justify-center text-white flex-shrink-0">
            <CheckCircle2 className="w-4.5 h-4.5" />
          </div>
          <div className="flex-grow space-y-0.5">
            <h5 className="font-bold text-xs sm:text-sm font-outfit !text-white">Action Confirmed</h5>
            <p className="!text-slate-300 font-sans text-[11px] leading-relaxed">
              {toastMessage}
            </p>
          </div>
          <button 
            onClick={() => setShowToast(false)} 
            className="p-1 text-slate-500 hover:text-white rounded-lg transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
