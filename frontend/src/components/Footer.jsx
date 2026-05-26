import { Sparkles } from 'lucide-react';

export default function Footer({ currentView, handleTryAnother, triggerNotFoundView }) {
  if (currentView === 'login' || currentView === 'register') {
    return null;
  }

  return (
    <footer className="border-t border-slate-100 bg-white/40 backdrop-blur-sm px-6 lg:px-16 py-10 mt-auto">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        {/* Logo & Copyright */}
        <div className="flex flex-col items-center md:items-start gap-2">
          <div className="flex items-center gap-2 cursor-pointer" onClick={handleTryAnother}>
            <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <span className="text-lg font-bold font-outfit bg-gradient-to-r from-brand-900 to-brand-700 bg-clip-text text-transparent">
              SkillPath AI
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 font-sans text-center md:text-left">
            &copy; 2026 SkillPath AI. Pemberdayaan potensi karier lewat analisis presisi.
          </p>
        </div>

        {/* Links */}
        {/* <div className="flex flex-wrap justify-center gap-x-8 gap-y-3">
          {['Privacy Policy', 'Terms of Service', 'Contact Support', 'Documentation'].map((link) => (
            <button 
              key={link} 
              onClick={triggerNotFoundView}
              className="text-xs font-semibold text-slate-400 hover:text-slate-700 hover:underline transition-all font-sans cursor-pointer bg-transparent border-none py-0"
            >
              {link}
            </button>
          ))}
        </div> */}
      </div>
    </footer>
  );
}
