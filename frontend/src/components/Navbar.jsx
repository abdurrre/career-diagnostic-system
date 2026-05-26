import { motion } from 'framer-motion';
import { Sparkles, User } from 'lucide-react';

export default function Navbar({ 
  currentView, 
  activeTab, 
  handleTabChange, 
  triggerLoginView,
  authToken,
  loggedInEmail,
  handleSignOut
}) {
  if (currentView === 'login' || currentView === 'register') {
    return null;
  }

  return (
    <header className="sticky top-0 z-40 bg-[#f7f9ff]/80 backdrop-blur-md border-b border-slate-100/60 px-6 lg:px-16 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          onClick={() => handleTabChange('Home')}
          className="flex items-center gap-2 cursor-pointer"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-400 flex items-center justify-center shadow-lg shadow-brand-500/20">
            <Sparkles className="w-5 h-5 text-white animate-pulse" />
          </div>
          <span className="text-xl font-bold font-outfit bg-gradient-to-r from-brand-900 to-brand-700 bg-clip-text text-transparent">
            SkillPath AI
          </span>
        </motion.div>

        {/* Navigation Links */}
        <nav className="hidden md:flex items-center gap-8 bg-slate-200/40 px-5 py-2 rounded-full border border-slate-100">
          {['Home', 'History', 'About'].map((tab) => {
            const tabLabels = { Home: 'Beranda', History: 'Riwayat', About: 'Tentang' };
            return (
              <button
                key={tab}
                onClick={() => handleTabChange(tab)}
                className={`relative font-medium text-sm transition-colors py-1 px-3 rounded-full ${
                  activeTab === tab ? 'text-brand-700' : 'text-slate-500 hover:text-slate-900'
                }`}
              >
                {tabLabels[tab]}
                {activeTab === tab && (
                  <motion.div
                    layoutId="activeTabUnderline"
                    className="absolute inset-0 bg-white shadow-sm border border-slate-100 rounded-full -z-10"
                    transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                  />
                )}
              </button>
            );
          })}
        </nav>

        {/* Action Buttons */}
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          {authToken ? (
            <div className="flex items-center gap-3">
              <div className="hidden lg:flex items-center gap-2 px-4 py-2 rounded-full bg-slate-200/40 border border-slate-100 text-xs font-semibold text-slate-600 font-sans">
                <User className="w-3.5 h-3.5 text-slate-500" />
                <span className="truncate max-w-[140px]">{loggedInEmail}</span>
              </div>
              <button 
                onClick={handleSignOut}
                className="px-5 py-2.5 rounded-full border border-slate-200/80 hover:border-slate-300 bg-white hover:bg-slate-50 text-slate-600 hover:text-slate-800 font-bold text-xs shadow-sm transition-all active:scale-[0.98] cursor-pointer"
              >
                Keluar
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-4">
              <button 
                onClick={triggerLoginView}
                className="px-6 py-2 rounded-full font-medium text-sm text-slate-600 hover:text-slate-900 transition-colors cursor-pointer"
              >
                Login
              </button>
              <button 
                onClick={triggerLoginView}
                className="px-6 py-2.5 rounded-full bg-brand-600 hover:bg-brand-700 text-white font-medium text-sm shadow-md shadow-brand-600/10 hover:shadow-lg hover:shadow-brand-600/20 transition-all active:scale-[0.98] cursor-pointer"
              >
                Sign Up
              </button>
            </div>
          )}
        </motion.div>
      </div>
    </header>
  );
}
