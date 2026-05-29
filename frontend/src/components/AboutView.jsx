import { motion } from 'framer-motion';
import { Sparkles, Cpu, Layers, ShieldCheck, ArrowRight, Code2 } from 'lucide-react';

export default function AboutView({ currentView, handleTryAnother }) {
  if (currentView !== 'about') {
    return null;
  }

  return (
    <motion.main
      key="about-view"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.3 }}
      className="max-w-7xl mx-auto px-6 lg:px-16 py-12 lg:py-20 flex-grow w-full space-y-16"
    >
      {/* Title block */}
      <div className="text-center max-w-3xl mx-auto space-y-4">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-brand-50 border border-brand-100/60 text-xs font-bold text-brand-600 uppercase tracking-wider">
          <Sparkles className="w-3.5 h-3.5" />
          Analisis Cerdas & Presisi
        </div>
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-slate-900 tracking-tight leading-tight font-outfit">
          Tentang SkillPath AI
        </h1>
        <p className="text-sm sm:text-base text-slate-500 font-sans leading-relaxed">
          SkillPath AI adalah sistem diagnostik karier canggih yang dirancang untuk menganalisis profil profesional, memetakan kompetensi kandidat terhadap kebutuhan industri, serta merancang peta jalan karier yang konkret dan optimal.
        </p>
      </div>

      {/* Main Core Mission Details Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch max-w-5xl mx-auto">
        
        {/* Card 1: Our Mission */}
        <div className="premium-card p-8 flex flex-col justify-between space-y-6">
          <div className="space-y-4">
            <div className="w-12 h-12 rounded-xl bg-brand-50 text-brand-600 flex items-center justify-center border border-brand-100/40 shadow-sm">
              <Cpu className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold font-outfit text-slate-800">Misi Kami</h3>
            <p className="text-sm text-slate-500 font-sans leading-relaxed">
              Kami membantu para profesional ambisius dengan memecahkan parameter ATS rekrutmen. Mesin cerdas kami mendeteksi kesenjangan keahlian dalam tingkat kritis, penting, maupun pelengkap secara real-time berdasarkan peran target industri.
            </p>
          </div>
          <button 
            onClick={handleTryAnother}
            className="text-brand-600 hover:text-brand-700 font-bold text-xs flex items-center gap-1.5 hover:underline font-outfit self-start"
          >
            <span>Scan CV Anda</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Card 2: Tech Architecture */}
        <div className="premium-card p-8 flex flex-col justify-between space-y-6">
          <div className="space-y-4">
            <div className="w-12 h-12 rounded-xl bg-indigo-50 text-brand-500 flex items-center justify-center border border-indigo-100/40 shadow-sm">
              <Code2 className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold font-outfit text-slate-800">Arsitektur Teknologi</h3>
            <p className="text-sm text-slate-500 font-sans leading-relaxed">
              Dibangun menggunakan struktur modern yang dirancang untuk kecepatan dan responsivitas. Tata letak aplikasi memanfaatkan visual warna HSL, Vite, React core state, dan animasi Framer Motion.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full bg-slate-100 border border-slate-200/40 text-[10px] font-bold text-slate-500">Vite 8.0</span>
            <span className="px-2.5 py-0.5 rounded-full bg-slate-100 border border-slate-200/40 text-[10px] font-bold text-slate-500">React 19</span>
            <span className="px-2.5 py-0.5 rounded-full bg-slate-100 border border-slate-200/40 text-[10px] font-bold text-slate-500">Tailwind v4</span>
          </div>
        </div>

      </div>

      {/* Grid: 3 Column Bullet features */}
      <div className="max-w-5xl mx-auto space-y-8 pt-4 border-t border-slate-100">
        <h3 className="text-center font-bold text-lg font-outfit text-slate-800 uppercase tracking-wider">
          Kemampuan Utama Sistem
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center shadow-inner flex-shrink-0">
              <ShieldCheck className="w-4.5 h-4.5" />
            </div>
            <div className="space-y-1">
              <h5 className="font-bold text-sm text-slate-800 font-outfit">Analisis Teroptimasi ATS</h5>
              <p className="text-xs text-slate-400 font-sans leading-relaxed">
                Mencocokkan profil kandidat secara langsung dengan parameter ATS perekrut.
              </p>
            </div>
          </div>

          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-50 text-brand-600 flex items-center justify-center shadow-inner flex-shrink-0">
              <Layers className="w-4.5 h-4.5" />
            </div>
            <div className="space-y-1">
              <h5 className="font-bold text-sm text-slate-800 font-outfit">Analisis Celah Prioritas</h5>
              <p className="text-xs text-slate-400 font-sans leading-relaxed">
                Mengelompokkan masalah keahlian ke dalam tingkat Kritis, Penting, dan Pelengkap.
              </p>
            </div>
          </div>

          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-lg bg-brand-50 text-indigo-600 flex items-center justify-center shadow-inner flex-shrink-0">
              <Sparkles className="w-4.5 h-4.5" />
            </div>
            <div className="space-y-1">
              <h5 className="font-bold text-sm text-slate-800 font-outfit">Penyimpanan Arsip</h5>
              <p className="text-xs text-slate-400 font-sans leading-relaxed">
                Menyimpan riwayat pemindaian, skor, dan peta jalan Anda untuk diakses kapan saja.
              </p>
            </div>
          </div>
        </div>
      </div>

    </motion.main>
  );
}
