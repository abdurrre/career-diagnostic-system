import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, 
  FileText, 
  Calendar, 
  Trash2, 
  Eye, 
  Sparkles, 
  Plus,
  TrendingUp,
  Award,
  ListFilter
} from 'lucide-react';

export default function HistoryView({
  currentView,
  historyList,
  handleLoadReport,
  handleDeleteRecord,
  handleTryAnother
}) {
  const [searchQuery, setSearchQuery] = useState('');

  if (currentView !== 'history') {
    return null;
  }

  // Filter history based on search query
  const filteredHistory = historyList.filter(item => 
    item.role.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.fileName.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Calculate summary metrics
  const totalAnalyzed = historyList.length;
  const avgScore = totalAnalyzed > 0 
    ? Math.round(historyList.reduce((acc, curr) => acc + curr.score, 0) / totalAnalyzed) 
    : 0;

  return (
    <motion.main
      key="history-view"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.3 }}
      className="max-w-7xl mx-auto px-6 lg:px-16 py-12 lg:py-20 flex-grow w-full space-y-12"
    >
      {/* Header Block */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-2">
        <div className="space-y-3">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-brand-50 border border-brand-100/60 text-xs font-bold text-brand-600 uppercase tracking-wider">
            <Calendar className="w-3.5 h-3.5" />
            Arsip Diagnostik
          </div>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-slate-900 tracking-tight leading-tight font-outfit">
            Riwayat Analisis Anda
          </h1>
          <p className="text-sm sm:text-base text-slate-500 font-sans max-w-2xl">
            Catatan komprehensif dari semua pemindaian diagnostik CV, kualifikasi target, dan tingkat kecocokan keahlian Anda.
          </p>
        </div>

        {/* CTA to start new analysis */}
        <div>
          <button 
            onClick={handleTryAnother}
            className="px-6 py-3 rounded-full bg-brand-600 hover:bg-brand-700 text-white font-bold text-sm shadow-md shadow-brand-600/10 hover:shadow-lg hover:shadow-brand-600/20 transition-all flex items-center gap-2 active:scale-[0.98] cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            Analisis CV Baru
          </button>
        </div>
      </div>

      {/* Summary Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="premium-card p-6 flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-brand-50 text-brand-600 flex items-center justify-center flex-shrink-0 shadow-sm border border-brand-100/40">
            <Sparkles className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <span className="text-2xl font-black text-slate-800 font-outfit leading-none">{totalAnalyzed}</span>
            <p className="text-xs text-slate-400 mt-0.5 font-sans font-medium">CV Teranalisis</p>
          </div>
        </div>

        <div className="premium-card p-6 flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center flex-shrink-0 shadow-sm border border-emerald-100/40">
            <TrendingUp className="w-6 h-6" />
          </div>
          <div>
            <span className="text-2xl font-black text-slate-800 font-outfit leading-none">{avgScore}%</span>
            <p className="text-xs text-slate-400 mt-0.5 font-sans font-medium">Rata-rata Skor Kecocokan</p>
          </div>
        </div>

        <div className="premium-card p-6 flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-indigo-50 text-brand-500 flex items-center justify-center flex-shrink-0 shadow-sm border border-indigo-100/40">
            <Award className="w-6 h-6" />
          </div>
          <div>
            <span className="text-sm font-bold text-slate-700 font-outfit leading-none">Sistem Diagnostik Aktif</span>
            <p className="text-xs text-slate-400 mt-1 font-sans font-medium">Mesin Inti Vite + React</p>
          </div>
        </div>
      </div>

      {/* Controls & Search */}
      <div className="premium-card p-4 sm:p-6 flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Search Field */}
        <div className="relative w-full md:max-w-md">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Cari profesi impian atau file CV..."
            className="premium-input pl-11 text-sm font-sans w-full"
          />
          <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">
            <Search className="w-4.5 h-4.5" />
          </div>
        </div>

        {/* Filter count indicator */}
        <div className="flex items-center gap-2 text-xs text-slate-400 font-sans font-medium">
          <ListFilter className="w-4 h-4 text-slate-400" />
          Menampilkan {filteredHistory.length} dari {totalAnalyzed} riwayat analisis
        </div>
      </div>

      {/* History Data Table/Cards Block */}
      <div className="overflow-hidden">
        <AnimatePresence mode="wait">
          {filteredHistory.length > 0 ? (
            <motion.div
              key="table-results"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="premium-card overflow-x-auto p-0 border border-slate-100"
            >
              <table className="w-full text-left border-collapse min-w-[700px] font-sans">
                <thead>
                  <tr className="border-b border-slate-100 bg-slate-50/50 text-slate-400 font-bold text-[11px] uppercase tracking-wider font-outfit">
                    <th className="py-4.5 px-6">Profesi Impian</th>
                    <th className="py-4.5 px-6 text-center">Skor Kecocokan</th>
                    <th className="py-4.5 px-6">File CV</th>
                    <th className="py-4.5 px-6">Tanggal Analisis</th>
                    <th className="py-4.5 px-6 text-right">Aksi</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100/60 text-slate-600 text-xs sm:text-sm">
                  {filteredHistory.map((item, index) => {
                    // Match score color codes
                    let scoreBadgeColor = "bg-brand-50 text-brand-700 border-brand-100/60";
                    let scoreCircleColor = "stroke-brand-600";
                    if (item.score >= 82) {
                      scoreBadgeColor = "bg-emerald-50 text-emerald-700 border-emerald-100/60";
                      scoreCircleColor = "stroke-emerald-600";
                    } else if (item.score < 77) {
                      scoreBadgeColor = "bg-amber-50 text-amber-700 border-amber-100/60";
                      scoreCircleColor = "stroke-amber-500";
                    }

                    return (
                      <motion.tr
                        key={item.id}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.04 }}
                        className="transition-colors hover:bg-slate-50/40 group"
                      >
                        {/* Target Role */}
                        <td className="py-4 px-6 font-bold text-slate-800 font-outfit">
                          {item.role}
                        </td>
                        
                        {/* Match Score */}
                        <td className="py-4 px-6">
                          <div className="flex items-center justify-center gap-2">
                            {/* SVG Mini alignment gauge */}
                            <div className="relative w-6 h-6 flex items-center justify-center">
                              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                                <circle cx="50" cy="50" r="40" className="stroke-slate-100 fill-none" strokeWidth="12" />
                                <circle cx="50" cy="50" r="40" className={`${scoreCircleColor} fill-none`} strokeWidth="12.5" strokeDasharray="251.2" strokeDashoffset={251.2 - (251.2 * item.score) / 100} strokeLinecap="round" />
                              </svg>
                            </div>
                            <span className={`px-2.5 py-0.5 rounded-full border text-xs font-bold font-outfit ${scoreBadgeColor}`}>
                              {item.score}%
                            </span>
                          </div>
                        </td>

                        {/* Attached CV */}
                        <td className="py-4 px-6">
                          <div className="flex items-center gap-2 text-slate-400 group-hover:text-slate-600 transition-colors">
                            <FileText className="w-4 h-4 text-slate-400 flex-shrink-0" />
                            <span className="truncate max-w-[180px] font-sans font-medium text-xs">
                              {item.fileName}
                            </span>
                          </div>
                        </td>

                        {/* Date Analyzed */}
                        <td className="py-4 px-6 text-slate-400 font-sans text-xs">
                          {item.date}
                        </td>

                        {/* Actions */}
                        <td className="py-4 px-6 text-right">
                          <div className="flex items-center justify-end gap-2.5">
                            <button
                              onClick={() => handleLoadReport(item)}
                              className="px-3.5 py-1.5 rounded-lg bg-indigo-50 hover:bg-brand-600 text-brand-600 hover:text-white font-bold text-xs transition-all flex items-center gap-1 cursor-pointer active:scale-95 border border-indigo-100/50 hover:border-brand-600 hover:shadow-sm"
                              title="Lihat hasil penuh"
                            >
                              <Eye className="w-3.5 h-3.5" />
                              <span>Lihat Hasil</span>
                            </button>
                            <button
                              onClick={() => handleDeleteRecord(item.id)}
                              className="p-2 rounded-lg border border-slate-100 hover:border-rose-100 text-slate-400 hover:text-rose-600 hover:bg-rose-50/50 transition-all cursor-pointer active:scale-95"
                              title="Remove record"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>
            </motion.div>
          ) : (
            /* EMPTY STATE CARD */
            <motion.div
              key="table-empty"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="premium-card p-12 text-center bg-white/60 backdrop-blur-sm max-w-2xl mx-auto border border-dashed border-slate-200/80 space-y-6"
            >
              <div className="w-16 h-16 rounded-full bg-slate-50 border border-slate-100 flex items-center justify-center mx-auto text-slate-400 shadow-inner">
                <FileText className="w-8 h-8" />
              </div>
              <div className="space-y-2 max-w-sm mx-auto">
                <h3 className="text-xl font-bold font-outfit text-slate-800">
                  {searchQuery ? 'Tidak Ada Riwayat yang Cocok' : 'Belum Ada Riwayat Analisis'}
                </h3>
                <p className="text-sm text-slate-400 font-sans leading-relaxed">
                  {searchQuery 
                    ? `Tidak ada arsip analisis yang cocok dengan "${searchQuery}". Coba gunakan kata kunci pencarian lain.`
                    : 'Scan CV Anda pada profesi target untuk memetakan keselarasan dan melihat riwayat Anda di sini.'}
                </p>
              </div>
              <div>
                <button
                  onClick={handleTryAnother}
                  className="px-6 py-2.5 rounded-full bg-brand-600 hover:bg-brand-700 text-white font-bold text-sm shadow-md shadow-brand-600/10 hover:shadow-lg hover:shadow-brand-600/20 transition-all flex items-center justify-center gap-2 mx-auto active:scale-95 cursor-pointer"
                >
                  <Plus className="w-4 h-4" />
                  <span>Mulai Analisis Baru</span>
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

    </motion.main>
  );
}
