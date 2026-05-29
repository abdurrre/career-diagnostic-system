import { motion } from 'framer-motion';
import { 
  ListChecks, 
  RefreshCw, 
  Bookmark, 
  CheckCircle2, 
  Award, 
  TrendingUp,
  AlertCircle,
  Loader2
} from 'lucide-react';

const getFeedbackMessage = (score) => {
  const s = Math.round(score);
  const index = s % 5;
  
  if (s >= 0 && s <= 20) {
    return [
      "Profil CV Anda saat ini memiliki kecocokan yang sangat minim dengan standar industri. Diperlukan peningkatan keahlian fundamental yang signifikan.",
      "Masih terdapat kesenjangan besar antara kualifikasi Anda dan kebutuhan peran. Fokus pada pengembangan kompetensi dasar terlebih dahulu.",
      "Penyelarasan CV Anda berada pada tingkat awal. Kami menyarankan untuk mengambil sertifikasi dasar dan melengkapi keahlian inti yang wajib dimiliki.",
      "Tingkat kecocokan sangat rendah. Mulailah membangun portofolio dasar dan pelajari keterampilan teknis mendasar dari peran ini.",
      "CV Anda memerlukan perombakan besar untuk peran ini. Prioritaskan penguasaan modul dasar sebelum melamar posisi ini."
    ][index];
  } else if (s >= 21 && s <= 40) {
    return [
      "Kualifikasi Anda mulai menunjukkan kecocokan dasar, namun Anda perlu membangun fondasi keahlian teknis yang lebih solid.",
      "Anda memiliki beberapa keahlian dasar, tetapi kesenjangan utama masih menghambat daya saing profil Anda di industri.",
      "Penyelarasan profil berada di tingkat pemula. Fokuslah meningkatkan keterampilan prioritas untuk memperbesar peluang Anda lolos seleksi.",
      "Profil CV Anda telah mendeteksi beberapa kata kunci penting, namun kompetensi praktis inti masih perlu ditingkatkan secara menyeluruh.",
      "Langkah awal yang baik, tetapi profil Anda membutuhkan peningkatan keahlian tambahan untuk memenuhi ekspektasi minimal perekrut."
    ][index];
  } else if (s >= 41 && s <= 60) {
    return [
      "CV Anda memiliki kecocokan menengah. Dengan menambahkan beberapa kompetensi kunci yang hilang, profil Anda akan jauh lebih kompetitif.",
      "Kualifikasi Anda sudah mencakup sebagian besar kebutuhan dasar. Mengatasi celah keahlian penting akan sangat meningkatkan peluang Anda.",
      "Anda memiliki modal dasar yang cukup baik. Fokuslah mempelajari teknologi pendukung untuk memperkuat relevansi profil CV Anda.",
      "Relevansi profil Anda berada di tingkat moderat. Sedikit pemolesan pada portofolio dan keahlian spesifik akan membuat Anda lebih menonjol.",
      "Kompetensi Anda menunjukkan potensi yang baik. Menjembatani kesenjangan keahlian tingkat menengah akan memperkuat kesiapan kerja Anda."
    ][index];
  } else if (s >= 61 && s <= 80) {
    return [
      "Profil CV Anda sangat kuat dan memiliki keselarasan yang baik. Menambal sisa kesenjangan kecil akan menjadikan Anda kandidat unggulan.",
      "Anda memiliki kecocokan fondasi yang matang untuk peran ini. Tambahkan detail proyek praktis untuk menyempurnakan portofolio Anda.",
      "Kualifikasi Anda telah memenuhi mayoritas kriteria perekrut. Optimalkan keahlian pelengkap Anda untuk memenangkan persaingan kerja.",
      "Tingkat keselarasan CV Anda tinggi. Fokus pada penguasaan alat pendukung agar profil Anda terlihat sangat solid dan siap pakai.",
      "CV Anda menunjukkan potensi tinggi. Sedikit peningkatan pada keterampilan spesifik akan langsung menarik perhatian tim rekrutmen."
    ][index];
  } else {
    return [
      "Luar biasa! Profil CV Anda sangat selaras dengan standar industri terbaik. Anda adalah kandidat yang sangat ideal untuk peran ini.",
      "Kualifikasi Anda hampir sempurna dan memenuhi ekspektasi tertinggi perekrut. Anda sudah siap untuk segera melamar posisi ini!",
      "CV Anda menunjukkan penguasaan keahlian yang komprehensif dan mendalam. Pertahankan portofolio ini untuk menaklukkan proses wawancara.",
      "Profil Anda memiliki kecocokan yang sangat istimewa. Anda siap bersaing di level tertinggi industri untuk memperebutkan peran ini.",
      "Keselarasan yang luar biasa solid. Kompetensi utama dan pelengkap Anda telah terpenuhi dengan sangat baik untuk standar tahun 2026."
    ][index];
  }
};

export default function ReportView({
  currentView,
  analyzedRole,
  currentRoleData,
  handleTryAnother,
  handleSaveResults,
  saveLoading
}) {
  if (currentView !== 'report') {
    return null;
  }

  return (
    <motion.main
      key="report-view"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.3 }}
      className="max-w-7xl mx-auto px-6 lg:px-16 py-12 lg:py-20 flex-grow w-full space-y-12"
    >
      {/* Header / Titles Block */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-4">
        <div className="space-y-3 max-w-3xl">
          {/* Analysis Complete Badge */}
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg bg-indigo-50 border border-indigo-100/60 text-xs font-bold text-brand-600 uppercase tracking-wider">
            <ListChecks className="w-3.5 h-3.5" />
            Analisis Selesai
          </div>
          
          {/* Dynamic H1 Header */}
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-slate-900 tracking-tight leading-tight">
            Kecocokan Profil {analyzedRole}
          </h1>
          
          {/* Dynamically bound descriptive subtitle */}
          <p className="text-sm sm:text-base text-slate-500 font-sans max-w-2xl">
            Berdasarkan CV yang Anda upload dan persyaratan standar industri untuk peran <strong>{analyzedRole}</strong> saat ini.
          </p>
          <p className="text-xs text-slate-400 font-sans max-w-2xl leading-relaxed">
            Disclaimer: Skor dan rekomendasi ini adalah estimasi pendukung keputusan, bukan jaminan diterima kerja atau pengganti penilaian profesional dari perekrut.
          </p>
        </div>

        {/* Action Buttons: Try Another & Save */}
        <div className="flex flex-wrap items-center gap-3">
          <button 
            onClick={handleTryAnother}
            className="px-6 py-3 rounded-full border border-slate-200/80 hover:border-slate-300 bg-white hover:bg-slate-50 text-slate-600 hover:text-slate-800 font-bold text-sm shadow-sm transition-all flex items-center gap-2"
          >
            <RefreshCw className="w-4 h-4" />
            Coba Profesi Lain
          </button>
          
          <button 
            onClick={handleSaveResults}
            disabled={saveLoading}
            className={`px-6 py-3 rounded-full bg-brand-600 hover:bg-brand-700 text-white font-bold text-sm shadow-md shadow-brand-600/10 hover:shadow-lg hover:shadow-brand-600/20 transition-all flex items-center gap-2 active:scale-[0.98] ${
              saveLoading ? 'opacity-85 cursor-not-allowed' : ''
            }`}
          >
            {saveLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Bookmark className="w-4 h-4" />
            )}
            <span>{saveLoading ? 'Menyimpan...' : 'Simpan Hasil'}</span>
          </button>
        </div>
      </div>

      {/* Split Section: Match Score & Identified Skills */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
        
        {/* LEFT CARD: Match Score Percentage */}
        <div className="lg:col-span-4 premium-card border-l-4 border-l-brand-600 p-8 flex flex-col justify-between text-center space-y-6">
          <div>
            <h3 className="text-lg font-bold font-outfit text-slate-800">Skor Kecocokan</h3>
          </div>

          {/* Animated SVG Circular Progress Graph */}
          <div className="relative w-48 h-48 mx-auto flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="40"
                className="stroke-slate-100 fill-none"
                strokeWidth="7"
              />
              <motion.circle
                cx="50"
                cy="50"
                r="40"
                className="stroke-brand-600 fill-none"
                strokeWidth="7.5"
                strokeDasharray="251.2"
                initial={{ strokeDashoffset: 251.2 }}
                animate={{ strokeDashoffset: 251.2 - (251.2 * currentRoleData.matchScore) / 100 }}
                transition={{ duration: 1.5, ease: "easeOut" }}
                strokeLinecap="round"
              />
            </svg>
            
            <div className="absolute flex flex-col items-center justify-center">
              <span className="text-4xl font-extrabold font-outfit text-slate-900 tracking-tight">
                {currentRoleData.matchScore}%
              </span>
              <span className="text-[10px] font-bold text-slate-400 tracking-widest uppercase mt-0.5">
                Kesesuaian
              </span>
            </div>
          </div>

          <div className="px-2">
            <p className="text-sm text-slate-500 font-sans leading-relaxed">
              {getFeedbackMessage(currentRoleData.matchScore)}
            </p>
          </div>
        </div>

        {/* RIGHT CARD: Identified Skills Tags */}
        <div className="lg:col-span-8 premium-card p-8 flex flex-col justify-between space-y-6">
          
          <div className="flex items-start gap-3.5">
            <div className="w-10 h-10 rounded-full bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600 flex-shrink-0">
              <CheckCircle2 className="w-5.5 h-5.5" />
            </div>
            <div>
              <h3 className="text-lg font-bold font-outfit text-slate-800">Keahlian Teridentifikasi</h3>
              <p className="text-xs text-slate-400 font-sans">Kompetensi yang berhasil dikenali dari profil Anda</p>
            </div>
          </div>

          {/* Flex tag badge containers */}
          <div className="flex flex-wrap gap-2.5 py-4">
            {currentRoleData.skills.map((skill, index) => {
              const isLavenderStyle = index % 3 !== 2;
              return (
                <motion.span
                  key={skill}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.05 }}
                  className={"px-4 py-2 rounded-full font-medium text-xs sm:text-sm border transition-all hover:scale-[1.03] cursor-default bg-brand-50/70 text-brand-700 border-brand-100/60"}
                >
                  {skill}
                </motion.span>
              );
            })}
          </div>

          <div className="bg-slate-50 rounded-xl p-4 text-xs text-slate-400 font-sans flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-brand-500 flex-shrink-0" />
            <span>Catatan: Hasil analisis ini bersifat estimasi berbasis AI dan tidak dijamin 100% akurat. Gunakan sebagai referensi tambahan dalam pengembangan karier Anda.</span>
          </div>
        </div>

      </div>

      {/* Bottom Section: Target Skill Gaps Card Grid */}
      <div className="space-y-6 pt-4">
        
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-rose-50 text-rose-500 border border-rose-100/60 flex items-center justify-center shadow-sm">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 font-outfit">
              Kesenjangan Keahlian Target
            </h2>
            <p className="text-xs text-slate-400 font-sans mt-0.5">Area prioritas untuk pengembangan karier segera</p>
          </div>
        </div>

        {/* Three dynamic color-coded gaps cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {currentRoleData.gaps.map((gap, index) => {
            let borderLeftClass = "border-l-rose-500";
            let bgBadgeClass = "bg-rose-50 text-rose-600";
            if (gap.tier === "IMPORTANT") {
              borderLeftClass = "border-l-amber-500";
              bgBadgeClass = "bg-amber-50 text-amber-600";
            } else if (gap.tier === "SUPPLEMENTARY") {
              borderLeftClass = "border-l-slate-400";
              bgBadgeClass = "bg-slate-100 text-slate-600";
            }

            return (
              <motion.div
                key={gap.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + index * 0.1 }}
                className={`premium-card border-l-4 ${borderLeftClass} p-6 flex flex-col justify-between space-y-4`}
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <h4 className="font-bold text-base sm:text-lg text-slate-800 font-outfit truncate pr-2">
                      {gap.title}
                    </h4>
                    <span className={`px-2.5 py-1 rounded-lg text-[9px] font-extrabold font-outfit tracking-wider uppercase flex-shrink-0 ${bgBadgeClass}`}>
                      {gap.tier}
                    </span>
                  </div>
                  <p className="text-xs sm:text-sm text-slate-500 font-sans leading-relaxed">
                    {gap.description}
                  </p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

    </motion.main>
  );
}
