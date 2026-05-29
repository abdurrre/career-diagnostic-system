import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  FileText, 
  CheckCircle2, 
  AlertCircle, 
  Sparkles, 
  Briefcase, 
  FileUp, 
  ChevronDown, 
  X, 
  Loader2, 
  TrendingUp 
} from 'lucide-react';
import howItWorksImg from '../assets/how_it_works.png';
import { professions } from '../data/professions';

export default function HomeView({
  currentView,
  selectedRole,
  setSelectedRole,
  additionalContext,
  setAdditionalContext,
  cvFile,
  fileError,
  isDragActive,
  fileInputRef,
  isAnalyzing,
  handleAnalyze,
  handleDrag,
  handleDrop,
  handleFileChange,
  handleRemoveFile,
  triggerFileInput
}) {
  if (currentView !== 'home') {
    return null;
  }

  return (
    <motion.main 
      key="home-view"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -15 }}
      transition={{ duration: 0.3 }}
      className="max-w-7xl mx-auto px-6 lg:px-16 py-12 lg:py-20 flex-grow w-full space-y-24"
    >
      {/* Hero / Headline */}
      <section className="text-center max-w-4xl mx-auto space-y-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-950 font-outfit leading-tight">
            Jembatani kesenjangan keahlian Anda dengan <span className="bg-gradient-to-r from-brand-600 to-indigo-600 bg-clip-text text-transparent">AI</span>
          </h1>
        </motion.div>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.15 }}
          className="text-base sm:text-lg text-slate-500 leading-relaxed font-sans max-w-3xl mx-auto"
        >
          Upload CV Anda, pilih profesi impian, dan biarkan sistem AI kami merancang peta jalan karier khusus untuk Anda. Analisis presisi untuk profesional ambisius.
        </motion.p>
      </section>

      {/* CV Form inputs */}
      <section className="max-w-6xl mx-auto">
        <motion.form 
          onSubmit={handleAnalyze}
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="space-y-8"
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">
            
            {/* Left: CV Upload */}
            <div 
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={triggerFileInput}
              className={`premium-card cursor-pointer p-8 flex flex-col items-center justify-center text-center relative border-2 border-dashed transition-all min-h-[320px] ${
                isDragActive 
                  ? 'border-brand-500 bg-brand-50/40 shadow-inner' 
                  : cvFile 
                    ? 'border-emerald-300 bg-emerald-50/10' 
                    : 'border-slate-200/80 bg-white hover:border-brand-400'
              }`}
            >
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".pdf"
                className="hidden" 
              />

              <AnimatePresence mode="wait">
                {!cvFile ? (
                  <motion.div 
                    key="upload-prompt"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="space-y-5"
                  >
                    <div className="w-16 h-16 rounded-full bg-brand-50 flex items-center justify-center mx-auto text-brand-600 shadow-inner group-hover:scale-110 transition-transform">
                      <Upload className="w-8 h-8" />
                    </div>
                    <div className="space-y-2">
                      <h3 className="text-xl font-bold font-outfit text-slate-800">Upload CV Anda</h3>
                      <p className="text-sm text-slate-400 font-sans max-w-xs mx-auto">
                        Seret dan lepas file PDF di sini, atau <span className="text-brand-600 font-medium hover:underline">klik untuk memilih file</span>.
                      </p>
                    </div>
                    <span className="inline-block px-4 py-1.5 rounded-full bg-slate-100 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                      Format yang didukung: PDF (Maks. 5MB)
                    </span>
                  </motion.div>
                ) : (
                  <motion.div 
                    key="file-uploaded"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="space-y-5 w-full max-w-sm"
                  >
                    <div className="w-16 h-16 rounded-full bg-emerald-50 flex items-center justify-center mx-auto text-emerald-600 shadow-inner">
                      <FileUp className="w-8 h-8" />
                    </div>
                    <div className="space-y-2">
                      <h3 className="text-lg font-bold font-outfit text-slate-800 truncate px-4">
                        {cvFile.name}
                      </h3>
                      <p className="text-sm text-slate-400">
                        {(cvFile.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                    <div className="flex items-center justify-center gap-3">
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-100/60 text-xs font-semibold text-emerald-700">
                        <CheckCircle2 className="w-3.5 h-3.5" /> CV Siap
                      </span>
                      <button 
                        onClick={handleRemoveFile}
                        className="p-1.5 rounded-full hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
                        title="Hapus file"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {fileError && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="absolute bottom-4 left-4 right-4 bg-rose-50 border border-rose-100 text-rose-600 px-4 py-2.5 rounded-xl text-xs font-medium flex items-center gap-2 justify-center"
                  onClick={(e) => e.stopPropagation()}
                >
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span>{fileError}</span>
                </motion.div>
              )}
            </div>

            {/* Right: Select Role & context */}
            <div className="space-y-6 flex flex-col justify-between">
              
              {/* Target Profession Box */}
              <div className="premium-card p-6 md:p-8 flex-grow space-y-4">
                <div className="flex items-center gap-2.5 text-slate-800">
                  <div className="p-2 rounded-lg bg-indigo-50 text-brand-600">
                    <Briefcase className="w-5 h-5" />
                  </div>
                  <label className="font-bold text-base font-outfit">Profesi Target</label>
                </div>
                
                <div className="relative">
                  <select
                    value={selectedRole}
                    onChange={(e) => setSelectedRole(e.target.value)}
                    required
                    className="premium-input appearance-none pr-10 font-sans text-sm focus:border-brand-500 cursor-pointer"
                  >
                    <option value="" disabled>Pilih profesi target Anda...</option>
                    {professions.map((role) => (
                      <option key={role} value={role}>{role}</option>
                    ))}
                  </select>
                  <div className="absolute right-3.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">
                    <ChevronDown className="w-5 h-5" />
                  </div>
                </div>
              </div>

              {/* Additional Context Box */}
              <div className="premium-card p-6 md:p-8 flex-grow space-y-4">
                <div className="flex items-center gap-2.5 text-slate-800">
                  <div className="p-2 rounded-lg bg-indigo-50 text-brand-600">
                    <FileText className="w-5 h-5" />
                  </div>
                  <label className="font-bold text-base font-outfit">
                    Konteks Tambahan <span className="text-slate-400 font-normal text-xs">(Opsional)</span>
                  </label>
                </div>
                
                <textarea
                  value={additionalContext}
                  onChange={(e) => setAdditionalContext(e.target.value)}
                  placeholder="Sebutkan kursus terbaru, soft skill, atau fokus keahlian khusus yang ingin Anda tonjolkan..."
                  rows={4}
                  className="premium-input resize-none font-sans text-sm min-h-[110px]"
                />
              </div>

            </div>
          </div>

          {/* Submit button container */}
          <div className="flex flex-col items-center justify-center space-y-6 pt-4">
            <motion.button
              type="submit"
              disabled={isAnalyzing}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`relative px-8 py-4 rounded-full font-bold font-outfit text-white shadow-xl shadow-brand-600/10 flex items-center justify-center gap-2.5 min-w-[220px] transition-all overflow-hidden ${
                isAnalyzing 
                  ? 'bg-brand-500 cursor-not-allowed shadow-inner' 
                  : 'bg-brand-600 hover:bg-brand-700 hover:shadow-brand-600/20'
              }`}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Menganalisis...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  <span>Scan CV Saya</span>
                </>
              )}
            </motion.button>
            <p className="max-w-2xl text-center text-[11px] sm:text-xs text-slate-400 font-sans leading-relaxed">
              Catatan: Hasil analisis bersifat estimasi berbasis sistem AI dan data keterampilan yang tersedia. Hindari mengupload CV yang memuat informasi sangat sensitif seperti nomor identitas, alamat lengkap, atau data finansial.
            </p>
          </div>
        </motion.form>
      </section>

      {/* "How It Works" info card */}
      <section className="max-w-6xl mx-auto">
        <motion.div 
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="premium-card p-6 md:p-10 flex flex-col lg:flex-row items-center gap-10 md:gap-16 bg-white/60 backdrop-blur-sm"
        >
          <div className="w-full lg:w-1/2 max-w-md lg:max-w-none rounded-2xl overflow-hidden shadow-2xl shadow-indigo-600/10 border border-slate-100 flex items-center justify-center bg-slate-950/20">
            <img 
              src={howItWorksImg} 
              alt="AI skill diagnostic holographic bar chart" 
              className="w-full h-auto object-cover hover:scale-[1.03] transition-transform duration-500"
            />
          </div>

          <div className="w-full lg:w-1/2 space-y-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-brand-50 text-brand-600 shadow-sm">
                <TrendingUp className="w-6 h-6 animate-pulse" />
              </div>
              <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 font-outfit">
                Cara kerja
              </h2>
            </div>
            
            <p className="text-slate-500 leading-relaxed font-sans text-sm sm:text-base">
              Kecerdasan buatan kami menganalisis pengalaman Anda secara mendalam berdasarkan ribuan deskripsi pekerjaan standar industri. Kami mengidentifikasi kesenjangan keterampilan teknis dan soft skill Anda secara akurat untuk memberikan peta jalan konkret demi meningkatkan karier Anda.
            </p>

            <div className="grid grid-cols-2 gap-4 pt-2">
              <div className="border border-slate-100 bg-slate-50/50 p-4 rounded-xl">
                <span className="text-2xl font-black text-brand-600 font-outfit">10k+</span>
                <p className="text-xs text-slate-400 mt-1 font-sans">Deskripsi Kerja Teranalisis</p>
              </div>
              <div className="border border-slate-100 bg-slate-50/50 p-4 rounded-xl">
                <span className="text-2xl font-black text-brand-600 font-outfit">Real-time</span>
                <p className="text-xs text-slate-400 mt-1 font-sans">Skill Path</p>
              </div>
            </div>
          </div>
        </motion.div>
      </section>
    </motion.main>
  );
}
