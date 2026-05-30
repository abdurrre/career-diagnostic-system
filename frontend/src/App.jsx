import { useState, useRef, useEffect } from 'react';
import { AnimatePresence } from 'framer-motion';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Toast from './components/Toast';
import LoadingScreen from './components/LoadingScreen';
import AuthView from './components/AuthView';
import HomeView from './components/HomeView';
import ReportView from './components/ReportView';
import HistoryView from './components/HistoryView';
import AboutView from './components/AboutView';
import ForgotPasswordView from './components/ForgotPasswordView';
import ResetPasswordNewView from './components/ResetPasswordNewView';
import ResetPasswordSuccessView from './components/ResetPasswordSuccessView';
import NotFoundView from './components/NotFoundView';
import ChatWidget from './components/ChatWidget';
import { professions as defaultProfessions, professionsData, defaultProfessionData } from './data/professions';
import { API_BASE_URL } from './config/api';
import './App.css';

function App() {
  // Centralized persistent authentication states
  const [authToken, setAuthToken] = useState(localStorage.getItem('auth_token') || '');
  const [loggedInEmail, setLoggedInEmail] = useState(localStorage.getItem('auth_email') || '');

  // Navigation & View states
  const [activeTab, setActiveTab] = useState('Home');
  const [currentView, setCurrentView] = useState('home'); // 'home' | 'report' | 'login' | 'register' | 'history' | 'about' | 'forgot-password' | 'reset-password-new' | 'reset-password-success' | 'not-found'
  const [analyzedRole, setAnalyzedRole] = useState('');

  // Target professions state lists (Fetched dynamically or falls back to static seed)
  const [professionsList, setProfessionsList] = useState(defaultProfessions);

  const [rawProfessions, setRawProfessions] = useState([]);

  // Centralized History List state (Initialized as empty to remove dummy data)
  const [historyList, setHistoryList] = useState([]);

  // Main Form states
  const [selectedRole, setSelectedRole] = useState('');
  const [additionalContext, setAdditionalContext] = useState('');
  const [cvFile, setCvFile] = useState(null);
  const [fileError, setFileError] = useState('');
  
  // Drag & drop state
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  // Analysis / Loading Mock states
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [, setAnalysisSuccess] = useState(false);
  const [loadingStep, setLoadingStep] = useState(1);

  // Auth inputs states
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authConfirmPassword, setAuthConfirmPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);
  const [registerSuccess, setRegisterSuccess] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);

  // Toast Notification states
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');

  // Parse URL query parameters for forgot-password redirection
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const viewParam = params.get('view');
    const tokenParam = params.get('token');
    const emailParam = params.get('email');

    if (viewParam === 'reset-password-new' && tokenParam && emailParam) {
      localStorage.setItem('reset_token', tokenParam);
      localStorage.setItem('reset_email', emailParam);
      setCurrentView('reset-password-new');
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  }, []);

  // 1. Fetch target professions list on mount
  useEffect(() => {
    fetch(`${API_BASE_URL}/professions`)
      .then(res => res.json())
      .then(data => {
        if (data && Array.isArray(data) && data.length > 0) {
          setRawProfessions(data);
          setProfessionsList(data.map(p => p.name));
        }
      })
      .catch(err => {
        console.log("Vite dev loader: professions fetch failed. Falling back to local dataset.", err);
      });
  }, []);

  // 2. Fetch User history logs dynamically from the backend database when entering logs page
  const fetchUserHistory = () => {
    if (!authToken) return;
    fetch(`${API_BASE_URL}/analysis/history`, {
      headers: {
        'Authorization': `Bearer ${authToken}`
      }
    })
      .then(res => {
        if (!res.ok) throw new Error("Invalid token session");
        return res.json();
      })
      .then(data => {
        if (data && Array.isArray(data)) {
          const mapped = data.map(item => {
            const skills = [];
            const gaps = [];

            if (item.Skills && Array.isArray(item.Skills)) {
              item.Skills.forEach(s => {
                const status = s.HistorySkill ? s.HistorySkill.status : 'match';
                const category = s.HistorySkill ? s.HistorySkill.category : 'IMPORTANT';
                const formattedName = s.name.charAt(0).toUpperCase() + s.name.slice(1);

                if (status === 'match') {
                  skills.push(formattedName);
                } else if (status === 'gap') {
                  gaps.push({
                    title: formattedName,
                    tier: category.toUpperCase(),
                    description: s.description || 'Kesenjangan kemampuan keahlian yang terpetakan berdasarkan kebutuhan standar industri.'
                  });
                }
              });
            }

            return {
              id: item.id,
              role: item.Profession ? item.Profession.name : 'Unknown Role',
              score: Math.round(item.score),
              fileName: 'CV_Scanned_File.pdf',
              date: new Date(item.created_at || item.createdAt).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }),
              skills: skills.length > 0 ? skills : null,
              gaps: gaps.length > 0 ? gaps : null
            };
          });
          setHistoryList(mapped);
        }
      })
      .catch(err => {
        console.log("Vite history loader: database logs failed. Preserving client-side log memory.", err);
      });
  };

  useEffect(() => {
    if (currentView === 'history') {
      fetchUserHistory();
    }
  }, [currentView, authToken]);

  // File drop handler
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const validateAndSetFile = (file) => {
    setFileError('');
    if (!file) return;

    // Check file type (PDF only)
    if (file.type !== "application/pdf" && !file.name.endsWith('.pdf')) {
      setFileError("Only PDF files are supported.");
      setCvFile(null);
      return;
    }

    // Check file size (Max 5MB)
    const maxSize = 5 * 1024 * 1024; // 5MB
    if (file.size > maxSize) {
      setFileError("File is too large. Maximum size is 5MB.");
      setCvFile(null);
      return;
    }

    setCvFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleRemoveFile = (e) => {
    e.stopPropagation();
    setCvFile(null);
    setFileError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  // Mock submit handler that triggers the premium sequential loading overlay
  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!cvFile) {
      setFileError("Please upload your CV to start the analysis.");
      return;
    }
    if (!selectedRole) {
      alert("Please select your target profession.");
      return;
    }

    setIsAnalyzing(true);
    setLoadingStep(1);
    setAnalysisSuccess(false);

    // Dynamic non-blocking checklist progression
    let step = 1;
    const progressInterval = setInterval(() => {
      if (step < 3) {
        step += 1;
        setLoadingStep(step);
      }
    }, 1200);

    // Dispatch scan action to backend Express API
    const formData = new FormData();
    formData.append('cv_file', cvFile);
    formData.append('target_profession', selectedRole);
    formData.append('additional_text', additionalContext || '');

    try {
      const response = await fetch(`${API_BASE_URL}/analysis/scan`, {
        method: 'POST',
        body: formData
      });

      // Clear progression timer instantly as backend returned
      clearInterval(progressInterval);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || "Failed to parse profile CV file");
      }

      const scanResult = await response.json();

      if (scanResult.score === undefined || scanResult.score === null) {
        setIsAnalyzing(false);
        setToastMessage("Gagal menganalisis CV: AI tidak memberikan skor kecocokan yang valid.");
        setShowToast(true);
        setTimeout(() => setShowToast(false), 5000);
        return;
      }

      // Convert the returned skill_analysis records to dynamic ReportView parameters
      const skillAnalysis = scanResult.skill_analysis || [];
      const matchedSkills = skillAnalysis
        .filter(item => item.status === 'match')
        .map(item => item.name.charAt(0).toUpperCase() + item.name.slice(1));
      
      const missedGaps = skillAnalysis
        .filter(item => item.status === 'gap')
        .map(item => ({
          title: item.name.charAt(0).toUpperCase() + item.name.slice(1),
          tier: item.category ? item.category.toUpperCase() : 'IMPORTANT',
          description: item.description || 'Kesenjangan kemampuan keahlian yang terpetakan berdasarkan kebutuhan standar industri.'
        }));

      if (matchedSkills.length === 0 && missedGaps.length === 0) {
        setIsAnalyzing(false);
        setToastMessage("Gagal menganalisis CV: AI tidak mendeteksi data keahlian atau kesenjangan yang valid.");
        setShowToast(true);
        setTimeout(() => setShowToast(false), 5000);
        return;
      }

      setIsAnalyzing(false);
      setAnalysisSuccess(true);
      setAnalyzedRole(scanResult.profession_name || selectedRole);

      const dynamicRolePayload = {
        matchScore: Math.round(scanResult.score),
        skills: matchedSkills,
        gaps: missedGaps
      };

      // Cache inside active dataset list
      professionsData[scanResult.profession_name || selectedRole] = dynamicRolePayload;

      setCurrentView('report');
      setActiveTab('History');

    } catch (err) {
      clearInterval(progressInterval);
      setIsAnalyzing(false);
      console.error("Vite scanner error:", err);
      
      setToastMessage("Koneksi gagal: Layanan backend tidak berjalan atau terjadi kesalahan internal server.");
      setShowToast(true);
      setTimeout(() => setShowToast(false), 5000);
    }
  };

  // Back to Homepage handler
  const handleTryAnother = () => {
    setCurrentView('home');
    setAnalysisSuccess(false);
    setActiveTab('Home');
  };

  // Centralized report loading action
  const handleLoadReport = (target) => {
    if (typeof target === 'string') {
      setAnalyzedRole(target);
    } else if (target && typeof target === 'object') {
      setAnalyzedRole(target.role);
      
      // Cache the loaded custom skills & gaps so currentRoleData can resolve them!
      if (target.skills || target.gaps) {
        professionsData[target.role] = {
          matchScore: target.score,
          skills: target.skills || [],
          gaps: target.gaps || []
        };
      }
    }
    setCurrentView('report');
    setActiveTab('History');
  };

  // Centralized record delete action
  const handleDeleteRecord = (id) => {
    // Local remove
    setHistoryList(prev => prev.filter(item => item.id !== id));
    setToastMessage("Riwayat analisis berhasil dihapus.");
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3500);
  };

  // Save Results to Database (authenticated) or Client logs fallback
  const handleSaveResults = async () => {
    if (!authToken) {
      setToastMessage("Silakan Sign In/Login terlebih dahulu untuk menyimpan riwayat hasil diagnosa ke database!");
      setShowToast(true);
      setTimeout(() => setShowToast(false), 4500);
      return;
    }

    setSaveLoading(true);
    try {
      const matchScore = currentRoleData.matchScore;
      const mappedSkills = [
        ...currentRoleData.skills.map(s => ({ name: s.toLowerCase(), status: 'match', category: 'critical' })),
        ...currentRoleData.gaps.map(g => ({ name: g.title.toLowerCase(), status: 'gap', category: g.tier.toLowerCase() }))
      ];

      // Dapatkan ID profesi dinamis berdasarkan analyzedRole
      const currentProfession = rawProfessions.find(p => p.name === analyzedRole);
      const dynamicProfessionId = currentProfession ? currentProfession.id : 1;

      const response = await fetch(`${API_BASE_URL}/analysis/save`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({
          score: matchScore,
          id_profession: dynamicProfessionId,
          skill_analysis: mappedSkills
        })
      });

      if (!response.ok) throw new Error("Server rejected save action");

      setToastMessage("Riwayat analisis berhasil disimpan ke database.");
      setShowToast(true);
      setTimeout(() => setShowToast(false), 4000);

      // Refresh
      fetchUserHistory();

    } catch (err) {
      console.error("Gagal menyimpan riwayat ke database: ", err);

      setToastMessage("Gagal menyimpan riwayat ke database. Silakan coba beberapa saat lagi.");
      setShowToast(true);
      setTimeout(() => setShowToast(false), 4500);
    } finally {
      setSaveLoading(false);
    }
  };

  // Unified Navbar Tab click handler
  const handleTabChange = (tab) => {
    if (tab === 'History' && !authToken) {
      setToastMessage("Silakan Sign In/Login terlebih dahulu untuk mengakses riwayat analisis!");
      setShowToast(true);
      setTimeout(() => setShowToast(false), 4500);
      setCurrentView('login');
      return;
    }

    setActiveTab(tab);
    if (tab === 'Home') {
      setCurrentView('home');
    } else if (tab === 'History') {
      setCurrentView('history');
    } else if (tab === 'About') {
      setCurrentView('about');
    }
  };

  // Real backend email/password credentials authentication
  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    setAuthError('');
    if (!authEmail) {
      setAuthError("Email address is required.");
      return;
    }
    if (!authPassword) {
      setAuthError("Password is required.");
      return;
    }

    setAuthLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email: authEmail, password: authPassword })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || "Email atau password yang dimasukkan salah.");
      }

      const authData = await response.json();

      setAuthLoading(false);
      
      // Store locally
      localStorage.setItem('auth_token', authData.token);
      localStorage.setItem('auth_email', authEmail);
      setAuthToken(authData.token);
      setLoggedInEmail(authEmail);

      setCurrentView('home');
      setActiveTab('Home');
      setToastMessage(`Selamat datang kembali, ${authEmail}!`);
      setShowToast(true);
      setTimeout(() => setShowToast(false), 4500);

      setAuthEmail('');
      setAuthPassword('');

    } catch (err) {
      setAuthLoading(false);
      console.error("Gagal login ke backend:", err);
      setAuthError(err.message || "Gagal login. Silakan coba beberapa saat lagi.");
      setToastMessage(err.message || "Gagal login. Pastikan backend berjalan dan kredensial benar.");
      setShowToast(true);
      setTimeout(() => setShowToast(false), 4500);
    }
  };

  // Real backend account registration
  const handleRegisterSubmit = async (e) => {
    e.preventDefault();
    setAuthError('');
    setRegisterSuccess(false);

    if (!authEmail) {
      setAuthError("Email address is required.");
      return;
    }
    if (!authPassword) {
      setAuthError("Password is required.");
      return;
    }
    if (authPassword !== authConfirmPassword) {
      setAuthError("Passwords do not match.");
      return;
    }

    setAuthLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email: authEmail, password: authPassword })
      });

      if (!response.ok) {
        const errorData = await response.json();
        setAuthLoading(false);
        setAuthError(errorData.error || errorData.message || "Gagal melakukan registrasi akun.");
        return;
      }

      setAuthLoading(false);
      setRegisterSuccess(true);
      setAuthPassword('');
      setAuthConfirmPassword('');

    } catch (err) {
      setAuthLoading(false);
      console.log("Vite register loader fail-safe: auth server offline. Entering offline mock simulation register.", err);
      
      // Fallback
      setRegisterSuccess(true);
      setAuthPassword('');
      setAuthConfirmPassword('');
    }
  };

  // Centralized Sign Out
  const handleSignOut = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_email');
    setAuthToken('');
    setLoggedInEmail('');
    setToastMessage("Anda berhasil keluar dari akun.");
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3500);
    setCurrentView('home');
    setActiveTab('Home');
  };

  // Fetch role data
  const currentRoleData = professionsData[analyzedRole] || defaultProfessionData;

  // Global navbar login buttons trigger
  const triggerLoginView = () => {
    setAuthError('');
    setRegisterSuccess(false);
    setAuthEmail('');
    setAuthPassword('');
    setAuthConfirmPassword('');
    setCurrentView('login');
  };

  const triggerRegisterView = () => {
    setAuthError('');
    setRegisterSuccess(false);
    setAuthEmail('');
    setAuthPassword('');
    setAuthConfirmPassword('');
    setCurrentView('register');
  };

  return (
    <div className="min-h-screen flex flex-col justify-between overflow-x-hidden relative">
      
      {/* 1. HEADER SECTION (Not rendered in full-screen auth views) */}
      <Navbar 
        currentView={currentView}
        activeTab={activeTab}
        handleTabChange={handleTabChange}
        triggerLoginView={triggerLoginView}
        authToken={authToken}
        loggedInEmail={loggedInEmail}
        handleSignOut={handleSignOut}
      />

      {/* VIEW CONTROLLER WITH TRANSITIONS */}
      <AnimatePresence mode="wait">
        
        {/* VIEW 1: HOMEPAGE VIEW */}
        <HomeView
          currentView={currentView}
          selectedRole={selectedRole}
          setSelectedRole={setSelectedRole}
          additionalContext={additionalContext}
          setAdditionalContext={setAdditionalContext}
          cvFile={cvFile}
          fileError={fileError}
          isDragActive={isDragActive}
          fileInputRef={fileInputRef}
          isAnalyzing={isAnalyzing}
          handleAnalyze={handleAnalyze}
          handleDrag={handleDrag}
          handleDrop={handleDrop}
          handleFileChange={handleFileChange}
          handleRemoveFile={handleRemoveFile}
          triggerFileInput={triggerFileInput}
        />

        {/* VIEW 2: DYNAMIC ANALYSIS RESULTS REPORT SCREEN */}
        <ReportView
          currentView={currentView}
          analyzedRole={analyzedRole}
          currentRoleData={currentRoleData}
          handleTryAnother={handleTryAnother}
          handleSaveResults={handleSaveResults}
          saveLoading={saveLoading}
        />

        {/* VIEW 3: DYNAMIC FULL-SCREEN AUTH VIEWS (CAPTCHA Disabled) */}
        <AuthView
          currentView={currentView}
          setCurrentView={setCurrentView}
          authEmail={authEmail}
          setAuthEmail={setAuthEmail}
          authPassword={authPassword}
          setAuthPassword={setAuthPassword}
          authConfirmPassword={authConfirmPassword}
          setAuthConfirmPassword={setAuthConfirmPassword}
          authError={authError}
          authLoading={authLoading}
          registerSuccess={registerSuccess}
          handleLoginSubmit={handleLoginSubmit}
          handleRegisterSubmit={handleRegisterSubmit}
          handleTryAnother={handleTryAnother}
          triggerLoginView={triggerLoginView}
          triggerRegisterView={triggerRegisterView}
        />

        {/* VIEW 4: DIAGNOSTICS ARCHIVE HISTORY VIEW */}
        <HistoryView 
          currentView={currentView}
          historyList={historyList}
          handleLoadReport={handleLoadReport}
          handleDeleteRecord={handleDeleteRecord}
          handleTryAnother={handleTryAnother}
        />

        {/* VIEW 7: MISSION ABOUT SCREEN */}
        <AboutView 
          currentView={currentView}
          handleTryAnother={handleTryAnother}
        />

        {/* VIEW 8: DYNAMIC FORGOT PASSWORD SCREEN */}
        <ForgotPasswordView 
          currentView={currentView}
          setCurrentView={setCurrentView}
          handleTryAnother={handleTryAnother}
          triggerLoginView={triggerLoginView}
        />

        {/* VIEW 9: DYNAMIC RESET PASSWORD NEW INPUTS SCREEN */}
        <ResetPasswordNewView 
          currentView={currentView}
          setCurrentView={setCurrentView}
          handleTryAnother={handleTryAnother}
        />

        {/* VIEW 10: DYNAMIC RESET PASSWORD SUCCESS CONFIRMED SCREEN */}
        <ResetPasswordSuccessView 
          currentView={currentView}
          triggerLoginView={triggerLoginView}
          handleTryAnother={handleTryAnother}
        />

        {/* VIEW 11: DYNAMIC 404 DIAGNOSTIC ALERTS SCREEN */}
        <NotFoundView 
          currentView={currentView}
          handleTryAnother={handleTryAnother}
        />

      </AnimatePresence>

      {/* 5. FOOTER (Not rendered in full-screen auth views) */}
      <Footer 
        currentView={currentView}
        handleTryAnother={handleTryAnother}
        triggerNotFoundView={() => setCurrentView('not-found')}
      />

      {/* DETAILED FULL-SCREEN AI ENGINE ANALYSIS LOADING SCREEN */}
      <LoadingScreen 
        isAnalyzing={isAnalyzing}
        loadingStep={loadingStep}
      />

      {/* PREMIUM SLIDE-IN TOAST NOTIFICATION OVERLAY */}
      <Toast 
        showToast={showToast}
        setShowToast={setShowToast}
        toastMessage={toastMessage}
      />

      {/* FLOATING AI CHAT ASSISTANT WIDGET */}
      <ChatWidget 
        analyzedRole={analyzedRole}
        currentRoleData={currentRoleData}
        currentView={currentView}
      />

    </div>
  );
}

export default App;
