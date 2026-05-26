import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, X, Send, Sparkles, Loader2, Minimize2 } from 'lucide-react';
import { API_BASE_URL } from '../config/api';

export default function ChatWidget({ analyzedRole, currentRoleData, currentView }) {
  const [isOpen, setIsOpen] = useState(false);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      text: 'Halo! Saya Asisten Karir AI. Ada yang bisa saya bantu tentang keahlian, celah kemampuan, atau jalur peningkatan karier Anda?',
      sender: 'ai',
      time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })
    }
  ]);

  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom on new messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
      setTimeout(scrollToBottom, 100);
    }
  }, [messages, isOpen]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || loading) return;

    const userMsgText = inputText.trim();
    setInputText('');
    
    // Add User Message
    const userMessage = {
      id: `user-${Date.now()}`,
      text: userMsgText,
      sender: 'user',
      time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    // Prepare analysis context if we have one
    let payload = { message: userMsgText };
    
    if (analyzedRole && currentRoleData && currentRoleData.matchScore > 0) {
      // Map skills & gaps to match the api expect payload
      const mappedSkills = [
        ...currentRoleData.skills.map(s => ({ name: s, status: 'match', category: 'critical' })),
        ...currentRoleData.gaps.map(g => ({ name: g.title, status: 'gap', category: g.tier.toLowerCase() }))
      ];

      payload = {
        message: userMsgText,
        profession_name: analyzedRole,
        score: currentRoleData.matchScore,
        skill_analysis: mappedSkills
      };
    }

    try {
      const response = await fetch(`${API_BASE_URL}/analysis/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        if (response.status === 429) {
          throw new Error("Layanan terlalu sibuk (Limit 10 chat per menit). Silakan coba lagi beberapa saat lagi.");
        }
        const errorData = await response.json();
        throw new Error(errorData.detail || errorData.message || "Gagal mendapatkan balasan dari AI.");
      }

      const data = await response.json();
      
      // Add AI reply
      const aiReply = {
        id: `ai-${Date.now()}`,
        text: data.reply,
        sender: 'ai',
        time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, aiReply]);

    } catch (error) {
      console.error(error);
      const systemError = {
        id: `sys-${Date.now()}`,
        text: `Terjadi kendala: ${error.message}. Coba lagi beberapa saat lagi.`,
        sender: 'system',
        time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, systemError]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed bottom-6 right-6 z-50 font-sans">
      
      {/* 1. floating Action Trigger FAB */}
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            key="chat-fab"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.92 }}
            onClick={() => setIsOpen(true)}
            className="w-14 h-14 rounded-full bg-gradient-to-tr from-brand-600 to-indigo-600 text-white flex items-center justify-center shadow-xl shadow-brand-600/35 hover:shadow-brand-600/50 hover:brightness-105 transition-all relative group cursor-pointer border border-brand-500/20"
          >
            {/* Pulsing ring around button */}
            <span className="absolute -inset-1 rounded-full bg-brand-500/20 animate-ping group-hover:animate-none pointer-events-none -z-10" />
            <MessageSquare className="w-6 h-6 fill-white/10" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* 2. Chat Window Popover */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            key="chat-popup"
            initial={{ scale: 0.85, opacity: 0, y: 50, x: 30 }}
            animate={{ scale: 1, opacity: 1, y: 0, x: 0 }}
            exit={{ scale: 0.85, opacity: 0, y: 50, x: 30 }}
            transition={{ type: 'spring', stiffness: 260, damping: 25 }}
            className="w-[360px] sm:w-[380px] h-[520px] bg-white rounded-3xl border border-slate-100 shadow-[0_20px_60px_rgba(0,0,0,0.08)] flex flex-col overflow-hidden relative"
          >
            {/* Branded Header */}
            <div className="px-5 py-4 bg-white text-slate-800 flex items-center justify-between border-b border-slate-100 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-brand-50 flex items-center justify-center shadow-inner">
                  <Sparkles className="w-4.5 h-4.5 text-brand-600" />
                </div>
                <div className="text-left">
                  <h4 className="text-sm font-bold font-outfit text-slate-800 tracking-tight leading-none">Asisten Karir AI</h4>
                  <div className="flex items-center gap-1.5 mt-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="text-[10px] text-slate-400 font-sans font-medium">Online</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1.5 rounded-full hover:bg-slate-50 text-slate-400 hover:text-slate-600 transition-all cursor-pointer"
                  title="Perkecil"
                >
                  <Minimize2 className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1.5 rounded-full hover:bg-slate-50 text-slate-400 hover:text-slate-600 transition-all cursor-pointer"
                  title="Tutup"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Context Badge (Dynamic scan indicators) */}
            {analyzedRole && currentRoleData && currentRoleData.matchScore > 0 && (
              <div className="px-4 py-2 bg-slate-50 border-b border-slate-100 flex items-center justify-between gap-2 text-slate-500 text-[10px] font-sans">
                <div className="truncate max-w-[200px]">
                  Konteks: <strong>{analyzedRole}</strong> ({currentRoleData.matchScore}%)
                </div>
                <span className="px-2 py-0.5 rounded bg-brand-50 text-brand-700 font-bold border border-brand-100/50 uppercase">
                  Aktif
                </span>
              </div>
            )}

            {/* Message Area */}
            <div className="flex-grow p-4 overflow-y-auto space-y-4 bg-slate-50/50 scrollbar-thin">
              {messages.map((msg) => {
                const isUser = msg.sender === 'user';
                const isSystem = msg.sender === 'system';

                if (isSystem) {
                  return (
                    <div key={msg.id} className="flex justify-center my-2 max-w-xs mx-auto">
                      <div className="bg-rose-50 border border-rose-100 text-rose-600 px-3.5 py-2 rounded-2xl text-[11px] font-sans text-center leading-normal shadow-sm">
                        {msg.text}
                      </div>
                    </div>
                  );
                }

                return (
                  <div
                    key={msg.id}
                    className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[78%] px-4 py-3 rounded-2xl text-xs sm:text-sm font-sans shadow-sm leading-relaxed ${
                        isUser
                          ? 'bg-gradient-to-tr from-brand-600 to-brand-500 text-white rounded-tr-none'
                          : 'bg-white text-slate-700 border border-slate-100/60 rounded-tl-none'
                      }`}
                    >
                      <p className="whitespace-pre-line text-left">{msg.text}</p>
                      <span
                        className={`block text-[9px] mt-1.5 text-right font-medium ${
                          isUser ? 'text-indigo-200' : 'text-slate-400'
                        }`}
                      >
                        {msg.time}
                      </span>
                    </div>
                  </div>
                );
              })}
              
              {/* Typing Loader bubble */}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-white border border-slate-100/60 px-4 py-3 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin text-brand-600" />
                    <span className="text-xs text-slate-400 font-sans font-medium">Asisten sedang mengetik...</span>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>

            {/* Input Form Footer */}
            <form
              onSubmit={handleSend}
              className="p-3 bg-white border-t border-slate-100 flex items-center gap-2"
            >
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Tanyakan rekomendasi karir..."
                disabled={loading}
                className="flex-grow bg-slate-50 border border-slate-200/80 rounded-xl px-4 py-2.5 text-xs sm:text-sm font-sans focus:outline-none focus:border-brand-500 focus:bg-white transition-all disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={!inputText.trim() || loading}
                className={`p-2.5 rounded-xl text-white shadow-md flex items-center justify-center transition-all cursor-pointer ${
                  !inputText.trim() || loading
                    ? 'bg-slate-200 text-slate-400 cursor-not-allowed shadow-none'
                    : 'bg-brand-600 hover:bg-brand-700 active:scale-95 shadow-brand-600/10'
                }`}
              >
                <Send className="w-4 h-4" />
              </button>
            </form>

          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
