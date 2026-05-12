import streamlit as st
import joblib
import os
import sys

# Tambahkan src ke system path jika perlu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import extract_skills, analyze_cv

def load_job_encoder():
    # Load job_encoder secara dinamis
    encoder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'job_encoder.pkl')
    try:
        # Load menggunakan joblib
        job_encoder = joblib.load(encoder_path)
        # Mengembalikan classes_ untuk options selectbox
        return list(job_encoder.classes_)
    except Exception as e:
        st.error(f"Gagal memuat job_encoder.pkl: {e}")
        return []

def main():
    st.set_page_config(page_title="Career Diagnostic System", page_icon="🎯", layout="centered")
    
    st.title("Career Diagnostic System 🎯")
    st.markdown("Masukkan CV Anda dan pilih profesi yang dituju untuk mendapatkan analisis kesenjangan skill.")
    
    # Load options secara dinamis
    profession_options = load_job_encoder()
    
    if not profession_options:
        st.warning("Data profesi belum tersedia. Silakan retrain model terlebih dahulu.")
        st.stop()
        
    cv_text = st.text_area("Masukkan Teks CV Anda", height=200, placeholder="Saya memiliki pengalaman sebagai data analyst menggunakan Python dan SQL...")
    
    # Gunakan classes dari job encoder sebagai options di selectbox
    target_profession = st.selectbox("Pilih Target Profesi", options=profession_options)
    
    if st.button("Analisis", type="primary"):
        if not cv_text.strip():
            st.warning("Mohon masukkan teks CV Anda terlebih dahulu.")
            return
            
        with st.spinner("Sedang mengekstrak skill dan menganalisis kesenjangan..."):
            # 1. Ekstrak skill menggunakan NER
            extracted_skills = extract_skills(cv_text)
            
            # 2. Analisis CV
            hasil_analisis = analyze_cv(extracted_skills, target_profession)
            
            # 3. Error Handling
            if "error" in hasil_analisis:
                st.error(f"Terjadi Kesalahan: {hasil_analisis['error']}")
                return  # Jangan lanjutkan jika ada error
                
            # Jika tidak ada error, tampilkan hasil
            st.success("Analisis Selesai!")
            
            # Tampilkan Score
            st.subheader("Skor Kesesuaian")
            score_percent = int(hasil_analisis['score'] * 100)
            st.progress(hasil_analisis['score'])
            st.write(f"**{score_percent}%** sesuai dengan profil {target_profession}")
            
            # Tampilkan Skills yang sudah dikuasai
            st.subheader("Skill yang Sudah Dikuasai (Matched)")
            if hasil_analisis['matched_skills']:
                st.write(", ".join(hasil_analisis['matched_skills']))
            else:
                st.write("- Belum ada skill yang sesuai -")
                
            # Tampilkan Gap Analysis
            st.subheader("Gap Analysis (Skill yang Perlu Dipelajari)")
            gap = hasil_analisis.get('gap', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🔴 Critical**")
                for s in gap.get('critical', []):
                    st.write(f"- {s}")
            with col2:
                st.markdown("**🟠 Important**")
                for s in gap.get('important', []):
                    st.write(f"- {s}")
            with col3:
                st.markdown("**🟢 Supplementary**")
                for s in gap.get('supplementary', []):
                    st.write(f"- {s}")

if __name__ == "__main__":
    main()
