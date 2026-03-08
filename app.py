import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

# ==============================================================================
# KONFIGURASI HALAMAN (Sesuai Bab V.A.1 - Frontend)
# ==============================================================================
st.set_page_config(
    page_title="NEFO NLP - XAUUSD Decision Support",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CSS CUSTOM (Biar Kelihatan Pro)
# ==============================================================================
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1c1f26; padding: 10px; border-radius: 10px;}
    .reportview-container .main .block-container{padding-top: 2rem;}
    h1 {color: #FFA500;}
    h2 {color: #FFFFFF;}
    .stAlert {background-color: #262730;}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# KNOWLEDGE BASE (Sesuai Bab IV.A - Data Vektor Historis)
# ==============================================================================
knowledge_base = [
    {
        "keywords": ["inflasi", "raise rates", "hawkish", "tightening", "naik", "kuat"],
        "context": "Pidato The Fed - Maret 2023 (Hawkish)",
        "text": "The Federal Reserve remains committed to bringing inflation down to our 2 percent goal. We are prepared to raise rates further if appropriate.",
        "sentiment": "Hawkish"
    },
    {
        "keywords": ["easing", "pausing", "dovish", "assess", "turun", "lemah"],
        "context": "Pidato The Fed - Juni 2023 (Neutral/Dovish)",
        "text": "Inflation has shown signs of easing. The committee may consider pausing rate hikes to assess the impact of previous tightening.",
        "sentiment": "Dovish"
    },
    {
        "keywords": ["labor", "wage", "tight", "hawkish", "kerja", "upah"],
        "context": "Pidato The Fed - September 2023 (Hawkish)",
        "text": "The labor market remains tight. Wage growth is too high and needs to cool down to match productivity.",
        "sentiment": "Hawkish"
    },
    {
        "keywords": ["balanced", "maintain", "restrictive", "neutral"],
        "context": "Pidato The Fed - Desember 2023 (Neutral)",
        "text": "Risks to the economic outlook are roughly balanced. We can now maintain the policy rate at a restrictive level.",
        "sentiment": "Neutral"
    }
]

# ==============================================================================
# FUNGSI LOGIKA (RAG + Sentiment Analysis)
# ==============================================================================
def retrieve_context(query):
    """Simulasi Vector Retrieval (Bab II.2)"""
    query_lower = query.lower()
    best_match = None
    best_score = 0
    
    for item in knowledge_base:
        score = sum(1 for kw in item["keywords"] if kw.lower() in query_lower)
        if score > best_score:
            best_score = score
            best_match = item
    
    if best_match:
        return best_match
    return knowledge_base[0] # Default

def analyze_sentiment(news_text):
    """Simulasi LLM Inference (Bab V.C.3)"""
    text_lower = news_text.lower()
    hawkish_keywords = ["naik", "tinggi", "kuat", "above", "tight", "inflasi", "hijkah"]
    dovish_keywords = ["turun", "rendah", "lemah", "below", "easing", "pengangguran", "lunak"]
    
    hawkish_score = sum(1 for kw in hawkish_keywords if kw in text_lower)
    dovish_score = sum(1 for kw in dovish_keywords if kw in text_lower)
    
    if hawkish_score > dovish_score:
        return "HAWKISH 📉", "SELL", np.random.randint(80, 95), "Data ekonomi AS kuat. The Fed cenderung mempertahankan suku bunga tinggi. Tekanan jual pada emas meningkat."
    elif dovish_score > hawkish_score:
        return "DOVISH 📈", "BUY", np.random.randint(80, 95), "Data ekonomi AS lemah. The Fed mungkin melonggarkan kebijakan. Emas mendapat dukungan bullish."
    else:
        return "NEUTRAL ➡️", "WAIT", np.random.randint(50, 65), "Data ekonomi campuran. Pasar menunggu konfirmasi lebih lanjut."

# ==============================================================================
# TAMPILAN DASHBOARD (UI)
# ==============================================================================

# Header
st.title("📈 NEFO NLP - XAUUSD Decision Support System")
st.markdown("### Kelompok 4 | Foundations of Artificial Intelligence")
st.markdown("---")

# Sidebar Info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Microsoft_Fluent_Emojis_Security_Shield.svg/1200px-Microsoft_Fluent_Emojis_Security_Shield.svg.png", width=100)
    st.info("**Status Sistem:** 🟢 Online (Prototype Mode)")
    st.markdown("**Anggota:**\n- Jasfi Omarreza\n- Ferdinan Ferrel\n- Dewangga Sutra\n- Glen Kenneth\n- Nico Dwi Satrio")
    st.markdown("**Dosen:**\nDwinanda Kinanti Suci Sekarhati, S.Kom, M.T.I.")
    st.warning("**Catatan:**\nSesuai Bab VI, ini adalah DSS (Decision Support System), bukan bot trading otomatis.")

# Input Area
st.subheader("📰 Input Berita Makroekonomi Real-Time")
default_news = "Non-Farm Payrolls bertambah 300k, jauh di atas ekspektasi 180k. Upah rata-rata per jam naik 0.5%. Labor market remains tight."
input_news = st.text_area("Masukkan Teks Berita (NFP/CPI/The Fed):", value=default_news, height=100)

col1, col2 = st.columns([1, 5])
with col1:
    analyze_btn = st.button("🚀 ANALISIS SEKARANG", use_container_width=True)

# Proses Analisis
if analyze_btn:
    with st.spinner('🔍 Menghubungkan ke Vector Database...'):
        time.sleep(1) # Simulasi delay jaringan
        context = retrieve_context(input_news)
        
    with st.spinner('🤖 LLM Sedang Menganalisis Sentimen...'):
        time.sleep(1.5) # Simulasi proses AI
        sentimen, saran, prob, analisis_detail = analyze_sentiment(input_news)
        
    st.success("✅ Analisis Selesai!")
    
    # Hasil Dashboard
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric(label="🎯 Sentimen Pasar", value=sentimen)
    with c2:
        st.metric(label="📈 Probabilitas Dampak", value=f"{prob}%")
    with c3:
        st.metric(label="💡 Saran Aksi", value=saran)
    
    st.markdown("#### 🔍 Analisis Mendalam (RAG Context):")
    st.info(f"**Konteks Historis Terdekat:** {context['context']}\n\n**Isi:** {context['text']}")
    
    st.markdown("#### 📝 Kesimpulan AI:")
    st.write(analisis_detail)
    
    # Grafik Simulasi (Bab II.3 - Time Series Visualization)
    st.markdown("#### 📊 Simulasi Proyeksi Harga XAUUSD (M1)")
    chart_data = pd.DataFrame(
        np.random.randn(20, 1),
        columns=['Price Change']
    )
    if "SELL" in saran:
        chart_data = chart_data.cumsum() - 10 # Simulasi turun
    elif "BUY" in saran:
        chart_data = chart_data.cumsum() + 10 # Simulasi naik
    else:
        chart_data = chart_data.cumsum() # Sideways
        
    st.line_chart(chart_data)

else:
    st.info("👆 Klik tombol 'ANALISIS SEKARANG' untuk memproses berita.")

# Footer
st.markdown("---")
st.caption(f"© 2024 Kelompok 3 Binus University | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Prototype v1.0")
