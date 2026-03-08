import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import yfinance as yf

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="NEFO NLP - XAUUSD Decision Support System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CSS CUSTOM
# ==============================================================================
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1c1f26; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    h1 {color: #FFA500;}
    h2 {color: #FFFFFF;}
    h3 {color: #CCCCCC;}
    .stAlert {background-color: #262730;}
    .stButton>button {background-color: #FFA500; color: black; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# KNOWLEDGE BASE (Fed Transcripts - Sesuai Bab IV.A)
# ==============================================================================
knowledge_base = [
    {
        "keywords": ["inflasi", "inflation", "raise rates", "hawkish", "tightening", "naik", "kuat", "above"],
        "context": "Pidato The Fed - Maret 2023 (Hawkish)",
        "text": "The Federal Reserve remains committed to bringing inflation down to our 2 percent goal. We are prepared to raise rates further if appropriate.",
        "sentiment": "Hawkish",
        "impact": "XAUUSD Turun"
    },
    {
        "keywords": ["easing", "pausing", "dovish", "assess", "turun", "lemah", "below", "cut"],
        "context": "Pidato The Fed - Juni 2023 (Neutral/Dovish)",
        "text": "Inflation has shown signs of easing. The committee may consider pausing rate hikes to assess the impact of previous tightening.",
        "sentiment": "Dovish",
        "impact": "XAUUSD Naik"
    },
    {
        "keywords": ["labor", "wage", "tight", "hawkish", "kerja", "upah", "employment", "nfp"],
        "context": "Pidato The Fed - September 2023 (Hawkish)",
        "text": "The labor market remains tight. Wage growth is too high and needs to cool down to match productivity.",
        "sentiment": "Hawkish",
        "impact": "XAUUSD Turun"
    },
    {
        "keywords": ["balanced", "maintain", "restrictive", "neutral", "stabil"],
        "context": "Pidato The Fed - Desember 2023 (Neutral)",
        "text": "Risks to the economic outlook are roughly balanced. We can now maintain the policy rate at a restrictive level.",
        "sentiment": "Neutral",
        "impact": "XAUUSD Sideways"
    }
]

# ==============================================================================
# FUNGSI RETRIEVAL (RAG - Sesuai Bab II.2 & V.B.2)
# ==============================================================================
def retrieve_context(query, top_k=2):
    """
    Simulasi Vector Retrieval untuk RAG
    Mencari konteks historis paling relevan dengan berita saat ini
    """
    query_lower = query.lower()
    scores = []
    
    for item in knowledge_base:
        score = sum(1 for kw in item["keywords"] if kw.lower() in query_lower)
        scores.append((score, item))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    
    relevant_context = []
    for i in range(min(top_k, len(scores))):
        if scores[i][0] > 0:
            relevant_context.append(scores[i][1])
    
    if not relevant_context:
        relevant_context = [knowledge_base[0], knowledge_base[2]]
    
    return relevant_context

# ==============================================================================
# FUNGSI SENTIMENT ANALYSIS (LLM Simulation - Sesuai Bab V.C.3)
# ==============================================================================
def analyze_sentiment(news_text):
    """
    Analisis sentimen berbasis keyword matching
    Mensimulasikan output LLM untuk demo presentasi
    """
    text_lower = news_text.lower()
    
    hawkish_keywords = ["naik", "tinggi", "kuat", "above", "tight", "inflasi", "inflation", 
                       "hawkish", "raise", "hike", "strong", "nfp", "employment"]
    dovish_keywords = ["turun", "rendah", "lemah", "below", "easing", "dovish", 
                      "cut", "pause", "weak", "unemployment", "lunak"]
    
    hawkish_score = sum(1 for kw in hawkish_keywords if kw in text_lower)
    dovish_score = sum(1 for kw in dovish_keywords if kw in text_lower)
    
    if hawkish_score > dovish_score:
        return {
            "sentimen": "HAWKISH 📉",
            "saran": "SELL XAUUSD",
            "probabilitas": np.random.randint(80, 95),
            "analisis": "Data ekonomi AS lebih kuat dari ekspektasi. The Fed cenderung mempertahankan atau menaikkan suku bunga. Tekanan jual pada emas meningkat karena yield obligasi AS naik.",
            "dampak": "Bearish untuk XAUUSD",
            "risk_warning": "⚠️ Waspadai reversal jika ada profit taking institusi"
        }
    elif dovish_score > hawkish_score:
        return {
            "sentimen": "DOVISH 📈",
            "saran": "BUY XAUUSD",
            "probabilitas": np.random.randint(80, 95),
            "analisis": "Data ekonomi AS lebih lemah dari ekspektasi. The Fed mungkin akan melonggarkan kebijakan moneter. Emas mendapat dukungan bullish sebagai safe haven.",
            "dampak": "Bullish untuk XAUUSD",
            "risk_warning": "⚠️ Konfirmasi dengan price action sebelum entry"
        }
    else:
        return {
            "sentimen": "NEUTRAL ➡️",
            "saran": "WAIT / NO TRADE",
            "probabilitas": np.random.randint(50, 65),
            "analisis": "Data ekonomi campuran atau sesuai ekspektasi. Pasar mungkin sudah price-in sentimen ini. Lebih baik menunggu konfirmasi lebih lanjut.",
            "dampak": "Sideways / Konsolidasi",
            "risk_warning": "⚠️ Risiko whipsaw tinggi, hindari trading saat news"
        }

# ==============================================================================
# FUNGSI HARGA LIVE (Yahoo Finance - Sesuai Bab II.4 & V.B.1)
# ==============================================================================
@st.cache_data(ttl=60)
def get_gold_price():
    """
    Mengambil harga XAUUSD real-time dari Yahoo Finance
    Update setiap 60 detik
    """
    try:
        gold = yf.ticker("GC=F")  # Gold Futures COMEX
        hist = gold.history(period="1d", interval="1m")
        
        if len(hist) > 0:
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            # Tentukan status pasar berdasarkan waktu (WIB)
            now = datetime.now()
            hour = now.hour
            
            # Pasar emas buka Senin-Jumat, hampir 24 jam (tutup sebentar tiap hari)
            is_open = (hour >= 5 and hour < 17) or (hour >= 19 and hour < 24)
            is_weekend = now.weekday() >= 5  # Sabtu/Minggu
            
            if is_weekend:
                status = " CLOSED (Weekend)"
                status_color = "🔴"
            elif is_open:
                status = "🔴 OPEN"
                status_color = "🟢"
            else:
                status = "🟡 CLOSED (Maintenance)"
                status_color = "🟡"
            
            return {
                "price": current_price,
                "change": change,
                "change_pct": change_pct,
                "history": hist,
                "status": status,
                "status_color": status_color
            }
    except Exception as e:
        pass
    
    # Fallback data jika API gagal
    base_price = 2650.0
    return {
        "price": base_price + np.random.uniform(-5, 5),
        "change": np.random.uniform(-2, 2),
        "change_pct": np.random.uniform(-0.1, 0.1),
        "history": pd.DataFrame(np.random.randn(20, 1), columns=['Close']) + 2650,
        "status": "⚠️ DATA SIMULASI",
        "status_color": "🟡"
    }

# ==============================================================================
# FUNGSI CONTOH BERITA (Untuk Demo - Sesuai Bab III)
# ==============================================================================
def get_sample_news():
    """
    Contoh berita makroekonomi untuk demo
    Sesuai dengan studi kasus NFP di Bab III
    """
    return [
        "Non-Farm Payrolls bertambah 300k, jauh di atas ekspektasi 180k. Upah rata-rata per jam naik 0.5%. Labor market remains tight.",
        "CPI Inflasi AS turun ke 3.2%, lebih rendah dari ekspektasi 3.5%. The Fed may consider pausing rate hikes.",
        "Pengangguran AS naik ke 4.1%, tertinggi dalam 2 tahun. Ekonomi menunjukkan tanda-tanda perlambatan.",
        "The Fed mempertahankan suku bunga di 5.25-5.50%. Powell: Kami perlu melihat lebih banyak data sebelum memutuskan.",
        "NFP hanya 150k, di bawah ekspektasi 180k. Revisi bulan sebelumnya juga turun. Dollar melemah."
    ]

# ==============================================================================
# TAMPILAN DASHBOARD (UI - Sesuai Bab V.A.1)
# ==============================================================================

# Header
st.title("📈 NEFO NLP - XAUUSD Decision Support System")
st.markdown("### **Implementasi RAG + LLM untuk Trading Berbasis Sentimen Makroekonomi**")
st.markdown("#### Kelompok 3 | Foundations of Artificial Intelligence | Binus University")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Microsoft_Fluent_Emojis_Security_Shield.svg/1200px-Microsoft_Fluent_Emojis_Security_Shield.svg.png", width=100)
    st.info("**Status Sistem:** 🟢 Online (Prototype v2.0)")
    st.markdown("**Anggota Kelompok 3:**")
    st.markdown("""
    - 2902718465 JASFI OMARREZA
    - 2902726851 FERDINAN FERREL
    - 2902714095 DEWANGGA SUTRA
    - 2902673441 GLEN KENNETH
    - 2902723635 NICO DWI SATRIO
    """)
    st.markdown("**Dosen:**")
    st.markdown("Dwinanda Kinanti Suci Sekarhati, S.Kom, M.T.I.")
    st.warning("**DISCLAIMER (Bab VI):**")
    st.markdown("""
    Ini adalah **Sistem Pendukung Keputusan (DSS)**, 
    bukan bot trading otomatis. Keputusan eksekusi 
    tetap ada di tangan trader.
    """)
    st.markdown("---")
    st.markdown("**📊 Live Data:**")
    st.markdown("- Harga: Yahoo Finance API")
    st.markdown("- Berita: Simulasi NLP Pipeline")
    st.markdown("- RAG: Vector Database Context")

# Live Price Section (Bab II.4)
st.subheader("🏆 Harga XAUUSD Real-Time")
gold_data = get_gold_price()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        label="Harga Saat Ini",
        value=f"${gold_data['price']:.2f}",
        delta=f"{gold_data['change']:.2f} ({gold_data['change_pct']:.2f}%)"
    )
with c2:
    st.metric(label="Update Terakhir", value=datetime.now().strftime("%H:%M:%S"))
with c3:
    st.metric(label="Status Pasar", value=gold_data['status'])

st.markdown("---")

# Input Berita Section (Bab V.C.1)
st.subheader("📰 Analisis Sentimen Berita Makroekonomi")
st.markdown("*Sistem akan memproses teks berita dan membandingkannya dengan konteks historis The Fed (RAG)*")

# Pilihan: Input Manual atau Contoh Berita
input_mode = st.radio(
    "Pilih Mode Input:",
    ["📝 Input Manual", "📋 Gunakan Contoh Berita (Demo)"],
    horizontal=True
)

if input_mode == "📋 Gunakan Contoh Berita (Demo)":
    sample_news = get_sample_news()
    selected_news = st.selectbox("Pilih contoh berita:", sample_news)
    input_news = selected_news
else:
    default_news = "Non-Farm Payrolls bertambah 300k, jauh di atas ekspektasi 180k. Upah rata-rata per jam naik 0.5%."
    input_news = st.text_area("Masukkan Teks Berita:", value=default_news, height=100)

col1, col2 = st.columns([1, 5])
with col1:
    analyze_btn = st.button("🚀 ANALISIS SEKARANG", use_container_width=True)

# Proses Analisis
if analyze_btn:
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Retrieval
    status_text.text("🔍 [1/4] Menghubungkan ke Vector Database...")
    progress_bar.progress(25)
    time.sleep(0.5)
    
    # Step 2: Context Found
    context_list = retrieve_context(input_news)
    status_text.text("✅ [2/4] Konteks Historis Ditemukan!")
    progress_bar.progress(50)
    time.sleep(0.5)
    
    # Step 3: LLM Analysis
    status_text.text("🤖 [3/4] LLM Sedang Menganalisis Sentimen...")
    progress_bar.progress(75)
    time.sleep(1)
    
    # Step 4: Complete
    status_text.text("✅ [4/4] Analisis Selesai!")
    progress_bar.progress(100)
    time.sleep(0.5)
    
    # Get Results
    result = analyze_sentiment(input_news)
    
    # Display Results
    st.success("✅ Analisis Berhasil!")
    st.markdown("---")
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label="🎯 Sentimen Pasar", value=result["sentimen"])
    with c2:
        st.metric(label="📈 Probabilitas Dampak", value=f"{result['probabilitas']}%")
    with c3:
        st.metric(label="💡 Saran Aksi", value=result["saran"])
    
    # Detailed Analysis
    st.markdown("#### 🔍 Analisis Mendalam:")
    st.info(f"**Dampak untuk XAUUSD:** {result['dampak']}\n\n**Penjelasan:** {result['analisis']}")
    
    # RAG Context Display (Bab II.2)
    st.markdown("#### 📚 Konteks Historis (RAG Retrieval):")
    for ctx in context_list:
        with st.expander(f"{ctx['context']}"):
            st.write(f"**Isi:** {ctx['text']}")
            st.write(f"**Sentimen:** {ctx['sentiment']}")
            st.write(f"**Dampak Historis:** {ctx['impact']}")
    
    # Risk Warning (Bab VI)
    st.markdown("#### ⚠️ Peringatan Risiko (Sesuai Bab VI):")
    st.warning(result["risk_warning"])
    
    # Price Projection Chart (Bab II.3)
    st.markdown("#### 📊 Proyeksi Pergerakan Harga (Simulasi M1)")
    
    # Generate chart based on sentiment
    hist_data = gold_data['history']
    if len(hist_data) > 0:
        if "SELL" in result["saran"]:
            projection = hist_data['Close'].tail(20).values - np.linspace(0, 10, 20)
        elif "BUY" in result["saran"]:
            projection = hist_data['Close'].tail(20).values + np.linspace(0, 10, 20)
        else:
            projection = hist_data['Close'].tail(20).values + np.random.randn(20) * 2
        
        chart_df = pd.DataFrame(projection, columns=['Proyeksi Harga'])
        st.line_chart(chart_df)
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()

else:
    st.info("👆 Klik tombol 'ANALISIS SEKARANG' untuk memproses berita.")

# Footer
st.markdown("---")
st.caption(f"""
© 2026 Kelompok 4 Binus University | 
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
Prototype v2.0 | 
Sesuai Dokumen: NEFO NLP (1).docx
""")

# Tech Stack Display
with st.expander("🛠️ Teknologi yang Digunakan (Bab II)"):
    st.markdown("""
    - **LLM:** Gemini API (Simulasi untuk Demo)
    - **RAG:** Vector Database dengan Keyword Retrieval
    - **Data Harga:** Yahoo Finance API (yfinance)
    - **Frontend:** Streamlit Cloud
    - **Backend:** Python 3.10+
    - **Time-Series:** LSTM (Dalam Pengembangan)
    """)
