import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import base64
from io import BytesIO

# Fungsi untuk mengatur tema yang responsif
def setup_theme():
    # Gunakan CSS yang beradaptasi dengan tema apapun
    st.markdown(
        """
        <style>
        /* Style yang beradaptasi dengan tema light dan dark */
        .card {
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Responsif untuk tema gelap dan terang */
        .header-text {
            font-weight: bold;
            font-size: 2rem;
            text-align: center;
        }
        
        .subheader-text {
            font-weight: 500;
            text-align: center;
            opacity: 0.8;
        }
        
        .card-title {
            font-weight: 600;
            font-size: 1.2rem;
        }
        
        .highlight {
            background-color: rgba(59, 130, 246, 0.1);
            padding: 10px 15px;
            border-radius: 5px;
            font-weight: 600;
        }
        
        /* Menyembunyikan footer default */
        footer {
            visibility: hidden;
        }
        
        /* Membuat padding untuk tampilan yang lebih baik */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Fungsi untuk membuat kartu dengan efek bayangan yang responsif terhadap tema
def card_with_shadow(content):
    """
    Membuat kartu dengan efek bayangan menggunakan HTML dan CSS.
    Content akan langsung di-render sebagai HTML di dalam kartu.
    """
    st.markdown(
        f"""
        <div class="card">
            {content}
        </div>
        """, 
        unsafe_allow_html=True
    )

# Fungsi untuk menampilkan statistik dasar dalam kartu
def display_stats_card(df):
    stats_html = f"""
    <p><b>Jumlah Data:</b> {len(df)}</p>
    <p><b>MJD Min:</b> {df['MJD'].min():.2f}</p>
    <p><b>MJD Max:</b> {df['MJD'].max():.2f}</p>
    <p><b>REFSYS Min:</b> {df['REFSYS'].min():.2f}</p>
    <p><b>REFSYS Max:</b> {df['REFSYS'].max():.2f}</p>
    <p><b>REFSYS Mean:</b> {df['REFSYS'].mean():.2f}</p>
    """
    card_with_shadow(stats_html)

# Fungsi untuk menampilkan kartu target tercapai
def display_target_achieved_card(days_needed, mjd_target):
    target_html = f"""
    <div style="text-align: center;">
        <p style="font-size: 1.2rem; margin-bottom: 5px;">Target Tercapai</p>
        <p style="font-size: 2rem; font-weight: bold; color: inherit; margin: 5px 0;">
            {days_needed} hari
        </p>
        <p style="font-size: 1.2rem; margin-top: 5px;">setelah reset</p>
        <div class="highlight" style="margin-top: 15px;">
            MJD Target: {mjd_target}
        </div>
    </div>
    """
    card_with_shadow(target_html)

# Fungsi untuk menampilkan kartu target tidak tercapai
def display_target_not_achieved_card(max_days, threshold):
    not_achieved_html = f"""
    <div style="text-align: center;">
        <p style="font-size: 1.2rem; color: #EF4444; font-weight: bold;">
            Target Tidak Tercapai
        </p>
        <p style="font-size: 1rem;">
            Dalam {max_days} hari prediksi, 
            nilai REFSYS belum mencapai {threshold}
        </p>
    </div>
    """
    card_with_shadow(not_achieved_html)

# Fungsi untuk menampilkan animasi loading
def loading_animation():
    with st.spinner('Sedang memproses...'):
        time.sleep(2)

# Konfigurasi Streamlit
st.set_page_config(
    page_title="Sistem Prediksi REFSYS - BSN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Menerapkan tema yang responsif
setup_theme()

# Header dengan logo dan judul
col1, col2, col3 = st.columns([1, 2, 1])

# with col1:
#     # Gunakan placeholder image
#     st.markdown("![Logo](/api/placeholder/150/150)")

with col2:
    st.markdown('<h1 class="header-text">SISTEM PREDIKSI REFSYS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Badan Standarisasi Nasional</p>', unsafe_allow_html=True)

with col3:
    st.write("")

# Sidebar untuk informasi dan bantuan
with st.sidebar:
    st.markdown('<h3 class="card-title">Informasi Aplikasi</h3>', unsafe_allow_html=True)
    st.info("""
    Dashboard ini membantu memprediksi kapan nilai REFSYS akan mencapai ambang batas ≥ 500 setelah direset.
    
    Menggunakan model LSTM yang telah dilatih untuk memproyeksikan nilai di masa depan.
    """)
    
    st.markdown('<h3 class="card-title">Cara Penggunaan</h3>', unsafe_allow_html=True)
    st.success("""
    1. Upload file CSV yang berisi kolom MJD dan REFSYS
    2. Tentukan jumlah iterasi prediksi
    3. Masukkan nilai awal REFSYS setelah reset
    4. Klik tombol "Jalankan Prediksi"
    """)
    
    st.markdown('<h3 class="card-title">Kontak Dukungan</h3>', unsafe_allow_html=True)
    st.markdown("""
    📧 Email: time@bsn.go.id  
    """)

# Main content area
st.markdown('<h2 class="card-title">Upload Data REFSYS</h2>', unsafe_allow_html=True)

# Buat container untuk upload file
upload_container = st.container()

with upload_container:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload file CSV yang berisi kolom 'MJD' dan 'REFSYS':", type=["csv"])
    
    with col2:
        # Perbaikan: Gunakan st.markdown dengan unsafe_allow_html
        st.markdown("""
        <div class="card">
            <p style="margin: 0; font-size: 0.9rem;">Format yang didukung: CSV</p>
            <p style="margin: 0; font-size: 0.9rem;">Wajib memiliki kolom: MJD, REFSYS</p>
        </div>
        """, unsafe_allow_html=True)

# Jika file sudah diupload
if uploaded_file is not None:
    # Tampilkan loading animation
    loading_animation()
    
    # Load data
    df = pd.read_csv(uploaded_file)

    # Validasi kolom
    if 'MJD' not in df.columns or 'REFSYS' not in df.columns:
        st.error("❌ CSV harus mengandung kolom 'MJD' dan 'REFSYS'.")
    else:
        st.success("✅ File berhasil diunggah dan dibaca!")
        
        # Tab untuk menampilkan data dan prediksi
        tab1, tab2 = st.tabs(["Data", "Prediksi"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<h3 class="card-title">Pratinjau Data</h3>', unsafe_allow_html=True)
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.markdown('<h3 class="card-title">Statistik Dasar</h3>', unsafe_allow_html=True)
                # Gunakan fungsi untuk menampilkan statistik
                display_stats_card(df)
                
                # Plot data asli dengan tema yang adaptif
                st.markdown('<h3 class="card-title">Data REFSYS</h3>', unsafe_allow_html=True)
                
                # Buat plot dengan tema yang adaptif
                fig, ax = plt.subplots(figsize=(10, 5))
                # Gunakan warna yang kompatibel dengan tema apa pun
                ax.plot(df['MJD'], df['REFSYS'], marker='.', linestyle='-', color='#3B82F6', alpha=0.7)
                ax.set_xlabel('MJD')
                ax.set_ylabel('REFSYS')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab2:
            st.markdown('<h3 class="card-title">Parameter Prediksi</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_loops = st.number_input("Jumlah iterasi prediksi:", min_value=1, max_value=10, step=1, value=1, 
                                           help="Masukkan jumlah iterasi reset yang ingin diprediksi")
            
            with col2:
                threshold = st.number_input("Nilai ambang batas:", min_value=100, max_value=1000, value=500,
                                           help="Nilai REFSYS yang menjadi target prediksi")
            
            # Input nilai awal REFSYS
            st.markdown('<h4 class="card-title">Nilai Awal REFSYS Setelah Reset</h4>', unsafe_allow_html=True)
            
            refsys_inputs = []
            cols = st.columns(min(num_loops, 4))  # Max 4 columns for better layout
            
            for i in range(num_loops):
                with cols[i % 4]:
                    val = st.number_input(f"Reset #{i+1}:", key=f"refsys_input_{i}", value=-400.0)
                    refsys_inputs.append(val)
            
            # Upload model atau gunakan model default
            model_option = st.radio("Pilih model:", ["Model Default", "Upload Model Kustom"], horizontal=True)
            
            if model_option == "Upload Model Kustom":
                model_file = st.file_uploader("Upload file model (.keras):", type=["keras"])
                if model_file:
                    # Simpan model sementara
                    model_bytes = BytesIO(model_file.read())
                    model = load_model(model_bytes)
                    st.success("✅ Model berhasil diunggah!")
                else:
                    st.warning("⚠️ Model belum diunggah, akan menggunakan model default.")
                    model = None
            else:
                try:
                    model = load_model("model_lstm_refsys.keras")
                    st.info("ℹ️ Menggunakan model default")
                except:
                    st.error("❌ Model default tidak ditemukan. Silakan upload model Anda.")
                    model = None
            
            # Tombol untuk memulai prediksi
            predict_btn = st.button("Jalankan Prediksi", use_container_width=True)
            
            if predict_btn and model is not None:
                # Tampilkan animasi loading
                loading_animation()
                
                # Siapkan scaler
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(df[['REFSYS']].values.reshape(-1, 1))
                
                seq_length = 30
                mjd_last = df['MJD'].max()
                future_mjd = mjd_last + 1
                predicted_mjd_list = []
                refsys_progress_data = []
                
                # Container untuk hasil prediksi
                st.markdown('<h3 class="card-title">Hasil Prediksi</h3>', unsafe_allow_html=True)
                results_container = st.container()
                
                with results_container:
                    for loop in range(num_loops):
                        st.markdown(f'<h4 class="card-title">Iterasi {loop+1}</h4>', unsafe_allow_html=True)
                        
                        refsys_input = refsys_inputs[loop]
                        
                        # Animasi progress bar
                        progress_bar = st.progress(0)
                        
                        # Normalisasi nilai awal
                        scaled_input = scaler.transform([[refsys_input]])[0][0]
                        last_30_days = np.full((1, seq_length, 1), scaled_input)
                        
                        max_prediction_days = 200
                        days_needed = 0
                        future_refsys = []
                        
                        progress_step = 1.0 / max_prediction_days
                        
                        for day in range(max_prediction_days):
                            prediction = model.predict(last_30_days, verbose=0)[0][0]
                            pred_original = scaler.inverse_transform([[prediction]])[0][0]
                            
                            future_refsys.append((future_mjd, pred_original))
                            days_needed += 1
                            future_mjd += 1
                            
                            # Update progress bar
                            progress_bar.progress((day + 1) * progress_step)
                            
                            last_30_days = np.append(last_30_days[:, 1:, :], [[[prediction]]], axis=1)
                            
                            if pred_original >= threshold:
                                predicted_mjd_list.append(future_mjd - 1)
                                break
                        
                        # Setelah loop selesai, progress bar = 100%
                        progress_bar.progress(1.0)
                        
                        # Visualisasi hasil iterasi dengan warna yang adaptif ke tema
                        mjd_vals = [m[0] for m in future_refsys]
                        refsys_vals = [m[1] for m in future_refsys]
                        refsys_progress_data.append((mjd_vals, refsys_vals))
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            # Gunakan warna yang kompatibel dengan tema apa pun
                            ax.plot(mjd_vals, refsys_vals, marker='o', linestyle='-', color='#3B82F6', 
                                   label='Prediksi REFSYS')
                            ax.axhline(threshold, color='#EF4444', linestyle='--', label=f'Ambang {threshold}')
                            
                            # Styling
                            ax.set_xlabel('MJD')
                            ax.set_ylabel('Nilai REFSYS')
                            ax.set_title(f'Iterasi #{loop+1}: Proyeksi REFSYS Setelah Reset', fontsize=14)
                            ax.grid(True, alpha=0.3)
                            ax.legend()
                            
                            # Fill area dengan warna adaptif
                            ax.fill_between(mjd_vals, refsys_vals, threshold, 
                                          where=(np.array(refsys_vals) < threshold), 
                                          interpolate=True, color='#93C5FD', alpha=0.5)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with col2:
                            if days_needed < max_prediction_days:
                                # Gunakan fungsi khusus untuk menampilkan kartu
                                display_target_achieved_card(days_needed, predicted_mjd_list[-1])
                            else:
                                # Gunakan fungsi khusus untuk menampilkan kartu
                                display_target_not_achieved_card(max_prediction_days, threshold)
                    
                    # Ringkasan hasil semua iterasi
                    if predicted_mjd_list:
                        st.markdown('<h3 class="card-title">Rekapitulasi Hasil Prediksi</h3>', unsafe_allow_html=True)
                        
                        # Visualisasi gabungan semua iterasi dengan warna adaptif
                        fig, ax = plt.subplots(figsize=(12, 7))
                        
                        # Pilih palet warna yang berfungsi pada tema gelap dan terang
                        colors = ['#3B82F6', '#10B981', '#F59E0B', '#6366F1', '#EC4899']
                        
                        for i, (mjd_vals, refsys_vals) in enumerate(refsys_progress_data):
                            ax.plot(mjd_vals, refsys_vals, marker='.', linestyle='-', 
                                   color=colors[i % len(colors)],
                                   label=f'Iterasi #{i+1}', alpha=0.7)
                        
                        ax.axhline(threshold, color='#EF4444', linestyle='--', label=f'Ambang {threshold}')
                        ax.set_xlabel('MJD')
                        ax.set_ylabel('Nilai REFSYS')
                        ax.set_title('Perbandingan Semua Iterasi', fontsize=14)
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Tabel ringkasan
                        result_data = {
                            "Iterasi": [f"#{i+1}" for i in range(len(predicted_mjd_list))],
                            "MJD Target": predicted_mjd_list,
                            "Nilai REFSYS Awal": refsys_inputs[:len(predicted_mjd_list)],
                            "Hari Setelah Reset": [len(refsys_progress_data[i][0]) for i in range(len(predicted_mjd_list))]
                        }
                        
                        result_df = pd.DataFrame(result_data)
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Opsi download hasil
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Hasil Prediksi (CSV)",
                            data=csv,
                            file_name="prediksi_refsys_hasil.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; opacity: 0.7;">
    <p>© 2025 Badan Standarisasi Nasional. Hak Cipta Dilindungi.</p>
</div>
""", unsafe_allow_html=True)
