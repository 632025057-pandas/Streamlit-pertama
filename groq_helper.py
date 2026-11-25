import json
import os
import streamlit as st
from groq import Groq
from datetime import datetime, timedelta, timezone

def _get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


def tanya_groq(
    prompt,
    system_prompt=(
        "Kamu adalah tutor Machine Learning yang ramah dan edukatif. "
        "Jelaskan konsep dengan bahasa Indonesia yang singkat, terstruktur, "
        "mudah dipahami mahasiswa, dan sertakan contoh sederhana bila relevan."
    ),
    max_tokens=700,
):
    api_key = _get_groq_key()
    if not api_key:
        return None, "GROQ_API_KEY belum diatur di Streamlit Secrets."
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"Gagal memanggil Groq API: {e}"


def analisis_sentimen_komentar(komentar_list):
    """
    Menerima list dict: [{"id": "...", "komentar": "..."}]
    Mengembalikan tuple: (hasil_dict, error_str)
    """
    api_key = _get_groq_key()
    if not api_key:
        return None, "GROQ_API_KEY belum diatur di Streamlit Secrets."
    if not komentar_list:
        return None, "Tidak ada komentar untuk dianalisis."

    system_prompt = (
        "Kamu adalah analis sentimen komentar berbahasa Indonesia. "
        "Analisis setiap komentar dan kembalikan HANYA objek JSON valid dengan format:\n"
        "{\n"
        '  "ringkasan": "ringkasan singkat 1-2 kalimat dari seluruh komentar",\n'
        '  "hasil": [\n'
        '    {"id": "id_komentar", "sentimen": "Positif", "alasan": "alasan sangat singkat 3-7 kata"}\n'
        "  ]\n"
        "}\n"
        'Nilai "sentimen" WAJIB salah satu dari: "Positif", "Negatif", atau "Netral". '
        "Tulis alasan dengan huruf kecil tanpa tanda titik di akhir, misalnya: "
        '"menyatakan kepuasan dan memberi saran" atau "pertanyaan tanpa ekspresi emosi".'
    )

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(komentar_list, ensure_ascii=False),
                },
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content
        return json.loads(raw), None
    except Exception as e:
        return None, f"Gagal menganalisis komentar dengan Groq: {e}"


# ----------------------------------------------------------------------------
# Pertanyaan Edukatif Cepat untuk Halaman Asisten AI
# ----------------------------------------------------------------------------
QUICK_PROMPTS = [
    (
        "🧭 Supervised vs Unsupervised",
        "Apa perbedaan Supervised Learning dan Unsupervised Learning? Berikan contoh dari Regresi, Klasifikasi, dan Clustering.",
    ),
    (
        "📈 Cara Baca R², MAE & RMSE",
        "Bagaimana cara membaca nilai R² Score, MAE, dan RMSE pada Regresi Linear untuk menentukan apakah model sudah baik?",
    ),
    (
        "🎯 Precision vs Recall",
        "Apa bedanya Precision dan Recall, dan mengapa Recall sangat penting pada kasus medis seperti deteksi Breast Cancer?",
    ),
    (
        "⚖️ Mengapa Perlu StandardScaler?",
        "Mengapa fitur data perlu distandarisasi dengan StandardScaler sebelum melatih KNN, Logistic Regression, dan K-Means?",
    ),
    (
        "🧪 Train/Test Split & Overfitting",
        "Apa yang dimaksud dengan Overfitting dalam Machine Learning dan bagaimana pembagian Train/Test Split membantu mendeteksinya?",
    ),
    (
        "🔵 Memilih Cluster k Optimal",
        "Bagaimana cara memilih jumlah cluster (k) paling ideal pada K-Means menggunakan grafik Elbow Method dan Silhouette Score?",
    ),
]


def page_ai_assistant():
    st.markdown(
        '<h1 class="main-title-container">🤖 <span class="gradient-text">Asisten AI (Groq)</span></h1>',
        unsafe_allow_html=True,
    )
    st.info(
        "💡 Pilih topik cepat di bawah ini untuk belajar konsep Machine Learning secara instan, "
        "atau ketik pertanyaan Anda sendiri mengenai dataset, rumus, dan evaluasi model."
    )

    st.markdown("##### ⚡ Pertanyaan Edukatif Cepat")
    selected_quick = None
    cols = st.columns(2)
    for idx, (label, q_text) in enumerate(QUICK_PROMPTS):
        if cols[idx % 2].button(label, key=f"quick_q_{idx}", use_container_width=True):
            selected_quick = q_text

    st.write("")
    st.markdown("##### ✍️ Atau Tulis Pertanyaan Sendiri")
    prompt = st.text_area(
        "Pertanyaan Anda:",
        placeholder="Contoh: Apa arti nilai R² Score 0.50 pada dataset Diabetes?",
        height=100,
        label_visibility="collapsed",
    )
    submit_manual = st.button("🚀 Tanya Groq", type="primary")

    active_prompt = selected_quick or (prompt.strip() if submit_manual else None)

    if submit_manual and not prompt.strip():
        st.warning("Silakan tulis pertanyaan terlebih dahulu.")
    elif active_prompt:
        with st.spinner("Groq sedang menyusun penjelasan..."):
            jawaban, err = tanya_groq(active_prompt)
        if err:
            st.error(err)
        else:
            st.session_state["last_ai_q"] = active_prompt
            st.session_state["last_ai_a"] = jawaban

    if st.session_state.get("last_ai_a"):
        st.divider()
        st.markdown(f"**❓ Pertanyaan:** {st.session_state['last_ai_q']}")
        with st.container(border=True):
            st.markdown("### 💬 Penjelasan Groq")
            st.markdown(st.session_state["last_ai_a"])


# ----------------------------------------------------------------------------
# Halaman Mini Kuis Machine Learning
# ----------------------------------------------------------------------------
DEFAULT_KUIS = [
    {
        "pertanyaan": "Algoritma K-Means pada modul Clustering termasuk ke dalam paradigma pembelajaran apa?",
        "pilihan": [
            "Supervised Learning karena membutuhkan kolom target",
            "Unsupervised Learning karena mengelompokkan data tanpa label target",
            "Reinforcement Learning karena menggunakan sistem reward",
            "Regresi Linear karena memprediksi angka kontinu",
        ],
        "jawaban": 1,
        "pembahasan": "K-Means adalah algoritma Unsupervised Learning karena menemukan kelompok alami hanya dari kemiripan fitur tanpa label kelas awal.",
    },
    {
        "pertanyaan": "Jika model Regresi Linear menghasilkan R² Score sebesar 0.80 pada data uji, apa artinya?",
        "pilihan": [
            "Model salah memprediksi sebesar 80% dari seluruh data",
            "Sekitar 80% variasi pada nilai target dapat dijelaskan oleh fitur input model",
            "Rata-rata kesalahan prediksi bernilai 0.80 satuan",
            "Model mengalami overfitting sempurna",
        ],
        "jawaban": 1,
        "pembahasan": "R² Score (koefisien determinasi) mengukur proporsi variasi target yang berhasil dijelaskan oleh model. Nilai 0.80 berarti 80% variasi dijelaskan oleh fitur.",
    },
    {
        "pertanyaan": "Pada klasifikasi medis seperti dataset Breast Cancer, mengapa metrik Recall sangat krusial diperhatikan?",
        "pilihan": [
            "Untuk meminimalkan kasus tumor ganas yang terlewat (False Negative)",
            "Untuk mempercepat waktu pelatihan model Random Forest",
            "Untuk mengubah output kelas menjadi angka kontinu",
            "Untuk mengurangi jumlah dimensi fitur menjadi 2D",
        ],
        "jawaban": 0,
        "pembahasan": "Recall mengukur seberapa banyak kasus positif asli yang berhasil terdeteksi. Pada kasus medis, melewatkan pasien sakit (False Negative) jauh lebih berbahaya.",
    },
    {
        "pertanyaan": "Mengapa data fitur perlu distandarisasi menggunakan StandardScaler sebelum menjalankan K-Means atau KNN?",
        "pilihan": [
            "Agar seluruh data menjadi bilangan bulat positif",
            "Agar fitur dengan skala angka besar tidak mendominasi perhitungan jarak",
            "Agar jumlah baris dataset bertambah secara otomatis",
            "Agar label kelas berubah menjadi teks",
        ],
        "jawaban": 1,
        "pembahasan": "KNN dan K-Means menghitung jarak antar titik (misal Euclidean). Tanpa standarisasi, fitur berskala ribuan akan mendominasi fitur berskala satuan.",
    },
    {
        "pertanyaan": "Apa fungsi utama PCA (Principal Component Analysis) pada grafik visualisasi Clustering di dashboard ini?",
        "pilihan": [
            "Menghapus data duplikat dari MongoDB",
            "Meningkatkan jumlah cluster k menjadi 10",
            "Mereduksi banyak fitur menjadi 2 komponen utama (PC1 & PC2) agar bisa digambar di grafik 2D",
            "Menggantikan rumus Confusion Matrix",
        ],
        "jawaban": 2,
        "pembahasan": "Dataset seperti Wine atau Breast Cancer memiliki banyak fitur (multidimensi). PCA memproyeksikannya ke 2 sumbu utama (PC1 & PC2) untuk visualisasi 2D.",
    },
]


def buat_soal_kuis_groq(topik):
    api_key = _get_groq_key()
    if not api_key:
        return None, "GROQ_API_KEY belum diatur di Streamlit Secrets."

    system_prompt = (
        "Kamu adalah dosen Machine Learning. Buat TEPAT 5 soal pilihan ganda berbahasa Indonesia "
        "yang edukatif dan akurat sesuai topik yang diminta. Kembalikan HANYA JSON valid dengan format:\n"
        "{\n"
        '  "soal": [\n'
        "    {\n"
        '      "pertanyaan": "teks pertanyaan",\n'
        '      "pilihan": ["opsi A", "opsi B", "opsi C", "opsi D"],\n'
        '      "jawaban": 0,\n'
        '      "pembahasan": "penjelasan singkat mengapa jawaban tersebut benar"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        'Nilai "jawaban" adalah indeks integer (0, 1, 2, atau 3) dari list "pilihan" yang benar. '
        "Jangan tulis huruf A/B/C/D di awal teks pilihan."
    )

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Topik kuis: {topik}"},
            ],
            temperature=0.6,
            max_tokens=1800,
        )
        data = json.loads(response.choices[0].message.content)
        soal = data.get("soal", [])
        valid = []
        for s in soal:
            pilihan = s.get("pilihan", [])
            jwb = s.get("jawaban")
            if (
                isinstance(pilihan, list)
                and len(pilihan) == 4
                and isinstance(jwb, int)
                and 0 <= jwb < 4
                and s.get("pertanyaan")
            ):
                valid.append({
                    "pertanyaan": str(s["pertanyaan"]),
                    "pilihan": [str(p) for p in pilihan],
                    "jawaban": jwb,
                    "pembahasan": str(s.get("pembahasan", "")),
                })
        if not valid:
            return None, "Format soal dari Groq tidak valid. Silakan coba lagi."
        return valid, None
    except Exception as e:
        return None, f"Gagal membuat soal dengan Groq: {e}"


def page_kuis_ml(db=None):
    # ... (bagian judul, selectbox topik, dan tombol Groq tetap sama) ...

    with st.form(f"form_kuis_{ver}"):
        nama_peserta = st.text_input(
            "Nama Peserta (untuk mencatat skor ke MongoDB)",
            placeholder="Contoh: Hana",
            max_chars=50,
            key=f"kuis_nama_{ver}",
        )
        st.write("")

        jawaban_user = []
        for i, item in enumerate(soal_list):
            st.markdown(f"**{i + 1}. {item['pertanyaan']}**")
            pilih = st.radio(
                f"Pilihan soal {i + 1}",
                options=range(len(item["pilihan"])),
                format_func=lambda idx, opts=item["pilihan"]: opts[idx],
                index=None,
                label_visibility="collapsed",
                key=f"kuis_v{ver}_q{i}",
            )
            jawaban_user.append(pilih)
            st.write("")

        submitted = st.form_submit_button("✅ Periksa Jawaban", type="primary")

    if submitted:
        if any(j is None for j in jawaban_user):
            st.warning("Mohon jawab semua pertanyaan terlebih dahulu sebelum memeriksa skor.")
            return

        benar = sum(1 for u, item in zip(jawaban_user, soal_list) if u == item["jawaban"])
        total = len(soal_list)
        skor = (benar / total) * 100

        # Simpan otomatis ke MongoDB
        if db is not None:
            try:
                db.riwayat_kuis.insert_one({
                    "nama": nama_peserta.strip() or "Anonim",
                    "paket": st.session_state["kuis_sumber"],
                    "benar": int(benar),
                    "total": int(total),
                    "skor": float(skor),
                    "created_at": datetime.now(timezone.utc),
                })
                st.toast("✅ Skor kuis tersimpan ke MongoDB!", icon="🏆")
            except Exception as e:
                st.warning(f"Skor gagal disimpan ke MongoDB: {e}")

        # ... (bagian tampilan metrik skor & pembahasan tetap sama) ...

    # Tampilkan Riwayat Skor / Leaderboard dari MongoDB di bagian bawah
    if db is not None:
        st.divider()
        st.subheader("🏆 Riwayat Skor Kuis Terbaru (MongoDB)")
        try:
            riwayat = list(db.riwayat_kuis.find().sort("created_at", -1).limit(15))
            if not riwayat:
                st.info("Belum ada riwayat pengerjaan kuis tersimpan.")
            else:
                tabel = [
                    {
                        "Waktu": (r["created_at"] + timedelta(hours=7)).strftime("%d %b %Y %H:%M WIB"),
                        "Nama": r.get("nama", "Anonim"),
                        "Paket Soal": r.get("paket", "-"),
                        "Benar": f"{r.get('benar', 0)}/{r.get('total', 5)}",
                        "Skor": f"{r.get('skor', 0):.0f}",
                    }
                    for r in riwayat
                ]
                st.dataframe(tabel, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Gagal memuat riwayat kuis: {e}")
    st.markdown(
        '<h1 class="main-title-container">📝 <span class="gradient-text">Mini Kuis Machine Learning</span></h1>',
        unsafe_allow_html=True,
    )
    st.info(
        "💡 Uji pemahaman Anda tentang Regresi Linear, Klasifikasi, Clustering, dan Evaluasi Model. "
        "Anda dapat mengerjakan soal standar atau membuat paket soal baru secara dinamis menggunakan Groq AI."
    )

    if "kuis_soal" not in st.session_state:
        st.session_state["kuis_soal"] = DEFAULT_KUIS
    if "kuis_ver" not in st.session_state:
        st.session_state["kuis_ver"] = 0
    if "kuis_sumber" not in st.session_state:
        st.session_state["kuis_sumber"] = "Paket Soal Standar (Kurikulum TAS AI)"

    c1, c2, c3 = st.columns([2, 1.3, 1])
    topik = c1.selectbox(
        "Pilih Topik untuk Soal Baru (Groq AI)",
        [
            "Semua Materi (Regresi, Klasifikasi, Clustering)",
            "Regresi Linear & Metrik R², MAE, RMSE",
            "Klasifikasi, Confusion Matrix & Precision/Recall",
            "Clustering K-Means, PCA & Silhouette Score",
            "Train/Test Split, Overfitting & Retraining",
        ],
        key="kuis_topik",
    )
    c2.write("")
    c2.write("")
    if c2.button("✨ Buat Soal Baru (Groq)", use_container_width=True, type="primary"):
        with st.spinner("Groq sedang menyusun 5 soal baru..."):
            soal_baru, err = buat_soal_kuis_groq(topik)
        if err:
            st.error(err)
        else:
            st.session_state["kuis_soal"] = soal_baru
            st.session_state["kuis_ver"] += 1
            st.session_state["kuis_sumber"] = f"Soal Dinamis Groq AI — {topik}"
            st.toast("✅ 5 soal baru siap dikerjakan!", icon="📝")
            st.rerun()

    c3.write("")
    c3.write("")
    if c3.button("🔄 Soal Standar", use_container_width=True):
        st.session_state["kuis_soal"] = DEFAULT_KUIS
        st.session_state["kuis_ver"] += 1
        st.session_state["kuis_sumber"] = "Paket Soal Standar (Kurikulum TAS AI)"
        st.rerun()

    st.caption(f"📚 **Paket Aktif:** {st.session_state['kuis_sumber']}")

    soal_list = st.session_state["kuis_soal"]
    ver = st.session_state["kuis_ver"]

    with st.form(f"form_kuis_{ver}"):
        jawaban_user = []
        for i, item in enumerate(soal_list):
            st.markdown(f"**{i + 1}. {item['pertanyaan']}**")
            pilih = st.radio(
                f"Pilihan soal {i + 1}",
                options=range(len(item["pilihan"])),
                format_func=lambda idx, opts=item["pilihan"]: opts[idx],
                index=None,
                label_visibility="collapsed",
                key=f"kuis_v{ver}_q{i}",
            )
            jawaban_user.append(pilih)
            st.write("")

        submitted = st.form_submit_button("✅ Periksa Jawaban", type="primary")

    if submitted:
        if any(j is None for j in jawaban_user):
            st.warning("Mohon jawab semua pertanyaan terlebih dahulu sebelum memeriksa skor.")
            return

        benar = sum(1 for u, item in zip(jawaban_user, soal_list) if u == item["jawaban"])
        total = len(soal_list)
        skor = (benar / total) * 100

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Skor Akhir", f"{skor:.0f} / 100")
        m2.metric("Jawaban Benar", f"{benar} dari {total} soal")
        st.progress(benar / total)

        if skor >= 80:
            st.balloons()
            st.success("🎉 Hebat! Pemahaman konsep Machine Learning Anda sudah sangat baik.")
        elif skor >= 60:
            st.info("👍 Cukup baik! Pelajari kembali pembahasan pada soal yang masih keliru di bawah ini.")
        else:
            st.warning("📖 Yuk pelajari ulang kotak teori di tiap modul atau tanyakan ke Asisten AI (Groq), lalu coba lagi!")

        st.subheader("📋 Pembahasan Jawaban")
        for i, (u, item) in enumerate(zip(jawaban_user, soal_list)):
            kunci_idx = item["jawaban"]
            kunci_teks = item["pilihan"][kunci_idx]
            if u == kunci_idx:
                st.success(
                    f"**Soal {i + 1}: Benar** — Jawaban Anda: *{kunci_teks}*\n\n"
                    f"💡 **Pembahasan:** {item['pembahasan']}"
                )
            else:
                user_teks = item["pilihan"][u]
                st.error(
                    f"**Soal {i + 1}: Kurang Tepat** — Jawaban Anda: *{user_teks}* | "
                    f"Jawaban Benar: **{kunci_teks}**\n\n"
                    f"💡 **Pembahasan:** {item['pembahasan']}"
                )
