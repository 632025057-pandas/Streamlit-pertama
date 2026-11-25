import json
import os
import streamlit as st
from groq import Groq


def _get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


def tanya_groq(
    prompt,
    system_prompt="Kamu adalah asisten AI yang membantu menjelaskan konsep Machine Learning dengan bahasa Indonesia yang singkat, jelas, dan mudah dipahami.",
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
            max_tokens=512,
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"Gagal memanggil Groq API: {e}"


def page_ai_assistant():
    st.markdown(
        '<h1 class="main-title-container">🤖 <span class="gradient-text">Asisten AI (Groq)</span></h1>',
        unsafe_allow_html=True,
    )
    st.info(
        "💡 Tanyakan apa saja seputar Regresi Linear, Klasifikasi, Clustering, atau hasil evaluasi model di dashboard ini."
    )

    prompt = st.text_area(
        "Pertanyaan Anda:",
        placeholder="Contoh: Apa arti nilai R² Score 0.50 pada dataset Diabetes?",
        height=100,
    )

    if st.button("🚀 Tanya Groq", type="primary"):
        if not prompt.strip():
            st.warning("Silakan tulis pertanyaan terlebih dahulu.")
        else:
            with st.spinner("Groq sedang menyusun jawaban..."):
                jawaban, err = tanya_groq(prompt.strip())
            if err:
                st.error(err)
            else:
                st.markdown("### 💬 Jawaban Groq")
                st.write(jawaban)
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