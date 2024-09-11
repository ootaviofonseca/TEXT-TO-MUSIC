import os
import streamlit as st
import torch
import torchaudio
import numpy as np
import base64
from audiocraft.models.musicgen import MusicGen

# Configuração da página
st.set_page_config(page_title="Music Generator", page_icon=":headphones:")

@st.cache_resource  # Decorador para carregar o modelo apenas uma vez
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def generate_music_tensors(description, duration:int):
    model = load_model()
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration,
    )

    output = model.generate(
        descriptions=[description],
        progress=False,  # Remove o progresso para acelerar a geração
        return_tokens=True,
    )

    return output[0]

def save_audio(samples: torch.Tensor):
    samples_rate = 32000
    save_path = "results_audios/"
    os.makedirs(save_path, exist_ok=True)  # Cria o diretório se não existir

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, samples_rate)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def main():
    st.title("Gerando Música com IA")

    with st.expander("Sobre o projeto"):
        st.write("Esse é um projeto simples que gera música usando o modelo Meta Audiocraft Music Gen. "
                 "Baseado em uma descrição feita pelo usuário, o modelo gera uma música.")

    text_area = st.text_area("Digite uma descrição para a música")
    time_slider = st.slider("Duração da música (em segundos)", 2, 20, 5)

    if text_area and time_slider:
        st.json({
            "Descrição": text_area,
            "Duração": time_slider
        })
        st.subheader("Música gerada")

        with st.spinner('Gerando música...'):  # Adiciona um indicador de carregamento
            music_tensors = generate_music_tensors(text_area, time_slider)

        save_music = save_audio(music_tensors)

        audio_filepath = "results_audios/audio_0.wav"

        with open(audio_filepath, "rb") as audio_file:
            audio_bytes = audio_file.read()

        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
