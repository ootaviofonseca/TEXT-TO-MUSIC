import os

import streamlit as st  # Biblioteca para criar a interface web para o usuário 
import torch  # Biblioteca para rodar o modelo de deep learning para a geração de música
import torchaudio  # Biblioteca para rodar o modelo de deep learning para a geração de música
import numpy as np 
import base64

from audiocraft.models.musicgen import MusicGen  # Para converter o arquivo de áudio em base64 para reprodução no Streamlit

# Configuração da página deve vir antes de outros códigos Streamlit
st.set_page_config(
    page_title="Music Generator",
    page_icon=":headphones:"
)

@st.cache_resource  # Decorador para carregar o modelo apenas uma vez
def load_model():   
    model = MusicGen.get_pretrained('facebook/musicgen-small') 
    # Carrega o modelo pré-treinado 
    return model
def generate_music_tensors(description, duration:int):
    print ("Descricao: ", description)
    print ("Duração: ", duration)
    model = load_model()  # Carrega o modelo 
    model.set_generation_params( # Parâmetros para a geração da música 
        use_sampling= True, # Usa sampling para gerar a música
        top_k= 250, # Top k é o número de tokens a serem considerados para a geração da música
        duration= duration, # Duração da música
    )

    output = model.generate(
        descriptions= [description], # Descrição da música 
        progress= True, # Mostra o progresso da geração da música
        return_tokens= True, # Retorna os tokens gerados 
    )

    return output[0]  # Retorna a música gerada 


def save_audio(samples: torch.Tensor):
    samples_rate = 32000  # Taxa de sampling que siginifica quantas amostras de áudio são feitas por segundo
    save_path = "results_audios/"  # Caminho para salvar o arquivo de áudio

    assert samples.dim( ) == 2 or samples.dim( ) == 3 # Verifica se a dimensão do tensor é 2 ou 3

    samples = samples.detach( ).cpu( ) # Remove o tensor da GPU e o coloca na CPU

    if samples.dim( ) == 2:
        samples = samples[None, ...] # Adiciona uma dimensão extra ao tensor 
    
    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")  # Caminho para salvar o arquivo de áudio e o formato
        torchaudio.save(audio_path, audio, samples_rate)

def get_binary_file_downloader_html(bin_file, file_label='File'): # Função para converter o arquivo de áudio em base64
    
    with open(bin_file, 'rb') as f: # Abre o arquivo de áudio em binário
        data = f.read() # Lê o arquivo de áudio
    bin_str = base64.b64encode(data).decode() # Converte o arquivo de áudio em base64
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    # Cria um link para download do arquivo de áudio em base64 
    return href 
def main():
    st.title("Gerando Música com IA")  # Título da página

    with st.expander("Sobre o projeto"):  # Expander é um botão que expande um texto 
        st.write("Esse é um projeto simples que gera música usando o modelo Meta Audiocraft Music Gen. "
                 "Baseado em uma descrição feita pelo usuário, o modelo gera uma música. ")

    text_area = st.text_area("Digite uma descrição para a música")
    time_slider = st.slider("Duração da música (em segundos)", 2, 20, 5)  # Slider para escolher a duração da música

    if text_area and time_slider: 
        st.json({
            "Descrição": text_area,
            "Duração": time_slider
        })
        st.subheader("Música gerada")
        
        music_tensors = generate_music_tensors(text_area, time_slider) # Gera a música com base na descrição e duração
        print("Music tensors: ", music_tensors)  # Imprime os tensores da música gerada 

        save_music = save_audio(music_tensors)  # Salva a música gerada

        audio_filepath = "results_audios/audio_0.wav" 
        # Caminho para o arquivo de áudio gerado sempre vai sobrescrever o arquivo anterior , mudar a logica para gera mais de um arquivo

        audio_file = open(audio_filepath, "rb")
        audio_bytes = audio_file.read()

        st.audio(audio_bytes)  # Reproduz a música gerada
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)
        # Cria um link para download do arquivo de áudio gerado em base64 e o unsafe_allow_html=True permite a execução do HTML



if __name__ == "__main__":
    main()
