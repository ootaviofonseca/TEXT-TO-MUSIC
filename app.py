import os
import streamlit as st
import torch
import torchaudio
import base64
import time 
from audiocraft.models.musicgen import MusicGen
import ollama
import voice as vc
import speech_recognition as sr

# Configuração da página no Streamlit, com tema personalizado
st.set_page_config(page_title="Text-to-Music", page_icon=":headphones:")

st.markdown(
    """
    <style>
        .stApp {
            background-color: white;
        }
        
        * {
            color: black !important;
        }

        .stTextArea textarea {
            background-color: #d3d3d3 !important; 
            color: black !important;
        }
        
        div.stButton > button {
            background-color: #708090 !important; 
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        
        div.stDownloadButton > button {
            background-color: #00BFFF !important; 
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        
        .expander-content {
            display: flex;
            justify-content: center;
            width: 70%;
        }
        
        
        .stExpander {\
            display: flex;
            margin-left: auto;
            margin-right: auto;

        }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource  # Decorador para carregar o modelo apenas uma vez
def load_model():
    # Carrega o modelo pré-treinado
    model = MusicGen.get_pretrained('facebook/musicgen-medium')
    return model

# Função para gerar os tensores da música com base na descrição e duração informada
def generate_music_tensors(description, duration: int):
    model = load_model()
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration,
    )

    output = model.generate(
        descriptions=[description],  
        progress=True,             
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

# Função para gerar o botão de download do áudio
def download_button(bin_file, button_label='Download'):
    with open(bin_file, 'rb') as f:
        data = f.read()  # Lê o arquivo binário
    bin_str = base64.b64encode(data).decode()  # Converte o arquivo binário em base64
    # Cria o botão de download
    st.download_button(label=button_label, data=data, file_name=os.path.basename(bin_file))

# Função para criar um prompt padronizado para a IA
def create_prompt(user_input):
     
    prompt = """
        Build a description based on the text provided by the user (tranlate the text to english), 
        which will be used as a prompt for another artificial intelligence that will create a 
        budist instrumental music based on this description. 
       
        
       
       the output must be in the following format  : 
            Description: (Here should be a description explaining how the music should be, feelings, 
           , some of the budists music types, and similar aspects,   not the description given by the user.)    
            Genre: (Here should be the genre of budist music )
                 
...         
    """
    
    response = ollama.chat(model='gemma:2b', messages=[
        {
          'role': 'system',
          'content': prompt
        },
    {
          'role': 'user',
          'content': user_input,
    },
    ])
    return response['message']['content']

# Função principal
def main():
    st.title("Gerando Músicas!")

    #expander para informações sobre o projeto
    with st.expander("Sobre o projeto"):
        st.markdown('<div class="expander-content">', unsafe_allow_html=True) 
        st.write("Esse é um projeto que gera música usando o modelo Meta Audiocraft MusicGen, "
                 "baseado em uma descrição feita pelo usuário.")
        st.markdown('</div>', unsafe_allow_html=True)


    # Seleção do microfone para captura do áudio
    device_index = 1
    
    time_slider = st.slider("Duração da música (em segundos) :", 2, 20, 5)

    # Botão para capturar o áudio e convertê-lo em texto
    gravar_audio = st.button("Capturar áudio")
    

    if gravar_audio:
        user_input = vc.speach_to_text(device_index)
       

        if user_input:
               
            st.subheader("Música gerada")

            prompt = create_prompt(user_input) # Cria o prompt para a IA com base na descrição do usuário
            st.info("Descicao gerada:" +  prompt)

            start_time = time.time()
            with st.spinner('Gerando música...'):  # Adiciona um indicador de carregamento
                music_tensors = generate_music_tensors(prompt, time_slider)
            end_time = time.time()
            totaltime = (end_time - start_time)/60
            st.write(f"Tempo de execução: {totaltime:.2f} minutos")
            audio_filepath = "results_audios/audio_0.wav"  # Caminho do arquivo de áudio
            save_audio(music_tensors)  # Salva o áudio no diretório results_audios

            # Exibe o áudio no Streamlit
            with open(audio_filepath, "rb") as audio_file:
                audio_bytes = audio_file.read()

            st.audio(audio_bytes)
            
            # Botão de download do áudio
            download_button(audio_filepath, 'Download Audio')

if __name__ == "__main__":    main()
