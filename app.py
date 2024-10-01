import os
import streamlit as st
import torch
import torchaudio
import base64
import time 
from audiocraft.models.musicgen import MusicGen
import ollama


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
        Build a text based on the text provided by the user(tranlate the text to english), 
        which will be used as a prompt for another artificial 
        intelligence that will create instrumental music based on this description. 
        All music must be based in the Buddhism musics culture.
        
        Select one of the following music genres based on the user description:

            1. Chanting and Mantras:   
                Common Instruments:
                    Bells: Symbolize wisdom and the clear, awakening mind.
                Mala beads: Used to count repetitions of mantras.
                Feelings Evoked: Peace, mindfulness, spiritual focus, and inner calm. Chanting often leads to a sense of transcendence and connection with the sacred.
            2. Ceremonial Music
                Description: Found primarily in Tibetan and Chinese Buddhism, this music accompanies rituals, such as pujas (offerings), funerals, and celebrations.
                Common Instruments:
                    Damaru: A small hand-held drum, symbolizing the sound of the dharma (teachings).
                    Dungchen: Long Tibetan trumpets that produce deep, resonant sounds, often used in monastery rituals.
                    Cymbals (Tingsha): Small metallic cymbals used to signify the start or end of a practice, invoking clarity and sharpness.
                    Gong: Used to mark transitions during ceremonies or meditative sessions.
                Feelings Evoked: Reverence, solemnity, and spiritual awakening. The deep vibrations of instruments like the dungchen can bring a sense of grounding and connection to higher states of consciousness.
            3. Zen Music (Shomyo)
                Description: In Japanese Zen Buddhism, Shomyo is a form of Buddhist chant with roots in Indian Vedic traditions. It is minimalist, focusing on deep breaths and slow, deliberate sounds.
                Common Instruments:
                    Wooden clappers (Mokugyo): Used to keep rhythm during chants.
                    Fuke shakuhachi (bamboo flute): Played by monks as a form of meditation and prayer.
                Feelings Evoked: Simplicity, introspection, and detachment from worldly distractions. The music encourages mindfulness and the exploration of inner stillness.
            4. Meditation Music
                Description: Instrumental music designed to aid meditation, often slow and calming, featuring sustained notes or gentle, flowing melodies.
                Common Instruments:
                    Sitar: A long-necked Indian instrument with sympathetic strings that create a meditative, hypnotic resonance.
                    Tanpura: A drone instrument, providing a constant hum in the background, which creates a foundation for deeper concentration.
                    Flute: Typically used to add soft, airy tones that evoke tranquility.
                Feelings Evoked: Deep relaxation, focus, and serenity. Meditation music is meant to slow the mind and create a space for contemplation, self-awareness, and inner peace.
            5. Tibetan Monastery Music
                Description: Music from Tibetan monasteries often features low, guttural throat singing and is performed during religious rituals and ceremonies.
                    Common Instruments:
                    Gyaling: A Tibetan reed instrument similar to an oboe, producing high-pitched, penetrating sounds.
                    Throat singing (Chanting): Monks use a unique vocal technique that allows them to produce multiple tones simultaneously, often combined with overtone singing.
                    Dungchen: Large Tibetan horns that produce deep, reverberating sounds.
                Feelings Evoked: Awe, transcendence, and spiritual depth. The combination of throat singing and low-frequency instruments can create an otherworldly atmosphere, encouraging detachment from the physical world and immersion in spiritual practice.

       the prompt must be in the following format  : 
            Description: (Here should be a description explaining how the music should be, feelings, 
            and similar aspects, and not the description given by the user.)
            

       
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

    

    text_area = st.text_area("Digite uma descrição para a música :")
    time_slider = st.slider("Duração da música (em segundos) :", 2, 20, 5)

    

    # Botão para gerar a música
    gerar_musica = st.button("Gerar Música")

    if gerar_musica and text_area and time_slider:
        st.subheader("Música gerada")

        prompt = create_prompt(text_area) # Cria o prompt para a IA com base na descrição do usuário
        print (prompt)


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
