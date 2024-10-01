import speech_recognition as sr
import streamlit as st


def print_mic_device_index(): # Printa os microfones disponíveis
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print("{1}\"  `Microphone(device_index={0})`".format(index, name))

def speach_to_text(device_index, language="pt-BR"): 
    # Função para converter fala em texto usando portugues
    r = sr.Recognizer()
    with sr.Microphone(device_index=device_index) as source:
        st.info("Fale alguma coisa!")
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio, language=language)
            st.success(f"Você disse: {text}")
            return text
        except:
            st.error("Não entendi o que você disse.")
            return None
    
                

