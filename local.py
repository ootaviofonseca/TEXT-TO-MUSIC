import os
import torch
import torchaudio
import numpy as np
import base64
from audiocraft.models.musicgen import MusicGen

# Função para carregar o modelo pré-treinado
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

# Função para gerar a música
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

# Função para salvar o áudio gerado
def save_audio(samples: torch.Tensor, save_path: str):
    samples_rate = 32000
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()

    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, samples_rate)
        print(f"Áudio salvo em: {audio_path}")

# Função principal
def main():
    # Solicita a descrição e duração da música
    description = input("Digite uma descrição para a música: ")
    duration = int(input("Digite a duração da música (em segundos): "))
    
    # Gera a música
    print("Gerando música...")
    music_tensors = generate_music_tensors(description, duration)
    
    # Define o caminho para salvar o áudio
    save_path = "results_audio"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Salva o áudio gerado
    save_audio(music_tensors, save_path)
    print("Música gerada e salva com sucesso.")

if __name__ == "__main__":
    main()
