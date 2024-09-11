# Music Generator

## Descrição

Este projeto utiliza o modelo Meta Audiocraft MusicGen para gerar música com base em uma descrição textual fornecida pelo usuário. Através de uma interface simples criada com Streamlit, você pode inserir uma descrição e escolher a duração desejada para gerar uma peça musical personalizada. O áudio gerado pode ser baixado diretamente da interface.

## Requisitos

- Python 3.8 ou superior
- Streamlit
- PyTorch
- Torchaudio
- NumPy
- Audiocraft

## Instalação

1. Clone este repositório

2. Instale as dependências:
    ```bash
    pip install streamlit torch torchaudio numpy

3. Instale a biblioteca Audiocraft:
    ```bash
    pip install audiocraft

## Uso
1. Execute o aplicativo Streamlit:
    ```bash
    streamlit run app.py

2. Acesse a interface web que será aberta no seu navegador. Insira uma descrição para a música e ajuste a duração desejada. Clique em "Gerar Música" para criar o áudio.

3. Após a geração, você poderá ouvir a música diretamente na interface e também fazer o download do arquivo de áudio gerado.

## Código
### Funções Principais
- `load_model()`: Carrega o modelo pré-treinado MusicGen.
- `generate_music_tensors(description, duration)`: Gera a música com base na descrição e duração fornecidas.
- `save_audio(samples)`: Salva os áudios gerados em arquivos WAV.
- `get_binary_file_downloader_html(bin_file, file_label='File')`: Gera um link para download do arquivo de áudio.

### Estrutura do Projeto
- `app.py`: Arquivo principal contendo o código do aplicativo Streamlit.
- `results_audios/`: Pasta onde os arquivos de áudio gerados são salvos.

