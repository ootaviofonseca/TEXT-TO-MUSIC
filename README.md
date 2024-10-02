# Music Generator

## Descrição

Este projeto utiliza o modelo Meta Audiocraft MusicGen para gerar música com base em uma descrição falada fornecida pelo usuário. Através de uma interface simples criada com Streamlit, você pode falar uma descrição e escolher a duração desejada para gerar uma peça musical personalizada. O áudio gerado pode ser baixado ou ouvido diretamente da interface.

## Tecnologias e Bibliotecas
![Python](https://img.shields.io/badge/Python%2B-%2312343D.svg?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B00.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Torchaudio](https://img.shields.io/badge/Torchaudio-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Audiocraft](https://img.shields.io/badge/Audiocraft-lightgrey?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-%2326A2D9.svg?style=for-the-badge)
![Gemma 2B](https://img.shields.io/badge/Gemma_2B-%23226D85.svg?style=for-the-badge)
![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-%233F7E6B.svg?style=for-the-badge&logo=python&logoColor=white)


## Instalação

1. Clone este repositório

2. Instale as dependências:
    ```bash
    pip install requirements.txt

## Uso
1. Execute o aplicativo Streamlit:
    ```bash
    streamlit run app.py

2. Acesse a interface web que será aberta no seu navegador. Fale a descrição para a música e ajuste a duração desejada, a musica será gerada assim que terminar de falar.

3. Após a geração, você poderá ouvir a música diretamente na interface e também fazer o download do arquivo de áudio gerado.

## Código
### Estrutura do Projeto
- `app.py`: Arquivo principal contendo o código do aplicativo Streamlit.
- `voce.py`: Arquivo contendo a parte de conversão do audio em texto
- `results_audios/`: Pasta onde os arquivos de áudio gerados são salvos.
  

