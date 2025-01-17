# Usar uma imagem base oficial do Python
FROM python:3.10-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar o arquivo requirements.txt para o container
COPY requirements.txt .

# Instalar as dependências
RUN pip install -r requirements.txt

# Copiar o restante do código para o container
COPY . .

# Expor a porta usada pelo Streamlit
EXPOSE 8501

# Comando para rodar o aplicativo Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]