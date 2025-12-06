# 1. Imagem base (Python leve)
FROM python:3.9-slim

# 2. Definir a pasta de trabalho dentro do container
WORKDIR /app

# 3. Copiar primeiro as dependências (para aproveitar a cache do Docker)
COPY requirements.txt .

# 4. Instalar as bibliotecas
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar o resto do código (main.py, pickles, csv) para dentro do container
COPY . .

# 6. Expor a porta que o FastAPI usa
EXPOSE 8000

# 7. Comando para iniciar a API quando o container arrancar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t wine-quality-api .
# docker run -p 8000:8000 wine-quality-api
