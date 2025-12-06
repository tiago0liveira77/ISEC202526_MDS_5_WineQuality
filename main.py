from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
import csv
import os
from datetime import datetime


LOG_FILE = "production_logs.csv"


def log_prediction(input_data: dict, prediction: int):
    """
    Guarda os dados recebidos num CSV para monitorização futura (Data Drift).
    """
    # Adicionar timestamp e a previsão feita
    log_entry = input_data.copy()
    log_entry['timestamp'] = datetime.now().isoformat()
    log_entry['prediction'] = prediction

    # Verificar se o ficheiro existe para escrever o cabeçalho
    file_exists = os.path.isfile(LOG_FILE)

    try:
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()  # Escreve o header se for a primeira vez
            writer.writerow(log_entry)
    except Exception as e:
        print(f"Erro ao guardar log: {e}")

# --- 1. CONFIGURAÇÃO E DADOS DE EXEMPLO (TEMPLATES) ---

# Template de Vinho Premium (Quality >= 7) - Baseado nas médias reais
sample_premium = {
    "fixed_acidity": 8.85,
    "volatile_acidity": 0.41,
    "citric_acid": 0.38,
    "residual_sugar": 2.71,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 14.0,
    "total_sulfur_dioxide": 35.0,
    "density": 0.996,
    "pH": 3.29,
    "sulphates": 0.74,
    "alcohol": 11.52
}

# Template de Vinho Mau (Quality <= 4) - Baseado nas médias reais
sample_bad = {
    "fixed_acidity": 7.87,
    "volatile_acidity": 0.72,
    "citric_acid": 0.17,
    "residual_sugar": 2.68,
    "chlorides": 0.09,
    "free_sulfur_dioxide": 12.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.997,
    "pH": 3.38,
    "sulphates": 0.59,
    "alcohol": 10.22
}

# --- 2. INICIALIZAÇÃO ---

app = FastAPI(
    title="Wine Quality API (TDSP)",
    description="API para classificação de vinho tinto (Premium vs Normal). Inclui templates de teste.",
    version="1.0.0"
)

# Carregar Artefatos (Modelo e Scaler)
try:
    model = joblib.load('model_rf.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Artefatos carregados com sucesso.")
except FileNotFoundError:
    print("❌ ERRO CRÍTICO: 'model_rf.pkl' ou 'scaler.pkl' não encontrados.")
    print("   Executa o 'train.py' primeiro para gerar os ficheiros.")
    model = None
    scaler = None


# --- 3. DEFINIÇÃO DE ESQUEMAS (Pydantic) ---

class WineInput(BaseModel):
    # Usamos o Field com exemplos para aparecer bonito no Swagger UI
    fixed_acidity: float = Field(..., example=8.85, description="Acidez fixa")
    volatile_acidity: float = Field(..., example=0.41, description="Acidez volátil (indicador de defeito)")
    citric_acid: float = Field(..., example=0.38)
    residual_sugar: float = Field(..., example=2.71)
    chlorides: float = Field(..., example=0.076)
    free_sulfur_dioxide: float = Field(..., example=14.0)
    total_sulfur_dioxide: float = Field(..., example=35.0)
    density: float = Field(..., example=0.996)
    pH: float = Field(..., example=3.29)
    sulphates: float = Field(..., example=0.74, description="Sulfatos (Conservantes)")
    alcohol: float = Field(..., example=11.52, description="Teor Alcoólico (% vol)")

    # Configuração extra para o Swagger mostrar os dois exemplos no dropdown
    model_config = {
        "json_schema_extra": {
            "examples": [sample_premium, sample_bad]
        }
    }


# --- 4. LÓGICA DE PREDIÇÃO (Reutilizável) ---

def run_inference(data_dict):
    """
    Orquestrador do pipeline de inferência.
    Responsável por validar, transformar e classificar uma única instância de vinho.
    """

    # 1. Fail-Fast Check: Validação de Dependências
    # Garante que a API não tenta processar nada se os artefatos não estiverem carregados.
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Serviço indisponível: Modelo não carregado.")

    # 2. Adaptação de Dados (Data Ingestion)
    # Converte o dicionário (JSON) num DataFrame pandas, formato esperado pelo Scikit-Learn.
    input_df = pd.DataFrame([data_dict])

    # 3. Pré-processamento (Feature Scaling)
    # Aplica a mesma transformação matemática (Média/Desvio Padrão) usada no treino.
    # O bloco try-catch protege a API contra dados numéricos inválidos ou colunas em falta.
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro de Schema/Normalização: {str(e)}")

    # 4. Execução do Modelo (Core Logic)
    # .predict() devolve a classe vencedora (0 ou 1)
    pred_class = model.predict(input_scaled)[0]

    # .predict_proba() devolve as probabilidades de cada classe: [Prob_Normal, Prob_Premium]
    # Exemplo: [0.15, 0.85] significa 15% certeza que é Normal, 85% que é Premium.
    probs = model.predict_proba(input_scaled)[0]

    # 5. Extração da Confiança
    # Selecionamos a probabilidade correspondente à classe que ganhou.
    # Se pred_class for 1, pegamos o valor do índice 1.
    confidence = probs[pred_class]

    # 6. MLOps: Monitorização (Data Drift)
    # Regista o input e o output para auditoria futura (CSV).
    log_prediction(data_dict, int(pred_class))

    # 7. Formatação da Resposta (Human-Readable)
    label = "Premium" if pred_class == 1 else "Normal"

    # 8. Construção do Contrato de Resposta (DTO)
    return {
        "status": "success",
        # Retornamos um resumo dos inputs principais para confirmação
        "input_summary": {
            "alcohol": data_dict['alcohol'],
            "sulphates": data_dict['sulphates']
        },
        "prediction": {
            "label": label,
            "class_id": int(pred_class),
            "confidence": round(float(confidence), 4)  # Arredondamento para leitura limpa
        },
        "message": "Vinho de Excelência detetado!" if pred_class == 1 else "Vinho Standard."
    }


# --- 5. ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "Wine Quality API is running. Go to /docs for testing."}


# Endpoint Principal (POST)
@app.post("/predict", tags=["Production"])
def predict_quality(wine: WineInput):
    """
    Endpoint de produção que recebe um JSON completo e devolve a classificação.
    """
    return run_inference(wine.dict())


# Endpoints de Teste Rápido (GET)
@app.get("/test/premium", tags=["Testing"])
def test_premium_sample():
    """
    Executa uma predição usando o Template de Vinho Premium.
    Não requer input.
    """
    return run_inference(sample_premium)


@app.get("/test/bad", tags=["Testing"])
def test_bad_sample():
    """
    Executa uma predição usando o Template de Vinho Mau.
    Não requer input.
    """
    return run_inference(sample_bad)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)