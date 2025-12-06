import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Configurações
DATASET_PATH = 'winequality-red.csv'
MODEL_PATH = 'model_rf.pkl'
SCALER_PATH = 'scaler.pkl'
RANDOM_STATE = 42


def load_data(path):
    """Carrega o dataset assumindo separador ponto e vírgula."""
    try:
        # O dataset original usa ';' como separador
        df = pd.read_csv(path, sep=';')

        # --- LINHA NOVA: CORREÇÃO DE NOMES ---
        # Substituir espaços por underscores (ex: "fixed acidity" -> "fixed_acidity")
        df.columns = df.columns.str.replace(' ', '_')
        # -------------------------------------

        print(f"✅ Dados carregados com sucesso! Dimensão: {df.shape}")
        # Verificar se funcionou
        print(f"   Colunas: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"❌ Erro: Ficheiro '{path}' não encontrado.")
        exit()


def feature_engineering(df):
    """Aplica a transformação de negócio (Binarização) e separa X/y."""
    # 1. Criar Target Binário: 1 (Premium) se quality >= 7, senão 0 (Normal)
    df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

    # 2. Separar Features e Target
    # Removemos a 'quality' original (para não viciar) e o target criado
    X = df.drop(['quality', 'quality_label'], axis=1)
    y = df['quality_label']

    print("✅ Feature Engineering concluída.")
    print(f"   Distribuição de classes: {y.value_counts().to_dict()}")
    return X, y


def train_model():
    # 1. Carregar Dados
    df = load_data(DATASET_PATH)

    # 2. Preparar Dados
    X, y = feature_engineering(df)

    # 3. Split Estratificado (Crucial devido ao desbalanceamento)
    # Divide 80% Treino / 20% Teste mantendo a proporção de vinhos Premium
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 4. Normalização (StandardScaler)
    # Ajustamos (fit) apenas no treino para evitar data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Treino do Modelo (Random Forest)
    # class_weight='balanced' ajusta os pesos para penalizar erros na classe minoritária
    print("🔄 A treinar modelo Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    rf_model.fit(X_train_scaled, y_train)

    # 6. Avaliação
    y_pred = rf_model.predict(X_test_scaled)

    print("\n" + "=" * 40)
    print("📊 RELATÓRIO DE AVALIAÇÃO (Test Set)")
    print("=" * 40)
    print(classification_report(y_test, y_pred, target_names=['Normal (<7)', 'Premium (>=7)']))
    print(f"Accuracy Global: {accuracy_score(y_test, y_pred):.4f}")

    # 8. Serialização (Salvar Artefatos para Deploy)
    joblib.dump(rf_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✅ Artefatos salvos com sucesso:")
    print(f"   - Modelo: {MODEL_PATH}")
    print(f"   - Scaler: {SCALER_PATH}")


if __name__ == "__main__":
    train_model()