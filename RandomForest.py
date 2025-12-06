import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Carregar os dados
try:
    df = pd.read_csv('winequality-red.csv', sep=';')
except:
    df = pd.read_csv('winequality-red.csv')

# 2. Engenharia de Atributos (Feature Engineering)
# Criar target binário: 1 se quality >= 7 (Premium), 0 caso contrário (Normal)
df['quality_binary'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Separar Features (X) e Target (y)
# Removemos 'quality' (original) e 'quality_binary' (target) das features
X = df.drop(['quality', 'quality_binary'], axis=1)
y = df['quality_binary']

# 3. Divisão de Dados (Data Splitting) - Estratificada
# 80% Treino, 20% Teste, mantendo a proporção de classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature Scaling (Padronização)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Modelagem (Random Forest)
# class_weight='balanced' para lidar com o desbalanceamento
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# Previsões
y_pred = rf_model.predict(X_test_scaled)

# 6. Avaliação e Gráficos

# Gráfico 1: Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal (<7)', 'Premium (>=7)'],
            yticklabels=['Normal (<7)', 'Premium (>=7)'])
plt.title('Matriz de Confusão (Random Forest)', fontsize=14)
plt.xlabel('Previsão do Modelo')
plt.ylabel('Classe Real')
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png')
plt.show()

# Gráfico 2: Feature Importance
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances, palette='viridis')
plt.title('Importância das Variáveis (Feature Importance)', fontsize=14)
plt.xlabel('Importância Relativa')
plt.ylabel('Variáveis')
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
plt.show()

# Relatório de Classificação (Texto)
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Premium']))