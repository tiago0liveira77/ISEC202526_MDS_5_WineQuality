import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('winequality-red.csv', sep=';')

# 1. Quality Distribution Counts
quality_counts = df['quality'].value_counts().sort_index()
print("Distribuição de Quality:")
print(quality_counts)

# Verificar Nulos
print("\nValores Nulos por coluna:\n", df.isnull().sum())

# Identificar Disparidade de Escalas
print("\n Identificar Disparidade de Escalas\n" ,df[['total sulfur dioxide', 'density']].describe())
# Resultado: Sulfur vai até 289, Density apenas até 1.003

# 2. Threshold Analysis (Why >= 7?)
# Calculate count of 'Good' and 'Normal' for threshold 7
good_wines = df[df['quality'] >= 7].shape[0]
normal_wines = df[df['quality'] < 7].shape[0]

print(f"\nThreshold 7:")
print(f"Vinho Bom (>= 7): {good_wines}")
print(f"Vinho Normal (< 7): {normal_wines}")
print(f"Proportion Good: {good_wines / len(df):.2%}")

# 3. Visualizing the scarcity
plt.figure(figsize=(8, 5))
bars = plt.bar(quality_counts.index, quality_counts.values, color='skyblue', edgecolor='black')
plt.xlabel('Quality Score')
plt.ylabel('Number of Wines')
plt.title('Distribution of Wine Quality Scores')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight the split
plt.axvline(x=6.5, color='red', linestyle='--', linewidth=2, label='Threshold (>= 7)')
plt.legend()

# Save plot
plt.savefig('quality_distribution.png')

# Check extremes
extremes_low = df[df['quality'] <= 4].shape[0]
extremes_high = df[df['quality'] >= 8].shape[0]
print(f"\nExtremos Baixos (<=4): {extremes_low}")
print(f"Extremos Altos (>=8): {extremes_high}")

# Normalizar dataset - StandardScaler
# Separar Features (X) do Target (y)
X = df.drop('quality', axis=1)
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
