import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('winequality-red.csv', sep=';')
except:
    df = pd.read_csv('winequality-red.csv')

# Calculate correlation matrix
corr_matrix = df.corr()

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação de Pearson (Pearson Correlation Heatmap)', fontsize=16)
plt.tight_layout()

# Save the plot
plt.savefig('correlation_heatmap.png')
plt.close()

# Print correlation with quality to discuss in text
print("Correlação com a variável target 'quality':")
print(corr_matrix['quality'].sort_values(ascending=False))