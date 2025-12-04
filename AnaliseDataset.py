import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('winequality-red.csv', sep=';')
except:
    df = pd.read_csv('winequality-red.csv')

# Set aesthetic style
sns.set(style="whitegrid")

# 1. Visualization of Missing Values (Heatmap)
plt.figure(figsize=(10, 6))
# Create a heatmap of boolean values (True for null, False for not null)
# Since there are no nulls, it will be a uniform color, but this visualizes the check.
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Mapa de Calor de Valores Nulos (Missing Values Heatmap)\n(Área uniforme indica zero nulos)', fontsize=14)
plt.xlabel('Colunas')
plt.ylabel('Índice das Amostras')
plt.tight_layout()
plt.savefig('missing_values_heatmap.png')
plt.close() # Close to start a new figure

# 2. Visualization of Scales (Boxplot)
plt.figure(figsize=(14, 8))
# Exclude 'quality' for feature scale comparison, or keep it. Let's keep features only.
features = df.drop('quality', axis=1)
sns.boxplot(data=features, orient="h", palette="Set2")
plt.title('Comparação de Escalas das Variáveis (Boxplot)', fontsize=16)
plt.xlabel('Valor')
plt.ylabel('Features')
plt.xscale('log') # Log scale helps visualize vast differences better if they are extreme
# Let's try linear scale first to really show the "problem", but log is often "more appealing" for reading.
# The user wants to SEE the different scales. A linear scale shows 'total sulfur dioxide' huge and 'density' tiny.
# A log scale makes them comparable. Let's stick to linear to prove the point of "different scales".
# Actually, let's do linear, because the goal is to show WHY we need normalization.
plt.xscale('linear')
plt.tight_layout()
plt.savefig('feature_scales_boxplot.png')
plt.close()

# Let's generate a second version of scales plot without 'total sulfur dioxide' just to see the others better?
# No, the "appealing" part is showing the disparity.
# Let's create a Violin plot as well, it's often considered more "appealing" and informative.
plt.figure(figsize=(14, 8))
# Standardize data just for visualization of distribution shapes (optional),
# but user asked to show *different scales*, so raw data is needed.
sns.violinplot(data=features, orient="h", palette="muted", inner="quartile")
plt.title('Distribuição e Escala das Variáveis (Violin Plot)', fontsize=16)
plt.xlabel('Valor')
plt.tight_layout()
plt.savefig('feature_scales_violin.png')