import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('winequality-red.csv', sep=';')
except:
    df = pd.read_csv('winequality-red.csv')

# Create binary label
df['quality_label'] = df['quality'].apply(lambda x: 'Bom (>=7)' if x >= 7 else 'Normal (<7)')

# Set aesthetic style
sns.set(style="whitegrid")

# Create a figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Original Quality Distribution
sns.countplot(x='quality', data=df, ax=axes[0], palette='Blues_d')
axes[0].set_title('Distribuição Original das Notas (3-8)', fontsize=14)
axes[0].set_xlabel('Nota de Qualidade')
axes[0].set_ylabel('Contagem')
# Add count labels
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                     textcoords='offset points')

# Plot 2: Binary Class Distribution
sns.countplot(x='quality_label', data=df, ax=axes[1], palette=['#e74c3c', '#2ecc71']) # Red for Normal, Green for Good
axes[1].set_title('Desbalanceamento após Binarização (Target)', fontsize=14)
axes[1].set_xlabel('Categoria')
axes[1].set_ylabel('Contagem')
# Add count labels
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                     textcoords='offset points')

plt.tight_layout()
plt.savefig('class_imbalance.png')
print("Plot saved as class_imbalance.png")