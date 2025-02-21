import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
file_path = "german_transformado.xlsx"
df = pd.read_excel(file_path)

# Pré-processamento: Remover valores nulos
df.dropna(inplace=True)

# Codificar atributos categóricos
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separar features e target
target_column = "Valor_credito"
X = df.drop(columns=[target_column])
y = df[target_column]

# Normalizar os dados numéricos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir conjunto de dados
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

# Definir o modelo Random Forest Regressor
model = RandomForestRegressor(random_state=42)

# Validação cruzada
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("\n" + "=" * 60)
print("Validação Cruzada")
print("=" * 60)
print(f'Média do R²: {cv_scores.mean():.4f}')
print(f'Desvio padrão do R²: {cv_scores.std():.4f}')
print(f'R² em cada fold: {cv_scores}')

# Otimização de hiperparâmetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("\n" + "=" * 60)
print("Otimização de Hiperparâmetros")
print("=" * 60)
print(f'Melhores hiperparâmetros: {grid_search.best_params_}')

# Avaliação final
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("Avaliação Final")
print("=" * 60)
print(f'Erro Médio Absoluto (MAE): {mae:.4f}')
print(f'Erro Quadrático Médio (MSE): {mse:.4f}')
print(f'Coeficiente de Determinação (R²): {r2:.4f}')

# Importância das Features
feature_importances = best_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\n" + "=" * 60)
print("Importância das Features")
print("=" * 60)
print(importance_df)

# Gráfico de Importância das Features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Importância das Features')
plt.show()
