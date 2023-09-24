import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Generamos algunos datos de ejemplo
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos un modelo de regresión lineal
lin_reg = LinearRegression()

# Entrenamos el modelo de regresión lineal con los datos de entrenamiento
lin_reg.fit(X_train, y_train)

# Hacemos predicciones en el conjunto de prueba con regresión lineal
y_pred_lin_reg = lin_reg.predict(X_test)

# Calculamos el error cuadrático medio para la regresión lineal
mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)
print("Error Cuadrático Medio (Regresión Lineal):", mse_lin_reg)

# Creamos un modelo k-NN con k=3
knn_reg = KNeighborsRegressor(n_neighbors=3)

# Entrenamos el modelo k-NN con los datos de entrenamiento
knn_reg.fit(X_train, y_train)

# Hacemos predicciones en el conjunto de prueba con k-NN
y_pred_knn = knn_reg.predict(X_test)

# Calculamos el error cuadrático medio para k-NN
mse_knn = mean_squared_error(y_test, y_pred_knn)
print("Error Cuadrático Medio (k-NN):", mse_knn)

# Graficamos los resultados
plt.scatter(X_test, y_test, color='black', label='Datos Reales')
plt.plot(X_test, y_pred_lin_reg, color='blue', linewidth=3, label='Regresión Lineal')
plt.scatter(X_test, y_pred_knn, color='red', label='k-NN')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparación de Regresión Lineal y k-NN')
plt.legend()
plt.show()
