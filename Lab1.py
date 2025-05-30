import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np



data = pd.read_csv('bananas.csv', sep=',')
print(data.head())

# Предобработка данных (стандартизация признаков)
scaler = StandardScaler()
X = data.drop('Quality', axis=1)
y = data['Quality']
X = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Обучение модели
model = GaussianNB()
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_predict_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_predict_test)
print(f"Оценка модели на тестовой выборке: {accuracy_test}")
print(f"Отчет о классификации на тестовой выборке:\n {classification_report(y_test, y_predict_test)}")

# Предсказание на обучающей выборке
y_predict_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_predict_train)
print(f"Оценка модели на обучающей выборке: {accuracy_train}")

# Подсчет количества и процента неверных классификаций на обучающей выборке
total_num = len(y_train)
incorrect_classifications_num = (y_train != y_predict_train).sum()
incorrect_classifications = (incorrect_classifications_num / total_num) * 100
print(f"Количество Неверных классификаций на обучающей выборке: {incorrect_classifications_num} из {total_num}")
print(f"Процент неверных классификаций на обучающей выборке: {incorrect_classifications:.2f}%")


# ГРАФИК
# Признаки для визуализации - размер и вес
feature1 = 0
feature2 = 1

# Создание графика
plt.figure(figsize=(18, 10))  # Размер графика

# Отображение точек
for quality in y_train.unique():
    # Точки для текущего класса (это либо Good или Bad)
    class_indices = (y_train == quality)
    plt.scatter(X_train[class_indices, feature1], X_train[class_indices, feature2],
                label=quality,
                marker='o' if quality == 'Good' else 'x',  # Настройка маркеров
                color='yellow' if quality == 'Good' else 'green',  # Выбор цвета
                alpha=0.7)


# ГРАНИЦА
# Подсчет среднего значения признаков для каждого класса
means = {}
for quality in y_train.unique():
    class_indices = (y_train == quality)
    means[quality] = (X_train[class_indices, feature1].mean(), X_train[class_indices, feature2].mean())

# Нахождение центра (середина между средними)
mean_good = means['Good']
mean_bad = means['Bad']
center_x = (mean_good[0] + mean_bad[0]) / 2
center_y = (mean_good[1] + mean_bad[1]) / 2

#  Нахождение угла между прямой, соединяющей средние точки, и осью X
angle = np.arctan2(mean_good[1] - mean_bad[1], mean_good[0] - mean_bad[0])

#  Отрисовка отрезка, перпендикулярного этой прямой, проходящего через центр
line_length = 2
x_min, x_max = X_train[:, feature1].min(), X_train[:, feature1].max()
y_min, y_max = X_train[:, feature2].min(), X_train[:, feature2].max()

#  Задаем точки для отрезка
x_line = [center_x - line_length * np.sin(angle), center_x + line_length * np.sin(angle)]
y_line = [center_y + line_length * np.cos(angle), center_y - line_length * np.cos(angle)]

# Отрисовка линии
plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=3, label='Граница')

# Точки средних по классам (для наглядности)
plt.scatter(mean_good[0], mean_good[1], color='#540e8a', marker='*', s=100, label='Среднее Good')
plt.scatter(mean_bad[0], mean_bad[1], color='#052e0b', marker='*', s=100, label='Среднее Bad')


# Подписи к осям
plt.xlabel(data.columns[feature1])
plt.ylabel(data.columns[feature2])
plt.title('Обучающая выборка')
plt.legend()

# Отображение графика
plt.grid(True)
plt.show()