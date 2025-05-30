import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, \
    accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler

data = pd.read_csv('bananas.csv', sep=',')
print(data.head())

# Среднее значение, медиана, экстремумы параметра - Сладость
sweetness_stats = {
    'Среднее': data['Sweetness'].mean(),
    'Медиана': data['Sweetness'].median(),
    'Минимум': data['Sweetness'].min(),
    'Максимум': data['Sweetness'].max()
}

print("Параметр 'Сладость':")
for key, value in sweetness_stats.items():
    print(f"{key}: {value}")

# Удаление дубликатов, размер датасета
dubl_data_num = data.duplicated().sum()
start_data_size = data.shape
data.drop_duplicates(inplace=True)
del_data_size = data.shape
print(f"\nЧисло дубликатов: {dubl_data_num};\nНачальный размер датасета: {start_data_size};\n"
      f"Итоговый размер датасета: {del_data_size}")

# Проверка пустых значений
print("Количество пропущенных значений до заполнения:")
print(data.isnull().sum())

# Заполнение пустых значений медианой
for column in data.columns:
    if pd.api.types.is_numeric_dtype(data[column]):
        median_val = data[column].median()  # Вычисляем медиану столбца (параметра)
        data[column] = data[column].fillna(median_val)

print("Количество пропущенных значений после заполнения:")
print(data.isnull().sum())

# Обработка шумов
data_filtered = data.copy()  # Создаем копию DataFrame, чтобы не менять исходный

# Сладость должна быть положительной
data_filtered = data_filtered[(data_filtered['Sweetness'] >= 0)]
print(f"Размер данных после фильтрации по сладости: {data_filtered.shape}")

# min_sweetness = data['Sweetness'].min()  # Находим минимальное значение
# if min_sweetness < 0:
#     offset = abs(min_sweetness)  # Вычисляем смещение
#     data['Sweetness'] = data['Sweetness'] + offset
#     print(f"К столбцу 'Sweetness' добавлено смещение {offset} для перевода в положительную область.")
# else:
#     print("В столбце 'Sweetness' нет отрицательных значений.")

# Среднее значение, медиана, экстремумы параметра - Сладость
sweetness_stats = {
    'Среднее': data['Sweetness'].mean(),
    'Медиана': data['Sweetness'].median(),
    'Минимум': data['Sweetness'].min(),
    'Максимум': data['Sweetness'].max()
}

print("Параметр 'Сладость':")
for key, value in sweetness_stats.items():
    print(f"{key}: {value}")

# min_size = data['Size'].min()  # Находим минимальное значение
# if min_size < 0:
#     offset = abs(min_size)  # Вычисляем смещение
#     data['Size'] = data['Size'] + offset
#     print(f"К столбцу 'Size' добавлено смещение {offset} для перевода в положительную область.")
# else:
#     print("В столбце 'Size' нет отрицательных значений.")
#
# min_weight = data['Weight'].min()  # Находим минимальное значение
# if min_weight < 0:
#     offset = abs(min_weight)  # Вычисляем смещение
#     data['Weight'] = data['Weight'] + offset
#     print(f"К столбцу 'Weight' добавлено смещение {offset} для перевода в положительную область.")
# else:
#     print("В столбце 'Weight' нет отрицательных значений.")

size_stats = {
    'Среднее': data['Size'].mean(),
    'Медиана': data['Size'].median(),
    'Минимум': data['Size'].min(),
    'Максимум': data['Size'].max()
}
print("\nПараметр 'Size':")
for key, value in size_stats.items():
    print(f"{key}: {value}")

weight_stats = {
    'Среднее': data['Weight'].mean(),
    'Медиана': data['Weight'].median(),
    'Минимум': data['Weight'].min(),
    'Максимум': data['Weight'].max()
}
print("\nПараметр 'Weight':")
for key, value in weight_stats.items():
    print(f"{key}: {value}")
print('\n')

# Расчет дисперсии и стандартного отклонения
sweetness_variance = data['Sweetness'].var()
sweetness_std = data['Sweetness'].std()
print(f"Дисперсия параметра 'Сладость': {sweetness_variance}")
print(f"Стандартное отклонение параметра 'Сладость': {sweetness_std}")

size_variance = data['Size'].var()
print(f"Дисперсия параметра 'Размер': {size_variance}")

# Линейная зависимость
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Size', y='Sweetness', data=data)
plt.title('Зависимость между размером и сладостью')
plt.xlabel('Размер')
plt.ylabel('Сладость')
plt.show()

# Расчет коэффициента корреляции Пирсона
correlation = data['Size'].corr(data['Sweetness'])
print(f"Коэффициент корреляции Пирсона между размером и сладостью: {correlation}")

if correlation > 0:
    print("Положительная линейная зависимость: больше размер, больше сладость")
elif correlation < 0:
    print("Отрицательная линейная зависимость: больше размер, меньше сладость")
else:
    print("Линейная зависимость между размером и сладостью не обнаружена")

# Построение гистограммы для параметра "Сладость"
plt.figure(figsize=(10, 6))
sns.histplot(data['Sweetness'], kde=True, bins=30)
plt.title('Распределение параметра "Сладость"')
plt.xlabel('Сладость')
plt.ylabel('Частота')
plt.show()

# Регрессия
# Преобразование строковых значений качества в числовые
quality_mapping = {
    'Bad': 0,
   'Good': 1
}
data['Quality'] = data['Quality'].map(quality_mapping)

# Выбор признаков (features) и целевой переменной (target)
# features = data[['Size', 'Weight', 'Sweetness','Softness','HarvestTime','Ripeness','Acidity']]
features = data[['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity']]
target = data['Quality']
# 'Softness',
# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=116)

print('')


# # Применение RobustScaler
# robust_scaler = RobustScaler()
# X_train_scaled = robust_scaler.fit_transform(X_train)
# X_test_scaled = robust_scaler.transform(X_test)

# Стандартизация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print("Средние значения признаков:")
# print(scaler.mean_)
# print("\nСтандартные отклонения признаков:")
# print(scaler.scale_)

# # Вычисляем средние значения масштабированных данных
# mean_scaled = np.mean(X_train_scaled, axis=0)
# std_scaled = np.std(X_train_scaled, axis=0)
#
# print("\nСредние значения признаков (после масштабирования):")
# print(mean_scaled)
# print("\nСтандартные отклонения признаков (после масштабирования):")
# print(std_scaled)

# Обучение модели
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# # Обучение модели LinearRegression с Ridge регуляризацией
# # (Ridge помогает, даже если используется линейная регрессия для классификации)
# model = Ridge(alpha=0.01)  # Подберите alpha с помощью GridSearchCV
# model.fit(X_train_scaled, y_train)

# Предсказание модели
y_predict = model.predict(X_test_scaled)

# Обработка предсказаний в бинарные классы
threshold = 0.5
y_predict_binary = (y_predict > threshold).astype(int)
y_test_binary = y_test.astype(int)

print('')

# Оценка модели (метрики регрессии)
ms_error = mean_squared_error(y_test, y_predict)
ma_error = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
print(f"Средняя квадратическая ошибка: {ms_error}")
print(f"Средняя абсолютная ошибка: {ma_error}")
print(f"Коэффициент детерминации (R2): {r2}")
precision = precision_score(y_test, y_predict_binary)
recall = recall_score(y_test, y_predict_binary)
f1 = f1_score(y_test_binary, y_predict_binary)
print(f"Точность: {precision}")
print(f"Полнота: {recall}")
print(f"F1-метрика: {f1}")

# Коэффициенты модели
coefficients = pd.DataFrame(model.coef_, features.columns, columns=['Коэффициенты'])
print("\nКоэффициенты модели:")
print(coefficients)

