import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from sklearn.model_selection import train_test_split # Импортируем train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score

data = pd.read_csv('spam.csv', encoding='latin-1')
data.rename(columns={"v1": "метка", "v2": "текст"}, inplace=True)
columns_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

# Вычисление количества 'ham' и 'spam' и downsampled
ham_data = data[data['метка'] == 'ham']
spam_data = data[data['метка'] == 'spam']
num_ham_to_remove = 2 * len(spam_data)
print(num_ham_to_remove)
ham_downsampled = ham_data.sample(n=num_ham_to_remove, random_state=56)

# Объединение и перемешивание данных
data_downsampled = pd.concat([ham_downsampled, spam_data])
data_downsampled = data_downsampled.sample(frac=1, random_state=56).reset_index(drop=True)

# Преобразование меток в числовые значения
data_downsampled['метка'] = data_downsampled['метка'].map({'ham': 0, 'spam': 1})

# Токенизация
data_downsampled['токены'] = data_downsampled['текст'].apply(lambda text: re.sub(r'[^\w\s]', '', text).split())

# Удаление стоп-слов
# Список стоп-слов
stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "youre", "youve", "youll", "youd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "shes", 'her', 'hers',
              'herself', 'it', "its", 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', "thatll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
              'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
              'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
              'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
              'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
              "don't", 'should', "shouldve", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
              'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
              "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
              "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
              "wouldn't"}
data_downsampled['токены'] = data_downsampled['токены'].apply(lambda tokens: [token for token in tokens if token not in stop_words])

# Создание словаря и преобразование текста в последовательности индексов
max_words = 5000  # Максимальное количество слов в словаре
tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")  # Токен для неизвестных слов
tokenizer.fit_on_texts(data_downsampled['токены'].apply(lambda x: ' '.join(x)))
word_index = tokenizer.word_index
data_downsampled['индексы'] = tokenizer.texts_to_sequences(data_downsampled['токены'])

# Длина последовательности
max_length = 50
padded_sequences = pad_sequences(data_downsampled['индексы'], maxlen=max_length, padding='post', truncating='post')


print("\nРаспределение классов:")
print(data_downsampled['метка'].value_counts())
print("\nПропущенные значения:")
print(data_downsampled.isnull().sum())
print("\nПример случайных строк:")
print(data_downsampled.sample(5))

# Вывод информации
print("Пример оригинального текста:")
print(data_downsampled['текст'][0])
print("\nПример токенизированного текста:")
print(data_downsampled['токены'][0])
print("\nПример текста после преобразования в последовательность индексов:")
print(data_downsampled['индексы'][0])
print("\nПример последовательности после padding:")
print(padded_sequences[0])
print("\nРазмер словаря:", len(word_index))
print("\nФорма padded_sequences:", padded_sequences.shape)

# Разделение данных на обучающий и тестовый наборы
X = padded_sequences
y = data_downsampled['метка'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

# Создание модели RNN с LSTM
vocabulary_size = 5000
embedding_dim = 128

model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=max_length))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(128))
model.add(Dropout(rate=0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Установка скорости обучения
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=12, batch_size=32, validation_split=0.2)

# Оценка модели
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
y_pred = (model.predict(X_test) > 0.5).astype("int32")
f1 = f1_score(y_test, y_pred)
print('Метрики классификации:')
print('-' * 30)
print('Precision (Точность): {:.4f}'.format(precision))
print('Recall (Полнота): {:.4f}'.format(recall))
print('F1-score (F1-метрика): {:.4f}'.format(f1))
print('-' * 30)
print('Accuracy (Доля верных ответов): {:.4f}'.format(accuracy))
print('Loss (Функция потерь): {:.4f}'.format(loss))

# Визуализация результатов (опционально)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпоха')
plt.legend(['Обучение', 'Валидация'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Потери модели')
plt.ylabel('Потери')
plt.xlabel('Эпоха')
plt.legend(['Обучение', 'Валидация'], loc='upper right')
plt.show()