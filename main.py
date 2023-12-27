import asyncio
import logging
from aiogram import Bot, Dispatcher, types
import matplotlib.pyplot as plt
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile
from io import BytesIO
import numpy as np


class MeanEmbeddingVectorizer(object):
    """Get mean of vectors"""
    def __init__(self, model):
        self.word2vec = model
        self.dim = model.vector_size

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec.get_vector(w)
                for w in words if w in self.word2vec] or
                [np.zeros(self.dim)], axis=0)
            for words in X])

#Импортируем кастомные модули
from ParseNewsModule import parse_news
from PredictionModule import prediction

with open('C:\\Users\\админ\\Desktop\\MyProject\\Neural_Networks_and_NLP\\Hometasks\\Telegramm_bot\\TokenFile.txt', 'r') as file:
    TOKEN = file.read()

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def start(message: types.Message):
    await message.answer("Привет! Я бот для анализа новостей. Введи дату в формате ГГГГ-ММ-ДД:")

@dp.message()
async def process_date(message: types.Message):
    date_str = message.text  
    
    # Проверяем, соответствует ли введенная строка формату ГГГГ-ММ-ДД
    if not date_str or len(date_str) != 10 or date_str[4] != '-' or date_str[7] != '-':
        await message.answer("Неверный формат даты. Введи дату в формате ГГГГ-ММ-ДД.")
        return
        
    # Оповещаем пользователя о начале поиска
    await message.answer(f"Подождите минутку, ищу все новости за {date_str}.")
    
    # Парсим новости
    news = parse_news(date_str)

    # Предсказываем класс для каждой новости
    predictions = prediction(news)

    # Подсчитываем количество новостей в каждой категории
    counts = predictions['predict_topic'].value_counts()
    
    # Визуализация
    plt.figure(figsize=(12, 8))
    plt.bar(counts.index, counts)
    plt.xlabel('Категории')
    plt.ylabel('Количество новостей')
    plt.title('Распределение новостей по категориям')

    # Сохраняем график в байтовый объект
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    
    # Создаем объект BufferedInputFile из байтового объекта
    photo = BufferedInputFile(img_buf.getvalue(),  filename='chart.png')
    
    # Отправляем график пользователю
    await bot.send_photo(message.chat.id, photo=photo, caption='Распределение новостей по категориям')


    # Отправляем информацию по новостям
    for category, count in counts.items():
        # Получаем информацию о статьях для данной категории
        category_news = predictions[predictions['predict_topic'] == category]
        # Отправляем название и ссылку на каждую статью
        for index, row in category_news.iterrows():
            await message.answer(f"Категория: {category}\nНазвание: {row['title']}\nСсылка: {row['urls']}")
            
    # Отправляем сообщение "Хочешь больше новостей? Введи дату в формате ГГГГ-ММ-ДД."
    await message.answer("Хочешь больше новостей? Введи дату в формате ГГГГ-ММ-ДД.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(dp.start_polling(bot))
