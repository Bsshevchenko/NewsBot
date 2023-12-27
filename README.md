# Telegram **NewsBot 1.0.0**

В данном репозитории содержится код **телеграмм бота** для анализа новостей с сайта [Lenta.ru](https://lenta.ru/). 

### Принцип работы бота

1. Пользователь вводит дату в формате ГГГГ-ММ-ДД, дата выбирается от **2000-01-01** по **настоящую дату**

2. Бот скрапит сайт [Lenta.ru](https://lenta.ru/) на выбранную дату и выводит в чат гистограмму распределения новостей по категориям за выбранную дату, а так же список новостей. Процесс занимает 1-1.5 минуты.

    ![Alt text](image.png)

3. Чтобы продолжить работу с ботом, пользователь еще раз вводит дату в формате ГГГГ-ММ-ДД

### Структура бота

Проект состоит из трех модулей ``` main.py```, ``` ParserNewsModule.py```, ``` PredictionModule.py```

* В модуле ``` main.py``` содержится сам бот.
* В модуле ``` ParserNewsModule.py``` содержится парсер и предобработка данных для корректного предсказания категорий новостей на основе содержания статьи.
* В модуле  ``` PredictionModule.py``` содержится модель многоклассовой классификации для предсказания категории статьи по ее содержанию.

### Библиотеки 

* asyncio
* aiogram
* matplotlib
* io
* pandas
* numpy
* datetime
* requests
* BeautifulSoup
* joblib
* logging


