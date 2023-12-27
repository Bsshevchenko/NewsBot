import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import requests

from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
from nltk.corpus import stopwords

stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так',
                    'вот', 'быть', 'как',
                    'в', '—', 'к', 'за', 'из', 'из-за',
                    'на', 'ок', 'кстати',
                    'который', 'мочь', 'весь',
                    'еще', 'также', 'свой',
                    'ещё', 'самый', 'ул', 'комментарий',
                    'английский', 'язык', 'россия', "сша", 'украина'])
    
   
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

def text_prep(text) -> str:  
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    lemmas = [_.lemma for _ in doc.tokens]
    words = [lemma for lemma in lemmas if lemma.isalpha() and len(lemma) > 2]
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

def parse_news(date) -> str:
    info = []
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    for i in range(1):
        response = requests.get(url = f'https://lenta.ru/{date_obj.strftime("%Y/%m/%d")}/page/{i}/')
        tree = BeautifulSoup(response.content, 'html.parser')
        news_list = tree.find_all('li', {'class':'archive-page__item _news'})
        
        if news_list:
            for news in news_list:
                title = news.h3.text
                urls = 'https://lenta.ru' + news.a.get('href')
                
                response_content = requests.get(urls)
                tree_content = BeautifulSoup(response_content.content, 'html.parser')
                
                # contents, content представлен заголовками с разными тегами поэтому придется воспользоваться циклом чтобы
                contents = tree_content.find_all('p', {'class': 'topic-body__content-text'})
                contents_str = ' '.join(i.text for i in contents)
                
                row = {'urls': urls,
                       'title':title,
                       'content':contents_str,
                       'datetime':date_obj.strftime("%d.%m.%Y")
                       }
                
                info.append(row)
        else:
            break
    
    #preprocessing
    data = pd.DataFrame(info)
    
    data['datetime'] = pd.to_datetime(data.datetime, format="%d.%m.%Y")
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.strftime("%B")
    data['weekday'] = data['datetime'].dt.strftime('%A')
    data['len_title'] = data['title'].str.len()
    data['len_content'] = data['content'].str.len()
    data['title_clean'] = data.title.apply(text_prep)
    data['content_clean'] = data.content.apply(text_prep)
       
    return data
    
if __name__ == '__main__':
    parse_news_result = parse_news('2023-12-26')
    print(parse_news_result)