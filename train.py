import telebot
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex
import joblib
import spacy
import string
import re


bot = telebot.TeleBot('6453356937:AAFPQVL0Ft7doOVkB_uGk7ixsz48_-LMW0c')

#загружаем датасет - результат модели (будет файл)
data = pd.read_csv('train.csv', index_col=None, sep=";")[["Индекс", 'Вопрос', "Ответ"]]


t = AnnoyIndex(2000, metric = 'angular')

tfidf = TfidfVectorizer(max_features=2000, max_df = 0.9,
                        analyzer='word', ngram_range=(1, 2), 
                        token_pattern=r'(?u)\b[а-яА-ЯёЁa-zA-Z][а-яА-ЯёЁa-zA-Z]+\b',
                        lowercase = True, sublinear_tf=True)

lemmatizer = spacy.load('ru_core_news_md', disable = ['parser', 'ner'])
stopwords_nltk=[]

#===============================================================================
# Функции
#===============================================================================

def tfidf_featuring(tfidf, df):   
    '''Преобразование текста в мешок слов'''
    X_tfidf = tfidf.transform(df)
    
    return X_tfidf.toarray().tolist()

def predict_nns(find, df, n=1):
    user_embed = tfidf_featuring(tfidf, [find])[0]
    idx, dist = t.get_nns_by_vector(user_embed, 1, search_k=-1, include_distances=True)  
    if dist[0] < 0.8:
        return df['Ответ'][idx].values[0]
    else:
        return "Не могу ответить на вопрос"

def full_clean(s):
    #подготовка текста к подаче в модель
    s=re.sub(r"[^a-zA-Zа-яА-ЯйЙ#]", " ", s)   
    s = s.lower()
    s = re.sub(" +", " ", s) #оставляем только 1 пробел
    text = " ".join([token.lemma_ for token in lemmatizer(s) if token.lemma_ not in stopwords_nltk])
    
    return text

#===============================================================================
# Работа с текстом
#===============================================================================

data['clean'] = data['Вопрос'].apply(lambda x: full_clean(x))
tfidf.fit(data['clean'])
X_tfidf = tfidf_featuring(tfidf, data['clean'])
data['embed']=X_tfidf

#индекс
for user_id, user_embedding in enumerate(data['embed']):
    t.add_item(user_id, user_embedding)
t.build(-1)

#===============================================================================
# Сохранение
#===============================================================================
joblib.dump(tfidf, 'tfidf.pkl') 
data.to_csv("train_df.csv", index=None, sep=';')
#сохранение индекса
t.save('train.ann')
