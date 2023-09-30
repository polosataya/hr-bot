import telebot
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from annoy import AnnoyIndex
import joblib
import spacy
import re
from datetime import datetime, timedelta
import os

dtn = datetime.now() + timedelta(hours=5) # Дата
bot = telebot.TeleBot('тут_токен')
admin = [тут_id]

data = pd.read_csv('train_df.csv', index_col=None, sep=";")[["Индекс", 'Вопрос', "Ответ", "embed"]]

t = AnnoyIndex(2000, 'angular')
t.load('train.ann') 
tfidf = joblib.load('tfidf.pkl')
lemmatizer = spacy.load('ru_core_news_md', disable = ['parser', 'ner'])
stopwords_nltk=[]

#===============================================================================
# Работа с текстом
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

#=========================================================================================
# Бот
#=========================================================================================

def log(message):
    botlogfile = open('HRBot.log', 'a', encoding="utf8")
    stats = os.stat('HRBot.log')
    if stats.st_size >= 1e+8: # При достижении 100 мб файл будет скинут в тг и удалён с сервера
        bot.send_document(admin, document = open("HRBot.log","rb"), caption=dtn.strftime('%Y-%m-%d %H:%M:%S'))
        os.remove('HRBott.log')
    else:
        print("------",file=botlogfile)
        print(dtn.strftime('%Y-%m-%d %H:%M:%S'),file=botlogfile)
        print("Сообщение от {0} (id = {2}) \n {3}".format(message.from_user.first_name, message.from_user.last_name, str(message.from_user.id), message.text), file=botlogfile)
        botlogfile.close()

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/start":
        bot.send_message(message.from_user.id, "Привет, этот бот поможет разобраться в кадровой и корпоративной политике Smart Consulting.")
        log(message)
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Напиши в сообщении вопрос.")
        log(message)
    else:
        # строка, которую вводит сотрудник
        find = full_clean(message.text)
        result = predict_nns(find, data)
        bot.send_message(message.from_user.id, result, parse_mode="Markdown")
        log(message)

bot.polling(none_stop=True, interval=0)