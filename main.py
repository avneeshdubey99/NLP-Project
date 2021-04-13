import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from plotly import graph_objs as go
import datetime as dt
import os
import tweepy as tw
import re
import nltk
from nltk.corpus import stopwords
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import praw
from datetime import timezone
import time
import requests
import json
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

from textblob import TextBlob
st.title('Sentiment Analysis on Publicly Traded Companies')

stocks = ('AMZN', 'DAL', 'KO')
selected_stock = st.selectbox('Select stock for analysis', stocks)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Load data ...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

def plot_raw_data():
	fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])
	fig.layout.update(title_text='Stock data with Rangeslider', xaxis_rangeslider_visible=True)
    
	st.plotly_chart(fig)
	
plot_raw_data()

data_1A = os.listdir(f'10k/{selected_stock}/10-K/Item 1A')
data_7 = os.listdir(f'10k/{selected_stock}/10-K/Item 7')

def get_filing_details(list_dir):
    df = pd.DataFrame(list_dir)
    df.columns = ['file_name']
    df['cik'] = df['file_name'].apply(lambda x : x.split('-')[0])
    df['year'] = df['file_name'].apply(lambda x : x.split('-')[1])
    df['accession_number'] = df['file_name'].apply(lambda x : x.split('.')[0])
    df['year'] = df['year'].astype(str).astype(int)
    df['year'] = df['year'] + 2000
    df.set_index(['year'], inplace=True)
    df.sort_index(axis=0, inplace=True, ascending=True)
    return df

def clean_text_str(text_str):
    """Returns a list of cleaned words for a given text string."""
    nltk_stopwords = set(stopwords.words('english'))
    words = text_str.split()
    words = [word.lower() for word in words if word.isalpha()]
    cleaned_words = [word for word in words if word not in nltk_stopwords]
    
    return cleaned_words

def get_cleaned_data(ticker):
    
    data_1A = os.listdir(f'10k/{ticker}/10-K/Item 1A')
    data_7 = os.listdir(f'10k/{ticker}/10-K/Item 7')
    data_1A_df = get_filing_details(data_1A)
    data_7_df = get_filing_details(data_7) 
    data_1A_df['Section_1A'] = ''
    data_7_df['Section_7'] = ''
    
    section_A1_text = {}
    section_7_text = {}
    count_1  = 0
    
    for file in data_1A_df['file_name'] :
        #year = file.split('-')[1]
        #year = int(year)
        #year = year + 2000
        with open(f'10k/{ticker}/10-K/Item 1A/' + file, encoding="utf8") as a:
            text = a.read()
        data_1A_df['Section_1A'].iloc[count_1] = text
        count_1 = count_1+1
        
    #a1 = pd.DataFrame(section_A1_text, index=[]).T
    #a1.columns = ['Section_1A']
    #data_1A_df = pd.concat([data_1A_df, a1], axis=1)
    
    count_2 = 0
    for file in data_7_df['file_name']:
        #year = file.split('-')[1]
        #year = int(year)
        #year = year + 2000
        with open(f'10k/{ticker}/10-K/Item 7/' + file, encoding="utf8") as a:
            text = a.read()
        data_7_df['Section_7'].iloc[count_2] = text
        count_2 = count_2+1
        
    #a2 = pd.DataFrame(section_7_text, index=['text']).T
    #a2.columns = ['Section_7']
    #data_7_df = pd.concat([data_7_df, a2], axis=1)
    
    data_1A_df['Section_1A_cleaned'] = data_1A_df['Section_1A'].apply(lambda b: clean_text_str(b))
    data_7_df['Section_7_cleaned'] = data_7_df['Section_7'].apply(lambda x: clean_text_str(x))
    
    data_1A_df['Section_1A_cleaned'] = data_1A_df['Section_1A_cleaned'].apply(lambda x: " ".join(x))
    data_7_df['Section_7_cleaned'] = data_7_df['Section_7_cleaned'].apply(lambda x: " ".join(x))
    
    data_1A_df['num_cleaned_words_Section_1A'] = data_1A_df['Section_1A_cleaned'].apply(lambda x: len(x))
    data_7_df['num_cleaned_words_Section_7'] = data_7_df['Section_7_cleaned'].apply(lambda x: len(x))
    
    data_all = data_1A_df
    data_all = pd.concat([data_all, data_7_df['Section_7'], data_7_df['Section_7_cleaned'], data_7_df['num_cleaned_words_Section_7']], axis=1)
    return data_all    

def get_LM_dictionary():
    negative = pd.read_csv('https://raw.githubusercontent.com/AayushTalekar/NLP/main/Negative.csv', header=None)
    positive = pd.read_csv('https://raw.githubusercontent.com/AayushTalekar/NLP/main/Positive.csv', header=None)
    constraining = pd.read_csv('https://raw.githubusercontent.com/AayushTalekar/NLP/main/Constraining.csv', header=None)
    uncertainity = pd.read_csv('https://raw.githubusercontent.com/AayushTalekar/NLP/main/Uncertainity.csv', header=None)
    weakmodal = pd.read_csv('https://raw.githubusercontent.com/AayushTalekar/NLP/main/Weak%20Modal.csv', header=None)
    strongmodal = pd.read_csv('https://raw.githubusercontent.com/AayushTalekar/NLP/main/Strong%20Modal.csv', header=None)
    litigious = pd.read_csv('https://raw.githubusercontent.com/AayushTalekar/NLP/main/Litigious.csv', header=None)
    negative.columns = ['Neg']
    positive.columns = ['Pos']
    constraining.columns = ['Cons']
    uncertainity.columns = ['Uncer']
    weakmodal.columns = ['WeakModal']
    strongmodal.columns = ['StrongModal']
    litigious.columns = ['Litigious']
    neg = negative['Neg'].str.lower().tolist()
    pos = positive['Pos'].str.lower().tolist()
    cons = constraining['Cons'].str.lower().tolist()
    uncer = uncertainity['Uncer'].str.lower().tolist()
    weakmodal = weakmodal['WeakModal'].str.lower().tolist()
    strongmodal = strongmodal['StrongModal'].str.lower().tolist()
    litigious = litigious['Litigious'].str.lower().tolist()
    return neg, pos, cons, uncer, weakmodal, strongmodal, litigious

def get_sentiment_analysis(ticker):
    data = get_cleaned_data(ticker)
    neg, pos, cons, uncer, weakmodal, strongmodal, litigious = get_LM_dictionary()


    count_vec_pos = CountVectorizer(vocabulary=pos)
    count_vec_neg = CountVectorizer(vocabulary=neg)
    count_vec_cons = CountVectorizer(vocabulary=cons)
    count_vec_uncer = CountVectorizer(vocabulary=uncer)
    count_vec_weakmodal = CountVectorizer(vocabulary=weakmodal)
    count_vec_strongmodal = CountVectorizer(vocabulary=strongmodal)
    count_vec_litigious = CountVectorizer(vocabulary=litigious)


    dtm_pos_words_1A = count_vec_pos.fit_transform(data['Section_1A_cleaned'])
    dtm_pos_words_7 = count_vec_pos.fit_transform(data['Section_7_cleaned'])
    dtm_neg_words_1A = count_vec_neg.fit_transform(data['Section_1A_cleaned'])
    dtm_neg_words_7 = count_vec_neg.fit_transform(data['Section_7_cleaned'])
    dtm_cons_words_1A = count_vec_cons.fit_transform(data['Section_1A_cleaned'])
    dtm_cons_words_7 = count_vec_cons.fit_transform(data['Section_7_cleaned'])
    dtm_uncer_words_1A = count_vec_uncer.fit_transform(data['Section_1A_cleaned'])
    dtm_uncer_words_7 = count_vec_uncer.fit_transform(data['Section_7_cleaned'])
    dtm_weakmodal_words_1A = count_vec_weakmodal.fit_transform(data['Section_1A_cleaned'])
    dtm_weakmodal_words_7 = count_vec_weakmodal.fit_transform(data['Section_7_cleaned'])
    dtm_strongmodal_words_1A = count_vec_strongmodal.fit_transform(data['Section_1A_cleaned'])
    dtm_strongmodal_words_7 = count_vec_strongmodal.fit_transform(data['Section_7_cleaned'])
    dtm_litigious_words_1A = count_vec_litigious.fit_transform(data['Section_1A_cleaned'])
    dtm_litigious_words_7 = count_vec_litigious.fit_transform(data['Section_7_cleaned'])


    df_dtm_pos_words_1A = pd.DataFrame(dtm_pos_words_1A.toarray(), index=data.index)
    df_dtm_pos_words_1A.columns = count_vec_pos.vocabulary_.keys()
    df_dtm_pos_words_7 = pd.DataFrame(dtm_pos_words_7.toarray(), index=data.index)
    df_dtm_pos_words_7.columns = count_vec_pos.vocabulary_.keys()
    df_dtm_neg_words_1A = pd.DataFrame(dtm_neg_words_1A.toarray(), index=data.index)
    df_dtm_neg_words_1A.columns = count_vec_neg.vocabulary_.keys()
    df_dtm_neg_words_7 = pd.DataFrame(dtm_neg_words_7.toarray(), index=data.index)
    df_dtm_neg_words_7.columns = count_vec_neg.vocabulary_.keys()
    df_dtm_cons_words_1A = pd.DataFrame(dtm_cons_words_1A.toarray(), index=data.index)
    df_dtm_cons_words_1A.columns = count_vec_cons.vocabulary_.keys()
    df_dtm_cons_words_7 = pd.DataFrame(dtm_cons_words_7.toarray(), index=data.index)
    df_dtm_cons_words_7.columns = count_vec_cons.vocabulary_.keys()
    df_dtm_uncer_words_1A = pd.DataFrame(dtm_uncer_words_1A.toarray(), index=data.index)
    df_dtm_uncer_words_1A.columns = count_vec_uncer.vocabulary_.keys()
    df_dtm_uncer_words_7 = pd.DataFrame(dtm_uncer_words_7.toarray(), index=data.index)
    df_dtm_uncer_words_7.columns = count_vec_uncer.vocabulary_.keys()
    df_dtm_weakmodal_words_1A = pd.DataFrame(dtm_weakmodal_words_1A.toarray(), index=data.index)
    df_dtm_weakmodal_words_1A.columns = count_vec_weakmodal.vocabulary_.keys()
    df_dtm_weakmodal_words_7 = pd.DataFrame(dtm_weakmodal_words_7.toarray(), index=data.index)
    df_dtm_weakmodal_words_7.columns = count_vec_weakmodal.vocabulary_.keys()
    df_dtm_strongmodal_words_1A = pd.DataFrame(dtm_strongmodal_words_1A.toarray(), index=data.index)
    df_dtm_strongmodal_words_1A.columns = count_vec_strongmodal.vocabulary_.keys()
    df_dtm_strongmodal_words_7 = pd.DataFrame(dtm_strongmodal_words_7.toarray(), index=data.index)
    df_dtm_strongmodal_words_7.columns = count_vec_strongmodal.vocabulary_.keys()
    df_dtm_litigious_words_1A = pd.DataFrame(dtm_litigious_words_1A.toarray(), index=data.index)
    df_dtm_litigious_words_1A.columns = count_vec_litigious.vocabulary_.keys()
    df_dtm_litigious_words_7 = pd.DataFrame(dtm_litigious_words_7.toarray(), index=data.index)
    df_dtm_litigious_words_7.columns = count_vec_litigious.vocabulary_.keys()


    sentiment_1A = pd.DataFrame(index=data.index)
    sentiment_1A['pos_count'] = df_dtm_pos_words_1A.sum(axis=1)
    sentiment_1A['phi_pos'] = sentiment_1A['pos_count'] / data['num_cleaned_words_Section_1A']
    sentiment_1A['neg_count'] = df_dtm_neg_words_1A.sum(axis=1)
    sentiment_1A['phi_neg'] = sentiment_1A['neg_count'] / data['num_cleaned_words_Section_1A']
    sentiment_1A['cons_count'] = df_dtm_cons_words_1A.sum(axis=1)
    sentiment_1A['phi_cons'] = sentiment_1A['cons_count'] / data['num_cleaned_words_Section_1A']
    sentiment_1A['neg_count'] = df_dtm_neg_words_1A.sum(axis=1)
    sentiment_1A['phi_neg'] = sentiment_1A['neg_count'] / data['num_cleaned_words_Section_1A']
    sentiment_1A['cons_count'] = df_dtm_cons_words_1A.sum(axis=1)
    sentiment_1A['phi_cons'] = sentiment_1A['cons_count'] / data['num_cleaned_words_Section_1A']
    sentiment_1A['uncer_count'] = df_dtm_uncer_words_1A.sum(axis=1)
    sentiment_1A['phi_uncer'] = sentiment_1A['uncer_count'] / data['num_cleaned_words_Section_1A']
    sentiment_1A['weakmodal_count'] = df_dtm_weakmodal_words_1A.sum(axis=1)
    sentiment_1A['phi_weakmodal'] = sentiment_1A['weakmodal_count'] / data['num_cleaned_words_Section_1A']
    sentiment_1A['strongmodal_count'] = df_dtm_strongmodal_words_1A.sum(axis=1)
    sentiment_1A['phi_strongmodal'] = sentiment_1A['strongmodal_count'] / data['num_cleaned_words_Section_1A']    
    sentiment_1A['litigious_count'] = df_dtm_litigious_words_1A.sum(axis=1)
    sentiment_1A['phi_litigious'] = sentiment_1A['litigious_count'] / data['num_cleaned_words_Section_1A']


    sentiment_7 = pd.DataFrame(index=data.index)
    sentiment_7['pos_count'] = df_dtm_pos_words_7.sum(axis=1)
    sentiment_7['phi_pos'] = sentiment_7['pos_count'] / data['num_cleaned_words_Section_7']
    sentiment_7['neg_count'] = df_dtm_neg_words_7.sum(axis=1)
    sentiment_7['phi_neg'] = sentiment_7['neg_count'] / data['num_cleaned_words_Section_7']
    sentiment_7['cons_count'] = df_dtm_cons_words_7.sum(axis=1)
    sentiment_7['phi_cons'] = sentiment_7['cons_count'] / data['num_cleaned_words_Section_7']
    sentiment_7['neg_count'] = df_dtm_neg_words_7.sum(axis=1)
    sentiment_7['phi_neg'] = sentiment_7['neg_count'] / data['num_cleaned_words_Section_7']
    sentiment_7['cons_count'] = df_dtm_cons_words_7.sum(axis=1)
    sentiment_7['phi_cons'] = sentiment_7['cons_count'] / data['num_cleaned_words_Section_7']
    sentiment_7['uncer_count'] = df_dtm_uncer_words_7.sum(axis=1)
    sentiment_7['phi_uncer'] = sentiment_7['uncer_count'] / data['num_cleaned_words_Section_7']
    sentiment_7['weakmodal_count'] = df_dtm_weakmodal_words_7.sum(axis=1)
    sentiment_7['phi_weakmodal'] = sentiment_7['weakmodal_count'] / data['num_cleaned_words_Section_7']
    sentiment_7['strongmodal_count'] = df_dtm_strongmodal_words_7.sum(axis=1)
    sentiment_7['phi_strongmodal'] = sentiment_7['strongmodal_count'] / data['num_cleaned_words_Section_7']    
    sentiment_7['litigious_count'] = df_dtm_litigious_words_7.sum(axis=1)
    sentiment_7['phi_litigious'] = sentiment_7['litigious_count'] / data['num_cleaned_words_Section_7']

    return sentiment_1A, sentiment_7    

st.subheader('Sentiment Ananlysis on 10-K report: ')

ten_load_state = st.text('Loading 10-K data ...')

sentiment_1A, sentiment_7 = get_sentiment_analysis(selected_stock)

ten_load_state.text('10-K data loaded...')

st.subheader('Sentiment Ananlysis on Section 1A of 10-K report: ')
st.write(sentiment_1A)

sentiment_1A_plot = sentiment_1A.drop(['phi_pos', 'phi_neg', 'phi_cons', 'phi_uncer', 'phi_weakmodal', 'phi_strongmodal', 'phi_litigious'], axis=1)

fig = go.Figure(data=[
    go.Bar(name='Positive', x=sentiment_1A_plot.index, y=sentiment_1A_plot['pos_count']),
    go.Bar(name='Negative', x=sentiment_1A_plot.index, y=sentiment_1A_plot['neg_count']),
    go.Bar(name='Constraining', x=sentiment_1A_plot.index, y=sentiment_1A_plot['cons_count']),
    go.Bar(name='Uncertainity', x=sentiment_1A_plot.index, y=sentiment_1A_plot['uncer_count']),
    go.Bar(name='WeakModal', x=sentiment_1A_plot.index, y=sentiment_1A_plot['weakmodal_count']),
    go.Bar(name='StrongModal', x=sentiment_1A_plot.index, y=sentiment_1A_plot['strongmodal_count']),
    go.Bar(name='Litigious', x=sentiment_1A_plot.index, y=sentiment_1A_plot['litigious_count']),
])
fig.update_layout(barmode='group', title = 'Sentiment Ananlysis on Section 1A of 10-K report')
st.plotly_chart(fig)

st.subheader('Sentiment Ananlysis on Section 7 of 10-K report: ')
st.write(sentiment_7)

sentiment_1A_plot = sentiment_7.drop(['phi_pos', 'phi_neg', 'phi_cons', 'phi_uncer', 'phi_weakmodal', 'phi_strongmodal', 'phi_litigious'], axis=1)

fig = go.Figure(data=[
    go.Bar(name='Positive', x=sentiment_1A_plot.index, y=sentiment_1A_plot['pos_count']),
    go.Bar(name='Negative', x=sentiment_1A_plot.index, y=sentiment_1A_plot['neg_count']),
    go.Bar(name='Constraining', x=sentiment_1A_plot.index, y=sentiment_1A_plot['cons_count']),
    go.Bar(name='Uncertainity', x=sentiment_1A_plot.index, y=sentiment_1A_plot['uncer_count']),
    go.Bar(name='WeakModal', x=sentiment_1A_plot.index, y=sentiment_1A_plot['weakmodal_count']),
    go.Bar(name='StrongModal', x=sentiment_1A_plot.index, y=sentiment_1A_plot['strongmodal_count']),
    go.Bar(name='Litigious', x=sentiment_1A_plot.index, y=sentiment_1A_plot['litigious_count']),
])
fig.update_layout(barmode='group', title = 'Sentiment Ananlysis on Section 7 of 10-K report')
st.plotly_chart(fig)



def get_finBERT(text):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    label_list=['positive','negative','neutral']
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return label_list[torch.argmax(outputs[0])]    

def get_news_sentiment(ticker):
    news_tables = {}
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    url = finwiz_url + ticker
    req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
    response = urlopen(req)    
    html = BeautifulSoup(response)
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table
    parsed_news = []

    for file_name, news_table in news_tables.items():

        for x in news_table.findAll('tr'):
            text = x.a.get_text() 
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]

            else:
                date = date_scrape[0]
                time = date_scrape[1]
            ticker = file_name.split('_')[0]

            parsed_news.append([ticker, date, time, text])
    
    columns = ['ticker', 'date', 'time', 'headline']
    parsed_news_sentiment = pd.DataFrame(parsed_news, columns=columns)
    
    #parsed_news_sentiment['Sentiment'] = parsed_news_sentiment['headline'].apply(lambda x: get_finBERT(x))
    parsed_news_sentiment = pd.read_csv('news_data.csv')
    
    return parsed_news_sentiment

st.subheader('Sentiment Ananlysis on news headlines from the past 4 days: ')


news = get_news_sentiment(selected_stock)
st.write(news.head())
tweet_plot = news['Sentiment'].value_counts()

fig = go.Figure(data=[
    go.Bar(name='Positive', x=['Positive'], y=[tweet_plot['positive']]),
    go.Bar(name='Negative', x=['Negative'], y=[tweet_plot['negative']]),
    go.Bar(name='Neutral', x=['Neutral'], y=[tweet_plot['neutral']]),
    
])
fig.update_layout(title = 'Sentiment Ananlysis on news headlines from the past 4 days:')
st.plotly_chart(fig)



TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stop_words = stopwords.words('english')

auth = tw.OAuthHandler('ausqqN1yIbBl2ZKK6bhEswwUZ', 'RXQ5CVcMa6uBp1onbiueXkG0PmoabXwsu7EOT6b3oQ9Lsjo57n')
auth.set_access_token('1278775916198391808-EOEoZEoS41ht2ptIETP2jtQNESkrQ1', 'fWSzNNZAbTFXKlFbBTt1tdThX2riZlN02Qh4cQfdhWfQv')
api = tw.API(auth, wait_on_rate_limit=True)
date_since = "2020-12-12"
n_tweets = 100   
def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    return text
def get_tweet_data(ticker):
    search_words = "#" + ticker + " -filter:retweets"
    tweets = tw.Cursor(api.search, 
                               q=search_words,
                               lang="en",
                               since=date_since).items(n_tweets)

    users_locs = [[tweet.user.screen_name,tweet.text] for tweet in tweets]
    tweet_text = pd.DataFrame(data=users_locs, columns=['user', 'tweet'])
    tweet_text['tweet_cleaned'] = tweet_text['tweet'].apply(lambda x : preprocess(x))
    return tweet_text

def get_twitter_sentiment(ticker):
  
    tweet_data = get_tweet_data(ticker)
    #tweet_data['Sentiment'] = tweet_data['tweet_cleaned'].apply(lambda x: get_finBERT(x))
    twitter_sentiment = pd.read_csv('tweet_data.csv')
    twitter_sentiment.drop(['Unnamed: 0'], axis=1, inplace=True)
    return twitter_sentiment

st.subheader('Sentiment Ananlysis on the last 100 tweets: ')

ten_load_state = st.text('Loading Twitter data ...')

twitter_sentiment = get_twitter_sentiment(selected_stock)

ten_load_state.text('Twitter data loaded...')
st.write(twitter_sentiment.head())

tweet_plot = twitter_sentiment['Sentiment'].value_counts()

fig = go.Figure(data=[
    go.Bar(name='Positive', x=['Positive'], y=[tweet_plot['positive']]),
    go.Bar(name='Negative', x=['Negative'], y=[tweet_plot['negative']]),
    go.Bar(name='Neutral', x=['Neutral'], y=[tweet_plot['neutral']]),
    
])
fig.update_layout(title = 'Sentiment Ananlysis on the last 100 tweets:')
st.plotly_chart(fig)