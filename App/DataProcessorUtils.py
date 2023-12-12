import random
from datetime import datetime, timedelta
from pprint import pprint

import yfinance as yf

from sentimentAnalysis.sentiment_analysis import NewsSentimentAnalyzer


def randomize_data():
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'headline': f'Random Headline {random.randint(1, 100)}',
        'description': f'Random Description {random.randint(1, 100)}',
        'sentiment': round(random.uniform(-1, 1), 2)
    }


def update_data(data):
    new_row = randomize_data()
    data.append(new_row)
    if len(data) > 500:
        data.pop(0)
    return data


def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d')
    live_price = data['Close'][-1]
    last_quote = data['Close'].iloc[-1]
    last_update_time = data.index[-1].tz_convert("UTC")
    return live_price, last_update_time


def get_sentiment_data(ticker):
    print("Updating sentiment for:", ticker)
    current_date = datetime.now().date()
    seven_days_ago = current_date - timedelta(days=7)
    test_from = seven_days_ago.strftime("%Y-%m-%d")
    test_to = current_date.strftime("%Y-%m-%d")

    news_sentiment_analyzer = NewsSentimentAnalyzer()
    df = news_sentiment_analyzer.calculate_sentiment_df(ticker, test_from, test_to)

    return df.to_json(orient='split', index=False)
