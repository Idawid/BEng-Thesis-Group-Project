import pandas as pd
import datetime
import os

from sentimentAnalysis.sentiment.FinbertSentiment import FinbertSentiment
from sentimentAnalysis.newsapi.news_client import NewsClient


def create_dir(directory_name):
    cwd = os.getcwd()
    path_to_new_dir = os.path.join(cwd, directory_name)
    if not os.path.exists(path_to_new_dir):
        os.mkdir(path_to_new_dir)


def cut_off_on_white_char(text, n):
    if len(text) <= n:
        return text

    last_white_space = text.rfind(' ', 0, n + 1)
    if last_white_space != -1:
        cut_string = text[:last_white_space].rstrip()
    else:
        cut_string = text[:n].rstrip()
    return cut_string


def transform_news_to_df(news_list) -> pd.DataFrame:
    if not isinstance(news_list, list):
        raise RuntimeError("This functions works only to transform list to dataframe")
    print("Transforming fetched news to fit input for sentiment analysis")
    result_df = pd.DataFrame.from_records(news_list)
    result_df["text"] = result_df["headline"] + ".\t" + result_df["summary"]
    result_df["text"] = result_df["text"].apply(lambda x: cut_off_on_white_char(x, 512))
    result_df["date"] = result_df["datetime"].map(lambda x: datetime.date.fromtimestamp(x))
    result_df = result_df.drop(columns=["category", "id", "related", "source", "url", "image",
                                        "headline", "summary", "datetime"])
    return result_df


class NewsSentimentAnalyzer:
    def __init__(self):
        create_dir("generated")
        self.news_client = NewsClient()
        self.sentiment_analysis_algorithm = FinbertSentiment()

    def calculate_sentiment_df(self, ticker, _from, to) -> pd.DataFrame:
        # Fetch news for given ticker and date range
        news_list = self.news_client.get_news(ticker=ticker, _from=_from, to=to)
        # Transform to dataframe to fit as input of sentiment analysis model
        news_df = transform_news_to_df(news_list)
        news_df.to_csv("generated/news_df.csv")
        print(len(news_df))
        # Calculate sentiment
        news_with_sentiment_df = self.sentiment_analysis_algorithm.calc_sentiment_score(news_df)
        # Delete neutral news - to be examined if this should be used
        news_with_sentiment_df = news_with_sentiment_df[news_with_sentiment_df['sentiment_score'] != 0.0]
        # Delete useless columns
        datetime_with_sentiment = news_with_sentiment_df.drop(columns=['text', 'sentiment'])
        # Group by date and calculate sentiment for the day as mean of the sentiments in that day
        # TODO: explore other methods than mean
        grouped_by_date = datetime_with_sentiment.groupby('date', as_index=False).mean()
        return grouped_by_date


test_ticker = "AAPL"
test_from = "2023-11-29"
test_to = "2023-12-06"


if __name__ == "__main__":
    news_sentiment_analyzer = NewsSentimentAnalyzer()
    df = news_sentiment_analyzer.calculate_sentiment_df(test_ticker, test_from, test_to)
    df.to_csv("generated/final_df.csv")
