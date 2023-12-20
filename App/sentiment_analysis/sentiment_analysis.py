from sentiment_analysis.newsapi.news_client import NewsClient
from sentiment_analysis.sentiment.FinbertSentiment import FinbertSentiment
from sentiment_analysis.sentiment_cacher import SentimentCacher, merge_sentiment_dataframes, create_cache_folder
from datetime import timedelta, datetime
import pandas as pd
import datetime as dt
import os


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
    result_df["date"] = result_df["datetime"].map(lambda x: dt.date.fromtimestamp(x))
    result_df = result_df.drop(columns=["category", "id", "related", "source", "url", "image",
                                        "headline", "summary", "datetime"])
    return result_df


class NewsSentimentAnalyzer:
    def __init__(self):
        create_dir("generated")
        create_cache_folder()
        self.news_client = NewsClient()
        self.sentiment_analysis_algorithm = FinbertSentiment()
        self.cacher = SentimentCacher()

    def fetch_new_sentiment(self, ticker, _from: datetime, to: datetime) -> pd.DataFrame:
        # Convert dates to string
        string_from = _from.strftime("%Y-%m-%d")
        string_to = to.strftime("%Y-%m-%d")
        news_list = self.news_client.get_news(ticker=ticker, _from=string_from, to=string_to)
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
        return datetime_with_sentiment.groupby('date', as_index=False).mean()

    def calculate_sentiment_df(self, ticker, _from, to) -> pd.DataFrame:
        # Get cached news
        self.cacher.set_ticker(ticker)
        merged_sentiment = self.cacher.get_cached()
        start_date = datetime.strptime(_from, "%Y-%m-%d")
        end_date = datetime.strptime(to, "%Y-%m-%d")
        last_not_cached_date = None
        # Iterate through dates
        content_date = start_date
        while content_date <= end_date:
            is_cached = self.cacher.is_date_cached(content_date)
            if not is_cached and last_not_cached_date is None:
                last_not_cached_date = content_date
            elif is_cached and last_not_cached_date is not None:
                print("Running fetch for dates: " + str(last_not_cached_date) + " " + str(content_date))
                new_sentiment_df = self.fetch_new_sentiment(ticker, last_not_cached_date, content_date)
                if new_sentiment_df is not None:
                    merged_sentiment = merge_sentiment_dataframes(new_sentiment_df, merged_sentiment)
                last_not_cached_date = None
            content_date += timedelta(days=1)
        if last_not_cached_date is not None:
            content_date -= timedelta(days=1)
            new_sentiment_df = self.fetch_new_sentiment(ticker, last_not_cached_date, content_date)
            if new_sentiment_df is not None:
                merged_sentiment = merge_sentiment_dataframes(new_sentiment_df, merged_sentiment)
        # Fetch news for given ticker and date range
        self.cacher.cache_sentiment_dataframe(merged_sentiment)

        final_df = merged_sentiment.loc[merged_sentiment['date'].apply(
            lambda x: start_date <= datetime.strptime(x, "%Y-%m-%d") <= end_date
        )]
        return final_df


test_ticker = "AAPL"
test_from = "2023-06-20"
test_to = "2023-07-14"

if __name__ == "__main__":
    news_sentiment_analyzer = NewsSentimentAnalyzer()
    df = news_sentiment_analyzer.calculate_sentiment_df(test_ticker, test_from, test_to)
    df.to_csv("generated/final_df.csv", index=False)