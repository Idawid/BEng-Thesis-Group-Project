import os
import pandas as pd
import datetime

main_cache_path = os.path.join(os.path.dirname(__file__), "cache")


def create_cache_folder():
    if not os.path.exists(main_cache_path):
        os.mkdir(main_cache_path)


def merge_sentiment_dataframes(new_dataframe: pd.DataFrame, old_dataframe: pd.DataFrame):
    if new_dataframe is None and old_dataframe is None:
        raise "Can not merge two None dataframes"
    if old_dataframe is None:
        return new_dataframe.apply(
            lambda x: x.strftime("%Y-%m-%d") if isinstance(x, datetime.date) else x
        )
    if new_dataframe is None:
        return old_dataframe.apply(
            lambda x: x.strftime("%Y-%m-%d") if isinstance(x, datetime.date) else x
        )
    old_dataframe['date'] = old_dataframe['date'].apply(
        lambda x: x.strftime("%Y-%m-%d") if isinstance(x, datetime.date) else x
    )

    new_dataframe['date'] = new_dataframe['date'].apply(
        lambda x: x.strftime("%Y-%m-%d") if isinstance(x, datetime.date) else x
    )
    merged_df = pd.merge(new_dataframe, old_dataframe, on='date', how='outer',
                         suffixes=('_new', '_old'))
    merged_df['sentiment_score'] = (merged_df['sentiment_score_new']
                                    .combine_first(merged_df['sentiment_score_old']))
    return merged_df[['date', 'sentiment_score']].sort_values(by=['date'])


class SentimentCacher:
    def __init__(self):
        self.main_cache_path = main_cache_path
        self.ticker = None
        self.ticker_df = None
        self.path_to_ticker_cache = None

    def set_ticker(self, ticker):
        self.ticker = ticker
        self.path_to_ticker_cache = os.path.join(self.main_cache_path, ticker)

    def read_cache_to_df(self):
        if os.path.exists(self.path_to_ticker_cache):
            self.ticker_df = pd.read_csv(self.path_to_ticker_cache)
        else:
            self.ticker_df = None

    def is_date_cached(self, content_date: datetime.datetime) -> bool:
        if self.ticker_df is None:
            return False
        string_date = content_date.strftime("%Y-%m-%d")
        return string_date in set(self.ticker_df['date'])

    def get_cached_value_for_date(self, content_date):
        if not self.is_date_cached(content_date):
            return None
        return self.ticker_df.loc[self.ticker['date'] == content_date, 'sentiment_score'].values[0]

    def get_cached(self):
        self.read_cache_to_df()
        return self.ticker_df

    def cache_sentiment_dataframe(self, sentiment_dataframe: pd.DataFrame):
        if self.ticker is None:
            raise "Ticker needs to be set before caching - .set_ticker(<ticker>)"
        if not os.path.exists(self.path_to_ticker_cache):
            sentiment_dataframe.to_csv(self.path_to_ticker_cache, index=False)
        else:
            # Merge df with cached
            merged_df = merge_sentiment_dataframes(sentiment_dataframe, self.ticker_df)
            merged_df.to_csv(self.path_to_ticker_cache, index=False)