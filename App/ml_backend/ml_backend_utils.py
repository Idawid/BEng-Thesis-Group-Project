import json
import io
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from common.data_request import DataRequest, MessageType
from time_series.time_series_model import TimeSeriesModel
import yfinance as yf

from sentiment_analysis.sentiment_analysis import NewsSentimentAnalyzer, transform_news_to_df


def process_request(data_request: DataRequest):
    function_mapping = {
        MessageType.PRICE_HISTORY: process_price_history,
        MessageType.PRICE_FORECAST: process_price_forecast,
        MessageType.SENTIMENT: process_sentiment,
        MessageType.ARTICLE_LIST: process_article_list,
        MessageType.CURRENT_PRICE: process_current_price,
    }

    message_type = data_request.message_type
    if message_type in function_mapping:
        handling_function = function_mapping[message_type]
        processed_data = handling_function(data_request)
        return processed_data
    else:
        print("Unsupported message type:", message_type)
        return None


def process_price_history(data_request: DataRequest):
    """
    :param data_request: The request containing ticker, date range, and message type.
    :return:
        str: A JSON string representing the historical price data. The JSON format includes the following columns:
            - 'date': Date in the format 'YYYY-MM-DD'
            - 'close_price': Closing price for the date
    """
    print("Processing price history for:", data_request.ticker)
    date_from = datetime.fromisoformat(data_request.datetime_from.rstrip("Z"))
    date_to = datetime.fromisoformat(data_request.datetime_to.rstrip("Z"))

    stock = yf.Ticker(data_request.ticker)
    df = stock.history(start=date_from, end=date_to)

    df = df.reset_index()
    df = df[["Date", "Close"]]

    df["Date"] = df["Date"].dt.strftime('%Y-%m-%d')

    df = df.rename(columns={"Date": "date", "Close": "close_price"})
    json_records = df.to_json(orient='records')

    return json_records


def process_price_forecast(data_request: DataRequest):
    """
    :param data_request: The request containing ticker, date range, and message type.
    :return:
        str: A JSON string representing the historical price data. The JSON format includes the following columns:
            - 'date': Date in the format 'YYYY-MM-DD'
            - 'close_price': Closing price for the date
    """
    model_settings = data_request.model_settings
    N_EPOCHS = int(model_settings.number_of_epochs)
    N_NEURONS = int(model_settings.number_of_neurons)
    N_LAYERS = int(model_settings.number_of_layers) 
    N_STACKS = 8
    HORIZON = 1
    WINDOW_SIZE = 7  # 8 with sentiment 7 without
    if model_settings.nlp_enable_flag:
        WINDOW_SIZE = WINDOW_SIZE + 1

    INPUT_SIZE = WINDOW_SIZE * HORIZON
    THETA_SIZE = INPUT_SIZE + HORIZON

    # Extract and parse the dates
    date_from_iso = datetime.fromisoformat(data_request.datetime_from.rstrip("Z"))
    date_to_iso = datetime.fromisoformat(data_request.datetime_to.rstrip("Z"))

    # Calculate the day difference
    day_difference = (date_to_iso - date_from_iso).days

    # Extract data_request information
    ticker = data_request.ticker
    start_date = date_from_iso - timedelta(days=30)  # from data 30 days before
    end_date = datetime.now()
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    lenOfFuturedf = np.busday_count(start_date_str, end_date_str)
    dayDifference = (lenOfFuturedf - 1) % 7
    new_start_date_str = subtract_working_days_from_date(start_date_str, int(dayDifference))

    time_series_model = TimeSeriesModel()

    historical_data = yf.download(ticker, start=new_start_date_str, end=end_date_str)

    if model_settings.nlp_enable_flag:
        news_sentiment_analyzer = NewsSentimentAnalyzer()
        df_sentiment = news_sentiment_analyzer.calculate_sentiment_df(ticker, start_date_str, end_date_str)
        df_sentiment.to_csv("generated/final_df.csv")
        df_sentiment = pd.read_csv("generated/final_df.csv",
                                   parse_dates=["date"],
                                   index_col=["date"])
        df_sentiment = df_sentiment[["sentiment_score"]]

        prepared_data = time_series_model.makeWindowedDataWithSentiment(historical_data, df_sentiment,
                                                                        WINDOW_SIZE, HORIZON)
        time_series_model.moveSentiment(df_sentiment, abs(int(model_settings.sentiment_shift_days)))

    else:
        prepared_data = time_series_model.makeWindowedDataWithOutSentiment(historical_data, WINDOW_SIZE, HORIZON)

    X_train, y_train, X_test, y_test = time_series_model.testTrainingSplit(prepared_data)
    train_dataset, test_dataset = time_series_model.prepareDataForTraining(X_train, y_train, X_test, y_test, 1024)
    dataset_all, X_all, y_all = time_series_model.prepareDataForPrediction(prepared_data, 1024, WINDOW_SIZE)

    # Load model
    # model = tf.keras.models.load_model('saved_model/my_model')

    # Train model
    model = time_series_model.trainModel(
        inputSize=INPUT_SIZE,
        thetaSize=THETA_SIZE,
        horizon=HORIZON,
        nNeurons=N_NEURONS,
        nLayers=N_LAYERS,
        nStacks=N_STACKS,
        train_dataset=train_dataset,
        nEpocks=N_EPOCHS,
        test_dataset=test_dataset)
    model.save(f"saved_model/model_{ticker}")

    if model_settings.nlp_enable_flag:
        future_forecast = time_series_model.make_future_forecast_with_sentiment(
            values=y_all,
            model=model,
            into_future=day_difference,
            window_size=WINDOW_SIZE
        )
    else:
        future_forecast = time_series_model.make_future_forecast_without_sentiment(
            values=y_all,
            model=model,
            into_future=day_difference,
            window_size=WINDOW_SIZE
        )

    # Get dates corresponding to the forecast
    forecast_dates = pd.date_range(start=date_to_iso, periods=len(future_forecast))
    forecast_df = pd.DataFrame({'date': forecast_dates.strftime('%Y-%m-%d'), 'close_price': future_forecast})

    json_data = forecast_df.to_json(orient='records')

    return json_data


def process_sentiment(data_request: DataRequest):
    """
    :param data_request: The request containing ticker, date range, and message type.
    :return:
        str: A JSON string representing the historical price data. The JSON format includes the following columns:
            - 'date': Date in the format 'YYYY-MM-DD'
            - 'sentiment_score': Sentiment score for the date
    """
    print("Processing sentiment for:", data_request.ticker)
    temp_dt = datetime.fromisoformat(data_request.datetime_from.rstrip("Z"))
    date_from = temp_dt.strftime("%Y-%m-%d")

    temp_dt = datetime.fromisoformat(data_request.datetime_to.rstrip("Z"))
    date_to = temp_dt.strftime("%Y-%m-%d")

    news_sentiment_analyzer = NewsSentimentAnalyzer()
    df = news_sentiment_analyzer.calculate_sentiment_df(data_request.ticker, date_from, date_to)

    return df.to_json(orient='records')


def process_article_list(data_request: DataRequest):
    """
    Process news list data for a given data_request.
    :param data_request: The request containing ticker, date range, and message type.
    :return:
        str: A JSON string representing the news list data. The JSON format includes the following columns:
            - 'date': The date of the news article in the format 'YYYY-MM-DD'
            - 'headline': The headline of the news article
            - 'summary': A summary of the news article
            - 'source': The source of the news article
    """
    print("Processing article list for:", data_request.ticker)
    temp_dt = datetime.fromisoformat(data_request.datetime_from.rstrip("Z"))
    date_from = temp_dt.strftime("%Y-%m-%d")

    temp_dt = datetime.fromisoformat(data_request.datetime_to.rstrip("Z"))
    date_to = temp_dt.strftime("%Y-%m-%d")

    news_sentiment_analyzer = NewsSentimentAnalyzer()
    with SuppressPrints():
        news_list = news_sentiment_analyzer.news_client.get_news(
            ticker=data_request.ticker,
            _from=date_from,
            to=date_to
        )

    news_df = transform_news_to_df(news_list)
    news_with_sentiment_df = news_sentiment_analyzer.sentiment_analysis_algorithm.calc_sentiment_score(news_df)
    news_with_sentiment_df = news_with_sentiment_df[news_with_sentiment_df['sentiment_score'] != 0.0]

    news_list_df = pd.DataFrame(news_list)
    news_list_df['date'] = pd.to_datetime(news_list_df['datetime'], unit='s').dt.date

    merged_df = pd.merge(news_list_df, news_with_sentiment_df[['date', 'sentiment_score']], on='date', how='inner')

    selected_fields = merged_df[['date', 'headline', 'summary', 'source', 'sentiment_score']]
    selected_fields['date'] = selected_fields['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

    records = selected_fields.to_dict(orient='records')
    records = records[:20]

    json_data = json.dumps(records)
    return json_data


def process_current_price(data_request: DataRequest):
    """
    Get the live price and last update time for a given stock ticker symbol.
    :param data_request:
    :return:
        str: A JSON string representing the live price data. The JSON format includes the following columns:
            - 'live_price': The current live price as a floating-point number
            - 'last_update_time': The date and time of the last update in the format 'YYYY-MM-DD HH:MM:SS UTC'
    """

    print("Processing current price for:", data_request.ticker)
    try:
        stock = yf.Ticker(data_request.ticker)
        data = stock.history(period='1d')
        live_price = data['Close'][-1]
        last_update_time = data.index[-1].tz_convert("UTC")

        return [
            {
                "live_price": float(live_price),
                "last_update_time": last_update_time.strftime('%Y-%m-%d %H:%M:%S %Z')
            }
        ]

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()  # Redirect stdout to a string buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout  # Restore the original stdout
        return False  # Don't suppress exceptions


def subtract_working_days_from_date(input_date, working_days_to_subtract):
    # Convert the string date to a datetime object
    date_object = datetime.strptime(input_date, "%Y-%m-%d")

    # Define a function to check if a given date is a weekend (Saturday or Sunday)
    def is_weekend(date):
        return date.weekday() in [5, 6]  # Saturday or Sunday

    # Subtract the specified number of working days
    while working_days_to_subtract > 0:
        date_object -= timedelta(days=1)
        # Check if the current date is not a weekend
        if not is_weekend(date_object):
            working_days_to_subtract -= 1

    # Convert the result back to a string format
    result_date = date_object.strftime("%Y-%m-%d")

    return result_date


# def is_room_opened(name):
#     print(appSocketIOContext.server.manager.rooms.items())
#     for room_path, room_data in appSocketIOContext.server.manager.rooms.items():
#         current_rooms = list(room_data.keys())
#         if name in current_rooms:
#             return True
#
#     return False
