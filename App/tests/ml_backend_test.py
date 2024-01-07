import unittest
from common.data_request import DataRequest, MessageType
from ml_backend.ml_backend_utils import process_request


class TestMLBackend(unittest.TestCase):
    def test_current_price_request(self):
        dummy_request = DataRequest(
            ticker='AAPL',
            datetime_from='2023-12-18T04:13:03.072Z',
            datetime_to='2023-12-31T04:13:03.072Z',
            message_type=MessageType.CURRENT_PRICE,
            model_settings=None
        )
        result = process_request(dummy_request)
        print(result)

    def test_sentiment_request(self):
        dummy_request = DataRequest(
            ticker='AAPL',
            datetime_from='2023-12-11T04:13:03.072Z',
            datetime_to='2023-12-15T04:13:03.072Z',
            message_type=MessageType.SENTIMENT,
            model_settings=None
        )
        result = process_request(dummy_request)
        print(result)

    def test_price_history_request(self):
        dummy_request = DataRequest(
            ticker='AAPL',
            datetime_from='2023-12-11T04:13:03.072Z',
            datetime_to='2023-12-18T04:13:03.072Z',
            message_type=MessageType.PRICE_HISTORY,
            model_settings=None
        )
        result = process_request(dummy_request)
        print(result)

    def test_article_list_request(self):
        dummy_request = DataRequest(
            ticker='AAPL',
            datetime_from='2023-12-11T04:13:03.072Z',
            datetime_to='2023-12-18T04:13:03.072Z',
            message_type=MessageType.ARTICLE_LIST,
            model_settings=None
        )
        result = process_request(dummy_request)
        print(result)

    def test_price_forecast_request(self):
        dummy_request = DataRequest(
            ticker='AAPL',
            datetime_from='2023-12-18T15:50:11.640Z',
            datetime_to='2023-12-25T15:50:11.640Z',
            message_type=MessageType.PRICE_FORECAST,
            model_settings=None
        )
        result = process_request(dummy_request)
        print(result)

if __name__ == '__main__':
    unittest.main()
