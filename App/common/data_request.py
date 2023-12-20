import json
import re
from enum import Enum


class MessageType(Enum):
    PRICE_HISTORY = 'price_history_chart'
    PRICE_FORECAST = 'price_forecast_chart'
    SENTIMENT = 'sentiment_chart'
    ARTICLE_LIST = 'article_list'
    CURRENT_PRICE = 'current_price'

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s):
        for member in MessageType:
            if member.value == s:
                return member
        raise ValueError(f"{s} is not a valid MessageType")


class DataRequest:
    def __init__(self, ticker: str, datetime_from: str, datetime_to: str, message_type: MessageType):
        self.ticker = ticker
        # if not self.is_valid_datetime(datetime_from) or not self.is_valid_datetime(datetime_to):
        #     print("Invalid datetime format", datetime_from, datetime_to)
            # raise ValueError("Invalid datetime format")
        self.datetime_from = datetime_from
        self.datetime_to = datetime_to
        self.message_type = message_type

    def to_json(self):
        """ Serialize the object to a JSON string """
        return json.dumps({
            'ticker': self.ticker,
            'datetime_from': self.datetime_from,
            'datetime_to': self.datetime_to,
            'message_type': str(self.message_type)
        })

    @classmethod
    def from_json(cls, json_str):
        """ Deserialize a JSON string to a RequestDataUpdateInfo object """
        data = json.loads(json_str)
        return cls(
            ticker=data['ticker'],
            datetime_from=data['datetime_from'],
            datetime_to=data['datetime_to'],
            message_type=MessageType.from_string(data['message_type'])
        )

