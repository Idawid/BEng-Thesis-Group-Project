import json
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


class ModelSettings:
    def __init__(self, nlp_enable_flag, sentiment_shift_days, number_of_neurons, number_of_layers, number_of_epochs):
        self.nlp_enable_flag = nlp_enable_flag
        self.sentiment_shift_days = sentiment_shift_days
        self.number_of_neurons = number_of_neurons
        self.number_of_layers = number_of_layers
        self.number_of_epochs = number_of_epochs

    def to_json(self):
        return {
            "nlp_enable_flag": self.nlp_enable_flag,
            "sentiment_shift_days": self.sentiment_shift_days,
            "number_of_neurons": self.number_of_neurons,
            "number_of_layers": self.number_of_layers,
            "number_of_epochs": self.number_of_epochs
        }

    @classmethod
    def from_json(cls, json_str):
        """ Deserialize a JSON string to a ModelSettings object """
        #data = json.loads(json_str)
        data = json_str
        return cls(
            nlp_enable_flag=data['nlp_enable_flag'],
            sentiment_shift_days=data['sentiment_shift_days'],
            number_of_neurons=data['number_of_neurons'],
            number_of_layers=data['number_of_layers'],
            number_of_epochs=data['number_of_epochs']
        )


class DataRequest:
    def __init__(self, ticker: str, datetime_from: str, datetime_to: str, message_type: MessageType,
                 model_settings: ModelSettings = None):
        self.ticker = ticker
        # if not self.is_valid_datetime(datetime_from) or not self.is_valid_datetime(datetime_to):
        #     print("Invalid datetime format", datetime_from, datetime_to)
        # raise ValueError("Invalid datetime format")
        self.datetime_from = datetime_from
        self.datetime_to = datetime_to
        self.message_type = message_type
        self.model_settings = model_settings

    def to_json(self):
        """ Serialize the object to a JSON string """
        return json.dumps({
            'ticker': self.ticker,
            'datetime_from': self.datetime_from,
            'datetime_to': self.datetime_to,
            'message_type': str(self.message_type),
            'model_settings': self.model_settings.to_json()
        })

    @classmethod
    def from_json(cls, json_str):
        """ Deserialize a JSON string to a DataRequest object """
        data = json.loads(json_str)
        model_settings = ModelSettings.from_json(data['model_settings'])
        return cls(
            ticker=data['ticker'],
            datetime_from=data['datetime_from'],
            datetime_to=data['datetime_to'],
            message_type=MessageType.from_string(data['message_type']),
            model_settings=model_settings
        )
