from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sentiment_analysis.sentiment.SentimentAnalysisBase import SentimentAnalysisBase


class FinbertSentiment (SentimentAnalysisBase):

    def __init__(self):
        finbert = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        #tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self._sentiment_analysis = pipeline(
            "text-classification", model=finbert, tokenizer=tokenizer)
        super().__init__()

    def calc_sentiment_score(self, df):
        print("Calculating sentiment using Finbert Sentiment Analysis model")
        df['sentiment'] = df['text'].apply(self._sentiment_analysis)
        df['sentiment_score'] = df['sentiment'].apply(
            lambda x: {x[0]['label'] == 'negative': -1, x[0]['label'] == 'positive': 1}.get(True, 0) * x[0]['score'])
        return df
