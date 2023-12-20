import os
import sys

relative_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(relative_path)

import shutil
import unittest
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime
from sentiment_analysis.sentiment_analysis import NewsSentimentAnalyzer
from sentiment_analysis.sentiment_analysis import create_dir, cut_off_on_white_char, transform_news_to_df
from sentiment_analysis.newsapi.news_client import NewsClient


class TestNewsSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.news_sentiment_analyzer = NewsSentimentAnalyzer()

    def tearDown(self):
        shutil.rmtree("generated")

    def test_create_dir(self):
        directory_name = "test_dir"
        create_dir(directory_name)
        self.assertTrue(os.path.exists(directory_name))

    def test_cut_off_on_white_char(self):
        text = "This is a test sentence with more than 20 characters."
        n = 20
        result = cut_off_on_white_char(text, n)
        self.assertTrue(len(result) <= n)

    def test_transform_news_to_df(self):
        news_list = [
            {
                "headline": "Headline 1",
                "summary": "Summary 1",
                "datetime": datetime.now().timestamp(),
                "category": "Category 1",
                "id": 1,
                "related": "Related 1",
                "source": "Source 1",
                "url": "URL 1",
                "image": "Image 1"
            },
            {
                "headline": "Headline 2",
                "summary": "Summary 2",
                "datetime": datetime.now().timestamp(),
                "category": "Category 2",
                "id": 2,
                "related": "Related 2",
                "source": "Source 2",
                "url": "URL 2",
                "image": "Image 2"
            }
        ]
        result_df = transform_news_to_df(news_list)
        self.assertIsInstance(result_df, pd.DataFrame)

    def test_calculate_sentiment_df(self):
        test_ticker = "AAPL"
        test_from = "2023-12-04"
        test_to = "2023-12-04"
        result_df = self.news_sentiment_analyzer.calculate_sentiment_df(test_ticker, test_from, test_to)
        self.assertIsInstance(result_df, pd.DataFrame)


class TestNewsClient(unittest.TestCase):
    def setUp(self):
        self.news_client = NewsClient()

    def test_get_page_of_news(self):
        # Mocking the finnhub.Client to avoid actual API calls
        self.news_client.fh_client.company_news = MagicMock(return_value=[
            {"headline": "Test Headline", "datetime": 1672566851}
        ])

        ticker = "AAPL"
        _from = "2023-01-01"
        to = "2023-01-31"

        result = self.news_client.get_page_of_news(ticker, _from, to)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["headline"], "Test Headline")

    def test_get_news(self):
        # Mocking get_page_of_news to avoid actual API calls
        self.news_client.fh_client.company_news = MagicMock(return_value=[
            {"headline": "Test Headline 1", "datetime": 1672739651},
            {"headline": "Test Headline 2", "datetime": 1672653251},
            {"headline": "Test Headline 3", "datetime": 1672566851}
        ])

        ticker = "AAPL"
        _from = "2023-01-01"
        to = "2023-01-03"

        result = self.news_client.get_news(ticker, _from, to)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["headline"], "Test Headline 1")
        self.assertEqual(result[1]["headline"], "Test Headline 2")
        self.assertEqual(result[2]["headline"], "Test Headline 3")


if __name__ == "__main__":
    unittest.main()