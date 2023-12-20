import finnhub
from sentiment_analysis.newsapi import config
from datetime import datetime


date_format = "%Y-%m-%d"


class NewsClient:

    def __init__(self):
        self.fh_client = finnhub.Client(config.API_KEY)

    def get_page_of_news(self, ticker, _from, to) -> list:
        if datetime.strptime(_from, date_format) > datetime.strptime(to, date_format):
            return list()
        print("Getting news from: " + _from + " to: " + to)
        result = self.fh_client.company_news(ticker, _from=_from, to=to)
        print("Got: " + str(len(result)))
        return result

    def get_news(self, ticker, _from, to) -> list:
        # Note that free version of this api allows for querying up to one year to the past and 100 calls per minute
        print("Fetching news for: ", ticker)

        news_list = self.get_page_of_news(ticker, _from, to)
        news_page = news_list
        local_to = datetime.strptime(to, date_format)
        local_from = datetime.strptime(_from, date_format)
        while local_from < local_to:
            fetched_to_date = datetime.fromtimestamp(news_page[-1]["datetime"])
            local_to = datetime(year=fetched_to_date.year, month=fetched_to_date.month, day=fetched_to_date.day)
            print("Got data until: " + str(fetched_to_date))
            news_page = self.get_page_of_news(ticker, _from, local_to.strftime(date_format))
            for item in news_page:
                if item not in news_list:
                    news_list.append(item)
        return news_list
