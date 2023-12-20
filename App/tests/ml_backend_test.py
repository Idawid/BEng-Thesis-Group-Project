from common.data_request import DataRequest, MessageType
from ml_backend.ml_backend_utils import process_request

if __name__ == '__main__':
    # ####################################################################################################################
    # dummy_request = DataRequest(
    #     ticker='AAPL',  # Replace with your desired ticker symbol
    #     datetime_from='2023-12-18T04:13:03.072Z',  # Replace with your desired start date
    #     datetime_to='2023-12-31T04:13:03.072Z',  # Replace with your desired end date
    #     message_type=MessageType.CURRENT_PRICE  # Replace with the desired message type
    # )
    #
    # result = process_request(dummy_request)
    # print(result)
    # ####################################################################################################################
    # dummy_request2 = DataRequest(
    #     ticker='AAPL',  # Replace with your desired ticker symbol
    #     datetime_from='2023-12-11T04:13:03.072Z',  # Replace with your desired start date
    #     datetime_to='2023-12-15T04:13:03.072Z',  # Replace with your desired end date
    #     message_type=MessageType.SENTIMENT  # Replace with the desired message type
    # )
    #
    # result = process_request(dummy_request2)
    # print(result)
    # ####################################################################################################################
    # dummy_request3 = DataRequest(
    #     ticker='AAPL',  # Replace with your desired ticker symbol
    #     datetime_from='2023-12-11T04:13:03.072Z',  # Replace with your desired start date
    #     datetime_to='2023-12-18T04:13:03.072Z',  # Replace with your desired end date
    #     message_type=MessageType.PRICE_HISTORY  # Replace with the desired message type
    # )
    #
    # result = process_request(dummy_request3)
    # print(result)
    # ####################################################################################################################
    dummy_request4 = DataRequest(
        ticker='AAPL',  # Replace with your desired ticker symbol
        datetime_from='2023-12-11T04:13:03.072Z',  # Replace with your desired start date
        datetime_to='2023-12-18T04:13:03.072Z',  # Replace with your desired end date
        message_type=MessageType.ARTICLE_LIST  # Replace with the desired message type
    )

    result = process_request(dummy_request4)
    print(result)
    # ###################################################################################################################
    # dummy_request5 = DataRequest(
    #     ticker='AAPL',  # Replace with your desired ticker symbol
    #     datetime_from='2023-12-18T15:50:11.640Z',  # Replace with your desired start date
    #     datetime_to='2023-12-25T15:50:11.640Z',  # Replace with your desired end date
    #     message_type=MessageType.PRICE_FORECAST  # Replace with the desired message type
    # )
    #
    # result = process_request(dummy_request5)
    # print(result)
