import json
import time

from DataProcessorUtils import get_sentiment_data
from RabbitMQClient import RabbitMQClient


class DataProcessor:
    def __init__(self):
        self.rabbitmq_client = RabbitMQClient()

    def start_processing(self):
        self.rabbitmq_client.channel.exchange_declare(exchange='stock_updates', exchange_type='topic')
        queue_name = 'data_processor'
        self.rabbitmq_client.channel.queue_declare(queue=queue_name)
        self.rabbitmq_client.channel.queue_bind(exchange='stock_updates', queue=queue_name, routing_key='request.*.*')

        def callback(ch, method, properties, body):
            data = json.loads(body)
            # Process data logic here
            print("Processing data for", data['ticker'], "...")
            processed_data = get_sentiment_data(ticker=data['ticker'])
            print("Done!")
            # Publishing to a specific stock and time frame queue
            target_queue = f"{data['ticker']}_{data['time_frame']}_data"
            self.rabbitmq_client.channel.queue_declare(queue=target_queue)
            self.rabbitmq_client.channel.basic_publish(exchange='', routing_key=target_queue, body=json.dumps(processed_data))

        self.rabbitmq_client.channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
        print("Data Processor started. Waiting for requests.")
        self.rabbitmq_client.channel.start_consuming()


if __name__ == '__main__':
    processor = DataProcessor()
    processor.start_processing()
