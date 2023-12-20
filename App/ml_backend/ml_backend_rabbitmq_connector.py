import json
import pika

from common.constants import Constants
from common.data_request import DataRequest, MessageType
from ml_backend.ml_backend_utils import process_request


class MLBackendRabbitMQConnector:
    def __init__(self):
        # One connection per process, one channel per thread
        host = 'localhost'
        port = 5672
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, heartbeat=0))

        self.channel = self.connection.channel()

        # Declare both queues. It's redundant if they exist
        self.channel.queue_declare(queue=Constants.QUEUE_NAME_REQUEST, exclusive=False)
        self.channel.queue_declare(queue=Constants.QUEUE_NAME_RESPOND, exclusive=False)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=Constants.QUEUE_NAME_REQUEST,
            on_message_callback=self.callback,
            auto_ack=False
        )

    def callback(self, ch, method, properties, body):
        try:
            print('update_request', 'MLBackend got', body)
            data = json.loads(body)
            data_request = DataRequest(
                ticker=data['ticker'],
                datetime_from=data['datetime_from'],
                datetime_to=data['datetime_to'],
                message_type=MessageType.from_string(data['message_type'])
            )

            processed_data = process_request(data_request)

            self.send_processed_data(processed_data, data_request)
            print("Request msg received, processed and acked.")

        except Exception as e:
            print(f"An error occurred: {e}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_processed_data(self, processed_data, data_request):
        """ Send the processed data back via RabbitMQ. """
        message_data = {
            "processed_data": processed_data,
            "original_data": json.dumps(data_request.to_json())
        }
        self.channel.basic_publish(
            exchange='',
            routing_key=Constants.QUEUE_NAME_RESPOND,
            body=json.dumps(message_data)
        )
        print("Data processor sends:", json.dumps(message_data))

    def start_processing(self):
        print("Data Processor started. Waiting for requests.")
        self.channel.start_consuming()
