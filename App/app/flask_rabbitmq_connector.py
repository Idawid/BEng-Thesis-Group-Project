import json
import threading
import pika

from common.constants import Constants
from common.data_request import DataRequest, MessageType


class BackendServer:
    def __init__(self, app_socket_io_context):
        self.socketio = app_socket_io_context
        self._init_queues()
        threading.Thread(target=self.listen_data_receive).start()

    def _create_connection(self):
        host = Constants.QUEUE_HOST
        port = Constants.QUEUE_PORT
        return pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, heartbeat=0))

    def _init_queues(self):
        """ Initialize the queues. It's redundant if they exist. """
        connection = self._create_connection()
        with connection.channel() as channel:
            channel.queue_declare(queue=Constants.QUEUE_NAME_REQUEST, exclusive=False)
            channel.queue_declare(queue=Constants.QUEUE_NAME_RESPOND, exclusive=False)
        connection.close()

    def send_data_request(self, data_request: DataRequest):
        connection = self._create_connection()
        try:
            serialized_request = data_request.to_json()
            with connection.channel() as channel:
                channel.basic_publish(
                    exchange='',
                    routing_key=Constants.QUEUE_NAME_REQUEST,
                    body=serialized_request
                )
            print('update_request', '2. sent to MLBackend', data_request)
        except Exception as e:
            print('Error:', str(e))
        finally:
            connection.close()

    def listen_data_receive(self):
        connection = self._create_connection()
        channel = connection.channel()

        def callback(ch, method, properties, body):
            print('update_request', '5. Flask got', body)
            deserialized_data = json.loads(body)
            processed_data = deserialized_data["processed_data"]
            serialized_request = deserialized_data["original_data"]

            data = json.loads(serialized_request)
            print(data)
            data_request = DataRequest.from_json(data)

            self.emit_event(data_request=data_request, processed_data=processed_data)

        channel.basic_consume(queue=Constants.QUEUE_NAME_RESPOND, on_message_callback=callback, auto_ack=True)
        try:
            channel.start_consuming()
        finally:
            channel.close()
            connection.close()

    def emit_event(self, data_request: DataRequest, processed_data):
        event_mapping = {
            MessageType.PRICE_HISTORY: 'update_price_history',
            MessageType.PRICE_FORECAST: 'update_price_forecast',
            MessageType.SENTIMENT: 'update_sentiment',
            MessageType.ARTICLE_LIST: 'update_article_list',
            MessageType.CURRENT_PRICE: 'update_current_price',
        }

        event_name = event_mapping.get(data_request.message_type)

        if event_name:
            self.socketio.emit(event_name, processed_data, namespace='/dashboard', to=data_request.ticker)
            print(f"Emitted event: {event_name}",
                  f"processed_data={processed_data}",
                  f"For Data Request: {data_request.to_json()}")
        else:
            print(f"Invalid event name: {event_name}")
