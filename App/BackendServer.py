import json
import threading

from RabbitMQClient import RabbitMQClient


class BackendServer:
    def __init__(self, app_socket_io_context):
        self.socketio = app_socket_io_context

    def subscribe(self, data):
        routing_key = f"request.{data['ticker']}.{data['time_frame']}"
        # Using a new RabbitMQClient instance for publishing
        publish_client = RabbitMQClient()
        publish_client.publish_message(exchange='stock_updates', routing_key=routing_key, message=data)
        publish_client.close_connection()

        # Subscribe to processed data
        self.subscribe_to_processed_data(data['ticker'], data['time_frame'])

    def subscribe_to_processed_data(self, stock, time_frame):
        # Start a new thread for each subscription
        thread = threading.Thread(target=self.listen_for_processed_data, args=(stock, time_frame))
        thread.start()

    def listen_for_processed_data(self, stock, time_frame):
        # Each thread has its own RabbitMQ client
        consumer_client = RabbitMQClient()
        queue_name = f"{stock}_{time_frame}_data"

        def callback(ch, method, properties, body):
            processed_data = json.loads(body)
            print(f"Received processed data: {processed_data}")
            self.socketio.emit('update_sentiment', {'df_data': processed_data},
                               namespace='/dashboard', to=stock)

        consumer_client.channel.queue_declare(queue=queue_name)
        consumer_client.channel.queue_bind(exchange='stock_updates', queue=queue_name, routing_key=f"data.{stock}.{time_frame}")
        consumer_client.channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

        consumer_client.channel.start_consuming()
