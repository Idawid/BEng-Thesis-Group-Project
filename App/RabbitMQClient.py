import pika
import json


class RabbitMQClient:
    def __init__(self, host='localhost', port=5672):
        # : - ) btw queue also sends heartbeat msgs that will result in disconnect if missed.
        # that means long tasks (> 3 min) cannot be performed without another wholesome hack.
        # heartbeat messages disabled for now with "heartbeat=0". but I'd honestly kms rather than recode it
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, heartbeat=0))
        self.channel = self.connection.channel()

    def publish_message(self, exchange, routing_key, message):
        self.channel.exchange_declare(exchange=exchange, exchange_type='topic')
        self.channel.basic_publish(exchange=exchange, routing_key=routing_key, body=json.dumps(message))

    def close_connection(self):
        self.connection.close()
