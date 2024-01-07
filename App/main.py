import threading
from app.web_app import WebApplication, WebApplicationSocketIO
from ml_backend.ml_backend_rabbitmq_connector import MLBackendRabbitMQConnector


def start_ml_backend_processor():
    processor = MLBackendRabbitMQConnector()
    processor.start_processing()


def start_web_application():
    web_application = WebApplication()
    socket_io = WebApplicationSocketIO(web_application.app, web_application.socket_io, web_application.backend_server)
    web_application.run()


if __name__ == '__main__':
    # Start the web application in its own thread
    web_app_thread = threading.Thread(target=start_web_application)
    web_app_thread.start()

    num_processors = 5
    for _ in range(num_processors):
        processor_thread = threading.Thread(target=start_ml_backend_processor)
        processor_thread.start()

    web_app_thread.join()
