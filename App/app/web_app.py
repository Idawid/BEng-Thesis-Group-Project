import json
from flask import Flask, request, render_template, abort
from flask_socketio import SocketIO, emit, join_room

from app.flask_rabbitmq_connector import BackendServer
from common.constants import Constants
from common.data_request import MessageType, DataRequest, ModelSettings


class WebApplication:
    def __init__(self):
        self.app = Flask(__name__, static_folder='static', template_folder='templates')
        self.socket_io = SocketIO(self.app, async_mode="threading", logger=True, engineio_logger=True)
        self.backend_server = BackendServer(app_socket_io_context=self.socket_io)
        self.setup_routes()

    def setup_routes(self):
        self.app.route('/')(self.index)
        self.app.route('/dashboard')(self.dashboard)
        self.app.errorhandler(404)(self.page_not_found)
        self.app.errorhandler(500)(self.server_error)

    def run(self):
        self.socket_io.run(self.app, allow_unsafe_werkzeug=True, debug=True, use_reloader=False)

    def index(self):
        return render_template('index.html')

    def dashboard(self):
        asset_id = request.args.get('id').lower()
        try:
            with open(Constants.STOCK_DATA_PATH, 'r') as file:
                stock_data = json.load(file)
                if asset_id not in stock_data:
                    abort(404)
                stock = stock_data[asset_id]
        except (FileNotFoundError, json.JSONDecodeError):
            abort(500)

        return render_template(
            'dashboard.html',
            async_mode=self.socket_io.async_mode,
            stock=stock
        )

    def page_not_found(self, error):
        return render_template('404.html'), 404

    def server_error(self, error):
        return render_template('500.html'), 500


class WebApplicationSocketIO:
    def __init__(self, app, socket_io, backend_server):
        self.app = app
        self.socket_io = socket_io
        self.backend_server = backend_server
        self.register_socket_io_events()

    def register_socket_io_events(self):
        @self.socket_io.on('connect', namespace='/dashboard')
        def handle_dashboard_connect():
            pass

        @self.socket_io.on('join_ticker', namespace='/dashboard')
        def handle_dashboard_join_ticker(data):
            join_room(room=data['ticker'], namespace='/dashboard')
            print('join_ticker', f'Client id={request.sid}', f"room={data['ticker']}")

        @self.socket_io.on('update_request', namespace='/dashboard')
        def handle_dashboard_update_request(data):
            print('update_request', f'Client id={request.sid}', f"room={data['ticker']}", data)

            if data['message_type'] not in [member.value for member in MessageType]:
                print('update_request', 'Unknown MessageType:', data['message_type'], data)
                return

            model_settings = ModelSettings(
                data['model_settings']['nlp_enable_flag'],
                data['model_settings']['sentiment_shift_days'],
                data['model_settings']['number_of_neurons'],
                data['model_settings']['number_of_layers'],
                data['model_settings']['number_of_epochs']
            )

            data_request = DataRequest(
                ticker=data['ticker'],
                datetime_from=data['datetime_from'],
                datetime_to=data['datetime_to'],
                message_type=data['message_type'],
                model_settings=model_settings
            )

            self.backend_server.send_data_request(data_request=data_request)

            print('update_request', '1. socketio - OK', data)

        @self.socket_io.on('disconnect', namespace='/dashboard')
        def handle_dashboard_disconnect():
            print('Client id=', request.sid, 'disconnected. Left all rooms.')
