import json

from BackendServer import BackendServer

from flask import Flask, request, render_template, jsonify, abort
from flask_socketio import SocketIO, emit, join_room, leave_room, Namespace, rooms
import DataProcessorUtils

article_data = []


appBaseContext = Flask(__name__)
appSocketIOContext = SocketIO(appBaseContext, async_mode="threading", logger=True, engineio_logger=True)
stockBackendServer = BackendServer(app_socket_io_context=appSocketIOContext)


@appBaseContext.route('/')
def index():
    return render_template('index.html')


@appBaseContext.route('/dashboard')
def dashboard():
    asset_id = request.args.get('id').lower()
    try:
        with open('static/DATA/stock_list.json', 'r') as file:
            stock_data = json.load(file)
            if asset_id not in stock_data:
                abort(404)
            stock = stock_data[asset_id]
    except (FileNotFoundError, json.JSONDecodeError):
        abort(500)  # Server error XD

    # Render the dashboard template with the asset data
    return render_template('dashboard.html', async_mode=appSocketIOContext.async_mode,
                           stock=stock)


@appBaseContext.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


@appBaseContext.errorhandler(500)
def page_not_found(error):
    return render_template('500.html'), 500


@appBaseContext.route('/get_latest_data')
def get_latest_article_data():
    latest_data = article_data[:10]
    return jsonify(latest_data)


def is_room_opened(name):
    print(appSocketIOContext.server.manager.rooms.items())
    for room_path, room_data in appSocketIOContext.server.manager.rooms.items():
        current_rooms = list(room_data.keys())
        if name in current_rooms:
            return True

    return False


def update_stock_price(ticker: str):
    while is_room_opened(ticker):
        price, datetime_utc = data_utilities.get_current_price(ticker)
        formatted_timestamp = datetime_utc.strftime("%d %B, %Y, %H:%M:%S UTC")
        appSocketIOContext.emit('update_price', {'price': price, 'date': formatted_timestamp},
                                namespace='/dashboard', to=ticker)
        appSocketIOContext.sleep(5)


def update_article_table(ticker):
    while is_room_opened(ticker):
        data_utilities.update_data(article_data)
        appSocketIOContext.sleep(2)


def update_sentiment_chart(ticker: str):
    print("Process got ticker:", ticker)
    #while is_room_opened(ticker):
    while True:
        print("Process starts processing data")
        df_json = data_utilities.get_sentiment_data(ticker)
        print(df_json)
        appSocketIOContext.emit('update_sentiment', {'df_data': df_json},
                                namespace='/dashboard', to=ticker)
        print("Process sent update_sentiment")
        appSocketIOContext.sleep(300)


@appSocketIOContext.on('connect', namespace='/dashboard')
def handle_connect():
    pass


@appSocketIOContext.on('update_price_request', namespace='/dashboard')
def handle_dashboard_request(data):
    # continuous async update
    # threads return when the last room session is closed
    join_room(room=data['ticker'], namespace='/dashboard')
    print('Client id=', request.sid, 'room=', data['ticker'], 'Requested updates')

    data['time_frame'] = 7

    stockBackendServer.subscribe(data)
    print("Subscription request sent")
    #socketio_app_context.start_background_task(target=update_sentiment_chart, ticker=data['ticker'])
    #threading.Thread(target=update_sentiment_chart, args=(data['ticker'], )).start()
    #eventlet.spawn_n(update_stock_price, data['ticker'])
    #eventlet.spawn_n(update_article_table, data['ticker'])


@appSocketIOContext.on('disconnect', namespace='/dashboard')
def handle_disconnect():
    print('Client id=', request.sid, 'disconnected. Left all rooms.')


if __name__ == '__main__':
    appSocketIOContext.run(appBaseContext, allow_unsafe_werkzeug=True, debug=True, use_reloader=False)

