import eventlet
eventlet.monkey_patch() # has to be called here

from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit
import data_utilities

data = []
app_context = Flask(__name__)
socketio_app_context = SocketIO(app_context, logger=True, engineio_logger=True)


@app_context.route('/')
def index():
    return render_template('index.html')


@app_context.route('/dashboard')
def dashboard():
    asset_id = request.args.get('id')

    # Perform any necessary data retrieval based on 'asset_id'
    # For example, fetch data related to the asset with the provided ID

    # Render the dashboard template with the asset data
    return render_template('dashboard.html', asset_id=asset_id, async_mode=socketio_app_context.async_mode)


@app_context.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


@app_context.errorhandler(500)
def page_not_found(error):
    return render_template('500.html'), 500


@app_context.route('/get_latest_data')
def get_latest_data():
    latest_data = data[:10]
    return jsonify(latest_data)


def update_article_table():
    while True:
        data_utilities.update_data(data)
        socketio_app_context.emit('update', {'data': data}, namespace='/')
        socketio_app_context.sleep(5)


@socketio_app_context.on('connect', namespace='/')
def handle_connect():
    emit('update', {'data': data})


# async article table update
eventlet.spawn(update_article_table)

if __name__ == '__main__':
    socketio_app_context.run(app_context, debug=True)
