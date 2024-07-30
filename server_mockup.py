import time
from flask import Flask, request, jsonify
from threading import Lock
from apscheduler.schedulers.background import BackgroundScheduler
from Recommender import Recommender

app = Flask(__name__)
recommender = Recommender(data_path='Creator Supplier Score Matrix.csv')
lock = Lock()

def update_recommender():
    with lock:
        print("Updating recommender engine...")
        recommender.update()
        print("Update complete.")

scheduler = BackgroundScheduler()
scheduler.add_job(func=update_recommender, trigger="interval", days=1)
scheduler.start()

@app.route('/get_scores', methods=['POST'])
def get_scores():
    user_id = request.json.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400

    with lock:
        try:
            scores = recommender.get_scores(user_id)
            return jsonify(scores)
        except AssertionError as e:
            return jsonify({'error': str(e)}), 404

@app.route('/add_data', methods=['POST'])
def add_data():
    data = request.json.get('data')
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    with lock:
        try:
            recommender.add_data(data)
            return jsonify({'message': 'Data added successfully'})
        except AssertionError as e:
            return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False)
    finally:
        scheduler.shutdown()
