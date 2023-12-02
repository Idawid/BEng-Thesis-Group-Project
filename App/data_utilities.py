import random
import datetime


def randomize_data():
    return {
        'date': datetime.datetime.now().strftime('%Y-%m-%d'),
        'headline': f'Random Headline {random.randint(1, 100)}',
        'description': f'Random Description {random.randint(1, 100)}',
        'sentiment': round(random.uniform(-1, 1), 2)
    }


def update_data(data):

    new_row = randomize_data()
    data.append(new_row)
    if len(data) > 500:
        data.pop(0)
    return data
