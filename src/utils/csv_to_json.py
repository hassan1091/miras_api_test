import csv
import json
import os
from pathlib import Path


def read_old_factory_data_to_json():
    data = []

    try:
        old_factory_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'csv', 'old_factory_data.csv')
        with open(old_factory_data_path, mode='r', newline='', encoding='ISO-8859-1') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                data.append(row)

        data = json.dumps(data, indent=4)
        return json.loads(data)

    except Exception as e:
        print(f"Error: {e}")
        return None
