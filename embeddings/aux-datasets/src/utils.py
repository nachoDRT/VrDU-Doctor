import json


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
