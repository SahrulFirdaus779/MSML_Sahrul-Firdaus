import requests
import random
import json

URL = "http://localhost:8000/predict"

# Fungsi untuk generate satu data dummy
def generate_dummy():
    return {
        "TransactionAmount": random.randint(10000, 1000000),
        "TransactionType": random.randint(1, 3),
        "Location": random.randint(1, 5),
        "Channel": random.randint(1, 2),
        "CustomerAge": random.randint(18, 60),
        "CustomerOccupation": random.randint(1, 4),
        "TransactionDuration": random.randint(10, 300),
        "LoginAttempts": random.randint(1, 5),
        "AccountBalance": random.randint(10000, 1000000),
        "AgeGroup": random.randint(1, 3),
        "TransactionSize": random.randint(1, 5)
    }

# Kirim batch data dummy (request sukses)
def send_batch(n=10):
    payload = [generate_dummy() for _ in range(n)]
    resp = requests.post(URL, json=payload)
    print(f"Batch request status: {resp.status_code}")
    print(resp.json())

# Kirim request error: payload kosong
def send_empty():
    resp = requests.post(URL, json=[])
    print(f"Empty payload status: {resp.status_code}")
    print(resp.text)

# Kirim request error: field salah
def send_invalid():
    payload = [{"WrongField": 123}]
    resp = requests.post(URL, json=payload)
    print(f"Invalid payload status: {resp.status_code}")
    print(resp.text)

if __name__ == "__main__":
    send_batch(20)      # Request sukses (20 data)
    send_empty()        # Request error (payload kosong)
    send_invalid()      # Request error (field salah)
