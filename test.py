import requests
import csv

from datetime import datetime

cur_datetime = datetime.now()
cur_year = cur_datetime.year
cur_month = cur_datetime.month
cur_day = cur_datetime.day
cur_year_month_day = "{}_{}_{}".format(cur_year, cur_month, cur_day)


url = "http://localhost:5002/api/detect"

payload = {"token": "12345678", "alg": "gmm", "audio_path": "Cough1.wav"}

response = requests.post(url, json=payload)

print(response.json())



if __name__ == '__main__':
    test1()
