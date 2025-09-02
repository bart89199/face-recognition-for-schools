import json

import requests

import settings


def send_data(door: bool, recognitions: list):
    data = json.dumps({
        "door": door,
        "recognitions": recognitions
    })


    headers = {
        "Authorization": f"Bearer {settings.KTOR_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(settings.KTOR_STATUS_URL, data, headers=headers)

        response.raise_for_status()

        settings.door_status = response.json()['status']

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при выполнении запроса: {e}")
    except Exception as e:
        print(f"Произошла другая ошибка: {e}")