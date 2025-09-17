import asyncio
import json

import numpy as np
import requests

import settings
from settings import SystemStatus


def send_data(door: bool, recognitions: list, system_status: str) -> bool:
    data = json.dumps({
        "door": door,
        "recognitions": recognitions,
        "system_status": system_status
    })


    headers = {
        "Authorization": f"Bearer {settings.KTOR_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(settings.KTOR_STATUS_URL, data, headers=headers)

        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return False
    except Exception as e:
        print(f"Произошла другая ошибка: {e}")
        return False
    return True

async def send_data_loop():
    while settings.system_status == SystemStatus.STARTING:
        send_data(False, [], SystemStatus.STARTING.name)
        await asyncio.sleep(1)
    print("Kotlin connection started")
    while settings.system_status == SystemStatus.RUNNING:
        status = send_data(settings.door_opened, list(settings.cur_alive_names), settings.system_status.name)
        settings.cur_alive_names.clear()
        if not status:
            await asyncio.sleep(1)
        else:
            await asyncio.sleep(0.2)
    send_data(False, [], SystemStatus.STOPPING.name)
    print("Kotlin connection stoped")