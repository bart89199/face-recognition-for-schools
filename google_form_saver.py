import io
import time

import cv2
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from matplotlib.image import imread

import global_vars
from loader import load_googleapi, load_known_data, load_mediapipe_image
from saver import save_from_frame, save_data_on_disk


def export_file(file_id):
    try:
        drive_service = build("drive", "v3", credentials=global_vars.creds)
        request = drive_service.files().get_media(fileId=file_id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        drive_service.close()
    except HttpError as error:
        print(f"An error occurred(file id = {file_id}: {error}")
        file = None
    return file

def load_data_from_forms():
    print("Checking forms...")
    forms_service = build("forms", "v1", credentials=global_vars.creds)

    result = forms_service.forms().responses().list(formId=global_vars.GOOGLE_FORM_ID).execute()

    for response in result['responses']:
        response_id = response['responseId']
        if global_vars.saved_form_answers.__contains__(response_id):
            continue
        name = response['respondentEmail'].replace(global_vars.EMAIL_DOMAIN, '')

        for answer in response['answers'][global_vars.FORM_ANSWER_ID]['fileUploadAnswers']['answers']:
            file_id = str(answer['fileId'])
            file = export_file(file_id)
            if file is not None:
                frame = imread(file, 'jpeg')
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                save_from_frame(rgb_frame, name, f'file id = {file_id}, name = {name}')
            file.close()
        global_vars.saved_form_answers.append(response_id)
        save_data_on_disk()
    forms_service.close()
    print("Forms checked.")
    cur_time = time.time()
    global_vars.last_forms_check_time = cur_time

def main_google_form_saver():
    load_mediapipe_image()
    load_googleapi()
    load_known_data()
    load_data_from_forms()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_google_form_saver()