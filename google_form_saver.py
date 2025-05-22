import io
import os
import time

import cv2
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from matplotlib.image import imread

import frame_handler
import global_vars
from frame_handler import read_frame_file
from loader import load_googleapi, load_known_data, load_mediapipe, load_main
from saver import save_from_frame, save_data_on_disk, forget_face


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

def get_forms_answers():
    forms_service = build("forms", "v1", credentials=global_vars.creds)

    responses = forms_service.forms().responses().list(formId=global_vars.GOOGLE_FORM_ID).execute()
    forms_service.close()
    return responses

def forget_forms_response(response, delete_format = '0'):
    if delete_format == '0' or delete_format == '2':
        global_vars.saved_form_answers.remove(response['responseId'])
    if delete_format == '0' or delete_format == '1':
        name = response['respondentEmail'].replace(global_vars.EMAIL_DOMAIN, '')
        for answer in response['answers'][global_vars.FORM_ANSWER_ID]['fileUploadAnswers']['answers']:
            file_id = str(answer['fileId'])
            filepath = os.path.join(global_vars.SAVED_FRAMES_FOLDER, name)
            filepath = os.path.join(filepath, f'{file_id}.jpg')
            i = global_vars.known_face_names.index(name)
            j = global_vars.known_face_images[i].index(filepath)
            forget_face(name, j)
    save_data_on_disk()

def load_response(response, skip_if_contains = True, save_format = '0'):
    response_id = response['responseId']
    if global_vars.saved_form_answers.__contains__(response_id) and skip_if_contains:
        return
    name = response['respondentEmail'].replace(global_vars.EMAIL_DOMAIN, '')

    if save_format == '0' or save_format == '2':
        global_vars.saved_form_answers.append(response_id)

    if save_format == '0' or save_format == '1':
        for answer in response['answers'][global_vars.FORM_ANSWER_ID]['fileUploadAnswers']['answers']:
            file_id = str(answer['fileId'])
            file = export_file(file_id)
            if file is not None:
                frame = read_frame_file(file)
                save_from_frame(frame, name, f'{file_id}.jpg', f'file id = {file_id}, name = {name}')
            file.close()

    save_data_on_disk()

def load_data_from_forms():
    print("Checking forms...")

    result = get_forms_answers()

    for response in result['responses']:
        load_response(response)
    print("Forms checked.")
    cur_time = time.time()
    global_vars.last_forms_check_time = cur_time

def main_google_form_saver():
    load_main()
    load_googleapi()
    load_known_data()
    load_mediapipe()
    load_data_from_forms()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_google_form_saver()