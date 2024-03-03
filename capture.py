import cv2
import time
import requests
import os
from dotenv import load_dotenv
import boto3
import json
import threading

from kivy.app import App
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.core.text import LabelBase
from kivy.animation import Animation
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout

Builder.load_string('''
<Button>:
    font_name: 'Roboto'
    font_size: 40
    size_hint_y: None
    height: '60dp'
    background_color: 0.1, 0.5, 0.6, 1
    color: 1, 1, 1, 1

<Label>:
    font_name: 'Roboto'
    font_size: 40
    color: 0.1, 0.5, 0.6, 1

<BoxLayout>:
    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: self.pos
            size: self.size

    Image:
        source: './parkwise_logo.png'  # Path of logo
        # size_hint_y: None
        height: 600
        width: 400
        # allow_stretch: True
        keep_ratio: True
''')

load_dotenv()

selected_variable = 2

# create global variable here najaaaa
api_url_text = ""
license_api_url_input_text = ""
api_key_input_text = ""
device_id_text = ""

class CameraSelectorApp(App):
    def build(self):
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        text_inputs_layout = GridLayout(cols=1, spacing=30, size_hint=(None, None))
        text_inputs_layout.width = 700
        text_inputs_layout.height = 300
        
        self.api_url = TextInput(size_hint_y=None, height=50)
        self.api_url.hint_text = "API Url"
        text_inputs_layout.add_widget(self.api_url)
        
        self.api_url_input = TextInput(size_hint_y=None, height=50)
        self.api_url_input.hint_text = "License Api Url"
        text_inputs_layout.add_widget(self.api_url_input)
        
        self.api_key_input = TextInput(size_hint_y=None, height=50)
        self.api_key_input.hint_text = "License Api Key Url"
        text_inputs_layout.add_widget(self.api_key_input)
        
        self.device_id = TextInput(size_hint_y=None, height=50)
        self.device_id.hint_text = "Device ID"
        text_inputs_layout.add_widget(self.device_id)
        text_inputs_layout.pos_hint = {'right': 1, 'top': 1}
        
        layout.add_widget(text_inputs_layout)

        self.label = Label(text="Camera Selected: ", size_hint_y=None, height=100)
        layout.add_widget(self.label)

        options = get_video_sources()
        # options = []

        self.dropdown = DropDown()
        for option in options:
            btn = Button(text=option, size_hint_y=None, height=120)
            btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
            self.dropdown.add_widget(btn)

        # Create main button
        mainbutton = Button(text='Select Camera')
        mainbutton.bind(on_release=self.dropdown.open)
        layout.add_widget(mainbutton)

        # Button to update the selected camera
        update_button = Button(text="Use this Camera", size_hint_y=None, height=130)
        update_button.bind(on_release=self.update_variable)
        layout.add_widget(update_button)

        self.dropdown.bind(on_select=lambda instance, x: setattr(mainbutton, 'text', x))
        self.dropdown.bind(on_select=self.update_label)

        return layout

    def update_variable(self, instance):
        global selected_variable
        selected_variable = self.label.text.split()[-1]

        detect_license_plate(int(int(selected_variable) - 1), resize_width=500, resize_height=None, quality=88)

    def update_label(self, instance, x):
        global api_url_text
        global license_api_url_input_text
        global api_key_input_text
        global device_id_text
        api_url_text = self.api_url.text
        license_api_url_input_text = self.api_url_input.text
        api_key_input_text = self.api_key_input.text
        device_id_text = self.device_id.text
        
        print("api url text", api_url_text)
        print("license_api_url_input_text", license_api_url_input_text)
        print("api_key_input_text", api_key_input_text)
        print("device_id_text", device_id_text)
        
        self.label.text = "Camera Selected: " + str(x)
        anim = Animation(color=(0.5, 0.5, 0.5, 1)) + Animation(color=(1, 1, 1, 1))
        anim.start(self.label)

def start_detection_in_thread(video_source):
    detection_thread = threading.Thread(target=detect_license_plate, args=(video_source,))
    detection_thread.start()

def update_variable(new_value):
    global selected_variable
    selected_variable = new_value
    messagebox.showinfo("Variable Updated", f"Selected Variable: {selected_variable}")
    print(selected_variable)
    
    detect_license_plate(int(selected_variable), resize_width=500, resize_height=None, quality=88)

def get_video_sources():
    sources = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            sources.append(str(i + 1))
            cap.release()
    return sources


def send_to_parkwise_api(license_plate, image_url):
    payload = {
        'licensePlate': license_plate,
        'licensePlateUrl': image_url,
        'deviceId': device_id_text
    }
    headers = {'Content-Type': 'application/json'}

    resp = requests.post(api_url_text, data=json.dumps(payload), headers=headers)

    print(resp)

def upload_file_to_s3(file_path, file_name):
    s3 = boto3.client('s3')
    bucket_name = 'parkwise-kmutt'

    file_key = file_name
    
    s3.upload_file(file_path, bucket_name, file_key)
    expiration_time = 365 * 24 * 60 * 60

    url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': file_key,
                'ResponseContentDisposition': 'inline'
            },
            ExpiresIn=expiration_time
        )
    print(url)
    return url

def send_to_api(image_path):
    payload = {'crop': '1', 'rotate': '1'}
    files = {'image': open(image_path, 'rb')}

    headers = {
        'Apikey': api_key_input_text,
    }

    try:
        response = requests.post(license_api_url_input_text, files=files, data=payload, headers=headers)
    except Exception as E:
        print("Exception is " + e)
        

    if response.status_code == 200:
        if response.text:
            data = response.json()
            if data:
                first_dict = data[0]
                lpr_value = first_dict.get('lpr')
            if lpr_value:
                print("License Plate Number:", lpr_value)
                return lpr_value
            else:
                print("No value for 'lpr' key in the dictionary.")
            print(response.json())
        else:
            print("Success, but no content returned.")
    elif response.status_code == 204:
        print("Success, but no content returned.")
    elif response.status_code == 500:
        print("Something wrong with API.")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def resize_image(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    """
    Resize the input image.
    Parameters:
        image (numpy.ndarray): Input image.
        width (int, optional): Target width. If None, it is calculated 
based on the given height.
        height (int, optional): Target height. If None, it is calculated 
based on the given width.
        interpolation (int, optional): Interpolation method. Default is 
cv2.INTER_AREA.

    Returns:
        numpy.ndarray: Resized image.
    """
    if width is None and height is None:
        raise ValueError("At least one of 'width' or 'height' must be provided.")

    if width is None:
        aspect_ratio = height / float(image.shape[0])
        width = int(image.shape[1] * aspect_ratio)
    elif height is None:
        aspect_ratio = width / float(image.shape[1])
        height = int(image.shape[0] * aspect_ratio)

    resized_image = cv2.resize(image, (width, height), interpolation=interpolation)
    return resized_image

def resize_and_save(image_path, output_path, width=None, height=None, quality=95):
    """
    Resize the input image, save it with reduced file size.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
        width (int, optional): Target width. If None, it is calculated 
based on the given height.
        height (int, optional): Target height. If None, it is calculated 
based on the given width.
        quality (int, optional): Compression quality (0-100). Default is 
95.
    Returns:
        None
    """
    # Read the original image
    original_image = cv2.imread(image_path)

    # Resize the image
    resized_image = resize_image(original_image, width=width, height=height)

    # Save the resized image with reduced file size
    cv2.imwrite(output_path, resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    print(f"Resized image saved to: {output_path}")


    send_to_parkwise_api_flag = False
    license_plate = ""

    if send_to_api(output_path):
        send_to_parkwise_api_flag = True
        license_plate = send_to_api(output_path)

    s3_file_name = os.path.basename(output_path)
    if send_to_parkwise_api_flag:
        send_to_parkwise_api(license_plate, upload_file_to_s3(output_path, s3_file_name))

def detect_license_plate(video_source, save_path="captured_frames/", resize_width=None, resize_height=None, quality=95):
    cap = cv2.VideoCapture(video_source)
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    if not cv2.os.path.exists(save_path):
        cv2.os.makedirs(save_path)

    capture_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y - 30), (x + w + 30, y + 20 + h), (0, 0, 255), 2)
            capture_count += 1

            if capture_count % 2 == 0:
                elapsed_time = time.time() - start_time
                if elapsed_time >= 2:
                    frame_filename = f"{save_path}captured_frame_{cv2.getTickCount()}.jpg"
                    cv2.imwrite(frame_filename, frame)
                    print(f"Captured: {frame_filename}")

                    resized_filename = f"{save_path}resized_frame_{cv2.getTickCount()}.jpg"
                    resize_and_save(frame_filename, resized_filename, width=resize_width, height=resize_height, quality=quality)

                    time.sleep(4)
                    start_time = time.time()
                    capture_count = 0

        cv2.imshow('Detected Plates', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    CameraSelectorApp().run()