from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import time
import threading
from dotenv import load_dotenv
import os
import base64
import requests
import json
import boto3

load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)

selected_variable = 1
video_source = None
detection_thread = None
cap = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/update_camera", methods=["POST"])
def update_camera():
    selected_camera = request.form.get("selected_camera")

    # Perform any necessary logic to update the camera here
    # For example, you might update the global variable or restart the camera stream

    return jsonify({"status": "success"})


def start_detection_in_thread():
    global cap, video_source

    # Create VideoCapture object
    cap = cv2.VideoCapture(video_source)

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                # Your license plate detection logic here
                # For example, you can call detect_license_plate function
                # with frame as an argument

                license_plate = detect_license_plate(frame)
                if license_plate:
                    send_to_parkwise_api_flag = send_to_api_and_parkwise(frame, license_plate)
                    if send_to_parkwise_api_flag:
                        print(f"License plate {license_plate} sent to Parkwise API")

                # Convert the frame to base64 and send it to the browser
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_stream', {'image': frame_base64})

        time.sleep(1)

def detect_license_plate(frame=None, resize_width=None, resize_height=None, quality=95):
    global cap

    if frame is None:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            return None

    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y - 30), (x + w + 30, y + 20 + h), (0, 0, 255), 2)
        license_plate_roi = frame[y-30:y+20+h, x:x+w]
        license_plate = recognize_license_plate(license_plate_roi)
        return license_plate

    return None



def recognize_license_plate(license_plate_roi):
    # Your license plate recognition logic here
    # For example, you can use OCR (Optical Character Recognition) libraries like Tesseract
    # to extract text from the license plate region
    # You'll need to install pytesseract and pillow libraries for this example:
    # pip install pytesseract pillow
    try:
        import pytesseract
        from PIL import Image

        # Convert BGR to RGB
        license_plate_roi_rgb = cv2.cvtColor(license_plate_roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(license_plate_roi_rgb)

        # Use pytesseract to extract text
        license_plate_text = pytesseract.image_to_string(pil_image)
        return license_plate_text.strip()
    except Exception as e:
        print(f"Error recognizing license plate: {e}")
        return None

def send_to_parkwise_api(license_plate, image_url):
    api_url = os.environ.get("API_URL")
    device_id = os.environ.get("DEVICE_ID")

    payload = {
        'licensePlate': license_plate,
        'licensePlateUrl': image_url,
        'zoneId': 1,
        'deviceId': device_id
    }

    headers = {'Content-Type': 'application/json'}  # Specify JSON content type

    resp = requests.post(api_url, data=json.dumps(payload), headers=headers)

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

def send_to_api_and_parkwise(frame, license_plate):
    api_url = os.environ.get("LICENSE_API_URL")
    payload = {'crop': '1', 'rotate': '1'}
    image_path = 'temp_license_plate.jpg'

    # Save the license plate region as an image
    cv2.imwrite(image_path, frame)

    # Upload the image to S3
    s3_file_name = os.path.basename(image_path)
    s3_url = upload_file_to_s3(image_path, s3_file_name)

    # Send the image to the license plate recognition API
    response = send_to_api(image_path)
    if response and 'lpr' in response:
        recognized_license_plate = response['lpr']

        # Check if recognized license plate matches with the detected one
        if recognized_license_plate == license_plate:
            # Send to Parkwise API
            send_to_parkwise_api(license_plate, s3_url)
            return True

    return False

def resize_image(image, width=None, height=None, 
interpolation=cv2.INTER_AREA):
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


    send_to_parkwise_api_flag = False  # Rename the variable to avoid naming conflict
    license_plate = ""

    if send_to_api(output_path):
        send_to_parkwise_api_flag = True
        license_plate = send_to_api(output_path)

    s3_file_name = os.path.basename(output_path)
    if send_to_parkwise_api_flag:
        send_to_parkwise_api(license_plate, upload_file_to_s3(output_path, s3_file_name))

if __name__ == '__main__':
    # Start the detection thread
    detection_thread = threading.Thread(target=start_detection_in_thread)
    detection_thread.start()

    # Start the Flask app with SocketIO
    socketio.run(app, debug=True, port=8000)
