import cv2
import time
import requests
import os
from dotenv import load_dotenv
import boto3
import json

load_dotenv()

def send_to_parkwise_api(license_plate, image_url):
    api_url = os.environ.get("API_URL")
    
    payload = {
        'licensePlate': license_plate,
        'licensePlateUrl': image_url,
        'zoneId': 1
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

def send_to_api(image_path):
    api_url = os.environ.get("LICENSE_API_URL")
    payload = {'crop': '1', 'rotate': '1'}
    files = {'image': open(image_path, 'rb')}

    api_key = os.environ.get("LICENSE_API_KEY")
    headers = {
        'Apikey': api_key,
    }

    response = requests.post(api_url, files=files, data=payload, headers=headers)

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
    else:
        print(f"Error: {response.status_code} - {response.text}")

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

def detect_license_plate(video_source=0, save_path="captured_frames/", resize_width=None, resize_height=None, quality=95):
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
            cv2.rectangle(frame, (x, y - 30), (x + w + 30, y + 20 + h), 
(0, 0, 255), 2)
            capture_count += 1

            if capture_count % 2 == 0:
                elapsed_time = time.time() - start_time
                if elapsed_time >= 2:  # Check if 2 seconds have passed
                    frame_filename = f"{save_path}captured_frame_{cv2.getTickCount()}.jpg"
                    cv2.imwrite(frame_filename, frame)
                    print(f"Captured: {frame_filename}")

                    resized_filename = f"{save_path}resized_frame_{cv2.getTickCount()}.jpg"
                    resize_and_save(frame_filename, resized_filename, width=resize_width, height=resize_height, quality=quality)

                    time.sleep(4)  # Introduce a delay of 4 seconds
                    start_time = time.time()  # Reset the start time
                    capture_count = 0

        cv2.imshow('Detected Plates', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_license_plate(resize_width=500, resize_height=None, quality=88)
   