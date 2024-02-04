import cv2
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class LicensePlateDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detector")

        self.video_source = 0
        self.cap = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.video_source_label = tk.Label(root, text="Video Source:")
        self.video_source_label.pack()

        self.video_source_select = tk.StringVar(root)
        self.video_source_select.set("0")
        self.video_source_dropdown = tk.OptionMenu(root, self.video_source_select, *self.get_video_sources())
        self.video_source_dropdown.pack()

        self.btn_browse = tk.Button(root, text="Browse Video", command=self.browse_video)
        self.btn_browse.pack(pady=10)

        self.btn_start_detection = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.btn_start_detection.pack(pady=10)

        self.btn_quit = tk.Button(root, text="Quit", command=self.quit)
        self.btn_quit.pack(pady=10)

        self.image_on_canvas = None

    def get_video_sources(self):
        sources = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                sources.append(str(i))
                cap.release()
        return sources
    
    

    def browse_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.video_source = file_path
            self.cap = cv2.VideoCapture(self.video_source)

    def start_detection(self):
        self.video_source = int(self.video_source_select.get())
        self.cap = cv2.VideoCapture(self.video_source)
        self.detect_license_plate()

    def detect_license_plate(self):
        plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to capture frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

            for (x, y, w, h) in plates:
                cv2.rectangle(frame, (x, y - 30), (x + w + 30, y + 20 + h), (0, 0, 255), 2)

            # Display the frame in the Tkinter window
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)

            if self.image_on_canvas is None:
                self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            else:
                self.canvas.itemconfig(self.image_on_canvas, image=img)

            self.root.update()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()

    def quit(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateDetectorGUI(root)
    root.mainloop()