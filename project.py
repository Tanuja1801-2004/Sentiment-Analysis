import cv2
import tkinter as tk
from tkinter import Label, Button, filedialog, Frame
from PIL import Image, ImageTk
from deepface import DeepFace
import matplotlib
matplotlib.use("TkAgg")  # Force TkAgg backend
import matplotlib.pyplot as plt
import datetime


class EmotionDetectorApp:
    def __init__(self, root):  # ✅ Fixed constructor
        self.root = root
        self.root.title("Emotion Detection")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")

        self.video_source = 0  # Default camera
        self.vid = cv2.VideoCapture(self.video_source)

        # Frame for Video Feed
        self.frame_video = Frame(root, bg="#34495e", padx=10, pady=10)
        self.frame_video.pack(pady=20)

        self.canvas = tk.Label(self.frame_video, bg="#34495e")
        self.canvas.pack()

        # Frame for Controls
        self.frame_controls = Frame(root, bg="#2c3e50")
        self.frame_controls.pack()

        self.emotion_label = Label(root, text="Emotion: ", font=("Arial", 18, "bold"), bg="#2c3e50", fg="#ecf0f1")
        self.emotion_label.pack(pady=10)

        self.start_recording_btn = Button(self.frame_controls, text="Start Recording", command=self.start_recording,
                                          font=("Arial", 12, "bold"), bg="#27ae60", fg="white", padx=10, pady=5)
        self.start_recording_btn.grid(row=0, column=0, padx=10, pady=5)

        self.stop_recording_btn = Button(self.frame_controls, text="Stop Recording", command=self.stop_recording,
                                         font=("Arial", 12, "bold"), bg="#e74c3c", fg="white", padx=10, pady=5,
                                         state=tk.DISABLED)
        self.stop_recording_btn.grid(row=0, column=1, padx=10, pady=5)

        self.plot_btn = Button(self.frame_controls, text="Show Emotion Graph", command=self.show_emotion_graph,
                               font=("Arial", 12, "bold"), bg="#f1c40f", fg="black", padx=10, pady=5)
        self.plot_btn.grid(row=0, column=2, padx=10, pady=5)

        self.btn_quit = Button(root, text="Quit", command=self.root.quit, font=("Arial", 12, "bold"), bg="#c0392b",
                               fg="white", padx=10, pady=5)
        self.btn_quit.pack(pady=10)

        self.recording = False
        self.emotion_log = []

        self.update_frame()

    def update_frame(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

            self.detect_emotion(frame)

        self.root.after(50, self.update_frame)

    def detect_emotion(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            self.emotion_label.config(text=f"Emotion: {emotion}")

            if self.recording:
                self.emotion_log.append((datetime.datetime.now(), emotion))
        except:
            self.emotion_label.config(text="Emotion: Not detected")

    def start_recording(self):
        self.recording = True
        self.emotion_log = []
        self.start_recording_btn.config(state=tk.DISABLED)
        self.stop_recording_btn.config(state=tk.NORMAL)

    def stop_recording(self):
        self.recording = False
        self.start_recording_btn.config(state=tk.NORMAL)
        self.stop_recording_btn.config(state=tk.DISABLED)

    def show_emotion_graph(self):
        if not self.emotion_log:
            return

        times, emotions = zip(*self.emotion_log)
        unique_emotions = list(set(emotions))
        emotion_counts = {emotion: [1 if e == emotion else 0 for e in emotions] for emotion in unique_emotions}

        plt.figure(figsize=(10, 5))
        for emotion, counts in emotion_counts.items():
            plt.plot(times, counts, label=emotion)

        plt.xlabel("Time")
        plt.ylabel("Emotion Presence")
        plt.title("Emotion Graph Over Time")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", ".png"), ("All Files", ".*")])
        if save_path:
            plt.savefig(save_path)

        plt.show()  # Only call this once at the end


# ✅ Fixed main check
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()
