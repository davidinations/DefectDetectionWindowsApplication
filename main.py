import tkinter
import tkinter.messagebox
from tracemalloc import start
import customtkinter
import os
import cv2
import threading
import time
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Modes: "System" (standard), "Dark", "Light"
customtkinter.set_appearance_mode("System")
# Themes: "blue" (standard), "green", "dark-blue"
customtkinter.set_default_color_theme("blue")

# Get the absolute path to the directory of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the icon path
icon_path = os.path.join(script_dir, "assets/images/favicon.ico")

# Load the model path
model_path = os.path.join(script_dir, "best.pt")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Defect Detection System From Images")
        self.geometry(f"{1280}x{720}+{150}+{50}")
        self.iconbitmap(icon_path)

        # configure grid layout (4x4) For Responsive Design At The Startup
        self.grid_columnconfigure((1, 2, 3), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(
            self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="Menu", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(
            self.sidebar_frame, text="Add Image", command=self.add_image_event)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(
            self.sidebar_frame, text="History", command=self.sidebar_button_event)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(
            self.sidebar_frame, text="Exit", command=self.exit_button_event)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(
            row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create label in main frame
        self.label = customtkinter.CTkLabel(
            self,
            text="Welcome to the Defect Detection Application\n\n",
            width=1150,
            font=customtkinter.CTkFont(size=20, weight="bold")
        )
        self.label.grid(row=0, column=1, columnspan=3,
                        padx=(20, 20), pady=(20, 0), sticky="n")

        # create frame in left main frame
        self.image_before_processing = customtkinter.CTkFrame(self)
        self.image_before_processing.grid(
            row=1, column=1, padx=(20, 20), pady=(20, 0), sticky="nw")
        self.image_before_processing_label = customtkinter.CTkLabel(
            self.image_before_processing, text="Image Before Processing", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.image_before_processing_label.grid(
            row=0, column=0, padx=20, pady=(20, 10))
        self.image_before_processing_canvas = customtkinter.CTkCanvas(
            self.image_before_processing, width=500, height=500)
        self.image_before_processing_canvas.grid(
            row=1, column=0, padx=20, pady=(10, 20))
        self.image_before_processing_canvas.create_rectangle(
            0, 0, 500, 500, fill="gray")

        # create frame in middle main frame for buttons
        self.buttons_frame = customtkinter.CTkFrame(self)
        self.buttons_frame.grid(
            row=1, column=2, padx=(20, 20), pady=(20, 0), sticky="ew")
        self.buttons_frame.grid_columnconfigure(0, weight=1)
        self.process_button = customtkinter.CTkButton(
            self.buttons_frame, text="Process Image", command=self.process_button_event, width=20)
        self.process_button.grid(row=0, column=0, padx=(
            10, 10), pady=(20, 5), sticky="ew")
        self.save_button = customtkinter.CTkButton(
            self.buttons_frame, text="Save Image", command=self.save_button_event, width=20)
        self.save_button.grid(row=1, column=0, padx=(
            10, 10), pady=(5, 20), sticky="ew")

        # create frame in right main frame
        self.image_after_processing = customtkinter.CTkFrame(self)
        self.image_after_processing.grid(
            row=1, column=3, padx=(20, 20), pady=(20, 0), sticky="ne")
        self.image_after_processing_label = customtkinter.CTkLabel(
            self.image_after_processing, text="Image After Processing", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.image_after_processing_label.grid(
            row=0, column=0, padx=20, pady=(20, 10))
        self.image_after_processing_canvas = customtkinter.CTkCanvas(
            self.image_after_processing, width=500, height=500)
        self.image_after_processing_canvas.grid(
            row=1, column=0, padx=20, pady=(10, 20))
        self.image_after_processing_canvas.create_rectangle(
            0, 0, 500, 500, fill="gray")

        # set default values
        self.sidebar_button_2.configure(
            state="disabled", text="History")
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.save_button.configure(
            state="disabled", text="Save Image")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")

    def process_button_event(self):
        global file_before_processing

        def process_image():
            # Start The Timer
            start_time = time.time()

            # Process the image
            global model_path, image
            image = cv2.imread(file_before_processing)
            model = YOLO(model_path)
            results = model.predict(image)
            result = results[0]
            output = []
            for box in result.boxes:
                x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                class_id = box.cls[0].item()
                prob = round(box.conf[0].item(), 2)
                output.append(
                    [x1, y1, x2, y2, result.names[class_id], prob])
                # Draw rectangle (bounding box) on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Optionally, add label and probability to the box
                label = f"{result.names[class_id]}: {prob}"
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Resize the image to fit the canvas if necessary
            image = cv2.resize(image, (500, 500))

            # Convert the OpenCV image (BGR format) to a PIL image (RGB format)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            # Convert the image to a format tkinter can use
            photo = ImageTk.PhotoImage(image)

            # Clear the canvas
            self.image_after_processing_canvas.delete("all")

            # Add the image to the canvas
            self.image_after_processing_canvas.create_image(
                0, 0, image=photo, anchor="nw")

            # Keep a reference to the image to prevent it from being garbage collected
            self.image_after_processing_canvas.image = photo

            # State of the buttons
            self.save_button.configure(state="normal", text="Save Image")

            # Stop Progress Bar
            self.progressbar_1.stop()
            self.slider_progressbar_frame.grid_forget()

            # Stop The Timer
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Show Result Time In A Message Box
            tkinter.messagebox.showinfo(
                "Time Result", f"Image has been processed in {elapsed_time:.3f} seconds.")

        if 'file_before_processing' not in globals() or file_before_processing == "":
            tkinter.messagebox.showerror(
                "Error", "Please add an image first.")

        else:
            # Show Progress Bar
            self.slider_progressbar_frame = customtkinter.CTkFrame(
                self, fg_color="transparent")
            self.slider_progressbar_frame.grid(
                row=2, column=1, columnspan=3, padx=(20, 20), pady=(5, 5), sticky="sew")
            self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
            self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)
            self.progressbar_1 = customtkinter.CTkProgressBar(
                self.slider_progressbar_frame)
            self.progressbar_1.grid(row=0, column=0, padx=(
                10, 10), pady=(5, 5), sticky="nsew")
            self.progressbar_1.configure(mode="indeterminnate")
            self.progressbar_1.start()

        # Create the thread
        process_image_thread = threading.Thread(target=process_image)

        # Start the thread
        process_image_thread.start()

    def add_image_event(self):
        global file_before_processing, width, height
        file_before_processing = filedialog.askopenfilename(
            filetypes=[('Allowed Files', '*.bmp;*.jpg;*.jpeg;*.png'),
                       ('BMP Files', '*.bmp'),
                       ('JPEG Files', '*.jpg;*.jpeg'),
                       ('PNG Files', '*.png')
                       ])
        if file_before_processing != "":
            # Load the image
            image = Image.open(file_before_processing)
            # Get the image size
            width, height = image.size
            # Resize the image to fit the canvas if necessary
            image = image.resize((500, 500))
            # Convert the image to a format tkinter can use
            photo = ImageTk.PhotoImage(image)

            # Clear the canvas
            self.image_before_processing_canvas.delete("all")
            self.image_after_processing_canvas.delete("all")
            self.image_after_processing_canvas.create_rectangle(
                0, 0, 500, 500, fill="gray")

            # Add the image to the canvas
            self.image_before_processing_canvas.create_image(
                0, 0, image=photo, anchor="nw")

            # Keep a reference to the image to prevent it from being garbage collected
            self.image_before_processing_canvas.image = photo

    def save_button_event(self):
        global image, width, height
        file_after_processing = filedialog.asksaveasfilename(
            filetypes=[('JPG Files', '*.jpg'),
                       ('PNG Files', '*.png'),
                       ('BMP Files', '*.bmp')],
            defaultextension=".jpg"
        )

        # Assuming image is a PIL Image object
        if image is not None and file_after_processing:
            # Resize the image to its original size before saving
            image = image.resize((width, height))
            # Save the PIL Image
            image.save(file_after_processing)
            if tkinter.messagebox.askokcancel("The File Has Been Saved", "Do you want to open the saved file?"):
                directory = os.path.dirname(file_after_processing)
                os.startfile(directory)
        else:
            messagebox.showinfo(
                "Unexpected error occurred. The image has not been saved.")

    def exit_button_event(self):
        if tkinter.messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.destroy()


if __name__ == "__main__":
    app = App()

    app.mainloop()
