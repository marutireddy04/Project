import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import sys
import sqlite3

sys.path.append(r'C:\Users\shital\Desktop\python\models')
import CNNModel

global fn, current_user
fn = ""
current_user = "guest"

# Initialize DB
def create_users_table():
    conn = sqlite3.connect("evaluation.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

def create_predictions_table():
    conn = sqlite3.connect("evaluation.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        image_path TEXT,
        result TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

def log_prediction(username, image_path, result):
    conn = sqlite3.connect("evaluation.db")
    c = conn.cursor()
    c.execute("INSERT INTO predictions (username, image_path, result) VALUES (?, ?, ?)",
              (username, image_path, result))
    conn.commit()
    conn.close()

create_users_table()
create_predictions_table()

root = tk.Tk()
root.configure(background="seashell2")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Automated Lung Cancer Detection via CNN")

img = ImageTk.PhotoImage(Image.open("C:\\Users\\shital\\Desktop\\python\\Images\\l1.jpg"))
img2 = ImageTk.PhotoImage(Image.open("C:\\Users\\shital\\Desktop\\python\\Images\\l2.jpg"))
img3 = ImageTk.PhotoImage(Image.open("C:\\Users\\shital\\Desktop\\python\\Images\\l3.jpg"))

logo_label = tk.Label()
logo_label.place(x=0, y=0)

x = 1
def move():
    global x
    if x == 4:
        x = 1
    logo_label.config(image=[img, img2, img3][x-1])
    x += 1
    root.after(2000, move)
move()

lbl = tk.Label(root, text="Automated Lung Cancer Detection via CNN", font=('times', 35, 'bold'), height=1, width=60, bg="maroon", fg="white")
lbl.place(x=0, y=0)

frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=220, height=350, bd=5, font=('times', 14, ' bold '), bg="blue2")
frame_alpr.place(x=10, y=90)

def update_label(str_T):
    result_label = tk.Label(root, text=str_T, width=50, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=300, y=450)

def openimage():
    global fn
    fileName = askopenfilename(initialdir='D:/22SS137 Lungs  Cancer/lung cancer/test_set', title='Select image',
                               filetypes=[("all files", "*.*")])
    IMAGE_SIZE = 200
    fn = fileName
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE, 200))
    img = np.array(img)
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    img = tk.Label(root, image=imgtk, height=250, width=250)
    img.image = imgtk
    img.place(x=300, y=100)

def convert_grey():
    global fn
    IMAGE_SIZE = 200
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE, 200))
    img = np.array(img)
    x1, y1 = img.shape[0], img.shape[1]
    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)
    gs = cv2.resize(gs, (x1, y1))
    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    img2 = tk.Label(root, image=imgtk, height=250, width=250, bg='white')
    img2.image = imgtk
    img2.place(x=580, y=100)
    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im)
    img3 = tk.Label(root, image=imgtk, height=250, width=250)
    img3.image = imgtk
    img3.place(x=880, y=100)

def test_model_proc(fn):
    from keras.models import load_model
    IMAGE_SIZE = 64
    model = load_model('lung_model.h5')
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3).astype('float32') / 255.0
    prediction = model.predict(img)
    plant = np.argmax(prediction)
    return ["Benign case", "Malignant case", "Normal case"][plant]

def test_model():
    global fn
    if fn != "":
        update_label("Model Testing Start...............")
        start = time.time()
        result = test_model_proc(fn)
        end = time.time()
        msg = f"Image Testing Completed..\n{result} disease is detected\nExecution Time: {end-start:.4f} seconds"
        update_label(msg)
        log_prediction(current_user, fn, result)
        fn = ""
    else:
        update_label("Please Select Image For Prediction....")

def window():
    root.destroy()

button1 = tk.Button(frame_alpr, text=" Select_Image ", command=openimage, width=15, height=1, font=('times', 15, ' bold '), bg="white", fg="black")
button1.place(x=10, y=40)

button2 = tk.Button(frame_alpr, text="Image_preprocess", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '), bg="white", fg="black")
button2.place(x=10, y=100)

button4 = tk.Button(frame_alpr, text="CNN_Prediction", command=test_model, width=15, height=1, bg="white", fg="black", font=('times', 15, ' bold '))
button4.place(x=10, y=160)

exit = tk.Button(frame_alpr, text="Exit", command=window, width=15, height=1, font=('times', 15, ' bold '), bg="red", fg="white")
exit.place(x=10, y=220)

root.mainloop()
