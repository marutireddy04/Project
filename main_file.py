import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import sqlite3
#import tfModel_test as tf_test  # Assuming this is not required for the current functionality

global fn
fn = ""

def login():
    from subprocess import call
    call(["python", "C:\\Users\\shital\\Desktop\\python\\Authentication\\login.py"])

def register():
    from subprocess import call
    call(["python", "C:\\Users\\shital\\Desktop\\python\\Authentication\\registration.py"])

def window():
    root.destroy()

root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")  # Uncomment if you want to specify window size

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Automated Lung Cancer Detection Via CNN")

# Background image
image2 = Image.open('C:\\Users\\shital\\Desktop\\python\\Images\\c3.jpg')  # Adjust path as needed
image2 = image2.resize((w, h), Image.Resampling.LANCZOS)
background_image = ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image, bd=5)
background_label.image = background_image
background_label.place(x=0, y=0)

# Frame for login/register buttons
frame_alpr = tk.LabelFrame(root, text=" ", width=820, height=900, bd=5, font=('times', 14, ' bold '), bg="#271983")
frame_alpr.grid(row=0, column=0)
frame_alpr.place(x=900, y=0)

# Application title label
lbl = tk.Label(root, text="Automated Lung Cancer Detection Via CNN", font=('Elephant', 35,' bold '), width=50, bg="brown", fg="white")
lbl.place(x=0, y=0)

# Motivational text labels within the frame
lbl = tk.Label(frame_alpr, text='Cancer is a part of our life ', font=('Lucida Calligraphy', 15,' bold '), bg="#271983", fg="white")
lbl.place(x=170, y=100)

lbl = tk.Label(frame_alpr, text="but it's not our whole life", font=('Lucida Calligraphy', 15,' bold '), bg="#271983", fg="white")
lbl.place(x=210, y=140)

# Login and register buttons
button1 = tk.Button(frame_alpr, text=" SIGN UP ", command=register, width=15, height=1, font=('times', 15, ' bold '), bg="#3BB9FF", fg="white")
button1.place(x=200, y=350)

button2 = tk.Button(frame_alpr, text="LOGIN", command=login, width=15, height=1, font=('times', 15, ' bold '), bg="#3BB9FF", fg="white")
button2.place(x=200, y=450)

root.mainloop()