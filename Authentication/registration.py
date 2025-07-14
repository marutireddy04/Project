import tkinter as tk
from tkinter import messagebox as ms
import sqlite3
from PIL import Image, ImageTk
import re
import random

window = tk.Tk()
window.geometry("700x700")
window.title("REGISTRATION FORM")
window.configure(background="grey")

# Variable Declarations (Strings for mobile and age to handle empty input properly)
Fullname = tk.StringVar()
address = tk.StringVar()
username = tk.StringVar()
Email = tk.StringVar()
Phoneno = tk.StringVar()
var = tk.IntVar()
age = tk.StringVar()
password = tk.StringVar()
password1 = tk.StringVar()

# Database Setup
db = sqlite3.connect('evaluation.db')
cursor = db.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS admin_registration
    (Fullname TEXT, address TEXT, username TEXT UNIQUE, Email TEXT,
     Phoneno TEXT, Gender TEXT, age TEXT , password TEXT)""")
db.commit()
db.close()

def password_check(passwd): 
    SpecialSym = ['$', '@', '#', '%'] 
    val = True
    if len(passwd) < 6 or len(passwd) > 20: val = False
    if not any(char.isdigit() for char in passwd): val = False
    if not any(char.isupper() for char in passwd): val = False
    if not any(char.islower() for char in passwd): val = False
    if not any(char in SpecialSym for char in passwd): val = False
    return val

def insert():
    fname = Fullname.get()
    addr = address.get()
    un = username.get()
    email = Email.get()
    mobile = Phoneno.get()
    gender = var.get()
    time = age.get()
    pwd = password.get()
    cnpwd = password1.get()

    email_valid = re.match(r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$', email)

    if fname.isdigit() or fname == "":
        ms.showerror("Error", "Please enter a valid name")
    elif addr == "":
        ms.showerror("Error", "Please Enter Address")
    elif email == "" or not email_valid:
        ms.showerror("Error", "Please Enter valid email")
    elif not mobile.isdigit() or len(mobile) != 10:
        ms.showerror("Error", "Please Enter 10 digit mobile number")
    elif not time.isdigit() or int(time) <= 0 or int(time) > 100:
        ms.showerror("Error", "Please Enter valid age")
    elif pwd != cnpwd:
        ms.showerror("Error", "Passwords must match")
    elif not password_check(pwd):
        ms.showerror("Error", "Password must contain at least 1 uppercase, 1 lowercase, 1 digit and 1 special character")
    else:
        try:
            with sqlite3.connect('evaluation.db') as db:
                cursor = db.cursor()
                cursor.execute("""INSERT INTO admin_registration
                    (Fullname, address, username, Email, Phoneno, Gender, age , password)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (fname, addr, un, email, mobile, "Male" if gender == 1 else "Female", time, pwd))
                db.commit()
                ms.showinfo("Success", "Account Created Successfully!")
                window.destroy()
                import subprocess
                subprocess.call(["python", "login.py"])
        except sqlite3.IntegrityError:
            ms.showerror("Error", "Username already exists")

# GUI Setup
bg_image = Image.open('C:\\Users\\shital\\Desktop\\python\\Images\\re1.jpg').resize((700, 700))
background = ImageTk.PhotoImage(bg_image)
tk.Label(window, image=background).place(x=0, y=0)

tk.Label(window, text="Registration Form", font=("Times new roman", 30, "bold"), bg="#192841", fg="white").place(x=190, y=50)

labels = ["Full Name", "Address", "E-mail", "Phone number", "Gender", "Age", "User Name", "Password", "Confirm Password"]
y_coords = [150, 200, 250, 300, 350, 400, 450, 500, 550]
vars = [Fullname, address, Email, Phoneno, None, age, username, password, password1]

for i, label in enumerate(labels):
    tk.Label(window, text=label + ":", width=15, font=("Times new roman", 15, "bold"), bg="snow").place(x=100, y=y_coords[i])
    if label == "Gender":
        tk.Radiobutton(window, text="Male", bg="snow", font=("bold", 13), variable=var, value=1).place(x=330, y=350)
        tk.Radiobutton(window, text="Female", bg="snow", font=("bold", 13), variable=var, value=2).place(x=430, y=350)
    else:
        show_char = "*" if "Password" in label else ""
        tk.Entry(window, textvar=vars[i], font=('', 15), show=show_char).place(x=330, y=y_coords[i])

tk.Button(window, text="Register", bg="#192841", font=("", 20), fg="white", width=10, command=insert).place(x=250, y=620)

window.mainloop()
