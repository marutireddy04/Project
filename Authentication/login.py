import tkinter as tk
from tkinter import messagebox as ms
import sqlite3
import subprocess
import sys

window = tk.Tk()
window.geometry("500x400")
window.title("Login Page")
window.configure(bg="lightblue")

username = tk.StringVar()
password = tk.StringVar()

def login_user():
    user = username.get().strip()
    pwd = password.get().strip()

    if not user or not pwd:
        ms.showwarning("Warning", "Please enter username and password")
        return

    try:
        with sqlite3.connect("evaluation.db") as db:
            cursor = db.cursor()
            cursor.execute("SELECT * FROM admin_registration WHERE username = ? AND password = ?", (user, pwd))
            result = cursor.fetchone()
            if result:
                ms.showinfo("Success", "Login successful!")
                window.destroy()
                # Use Popen to launch GUI_Master_old.py asynchronously
                subprocess.Popen([sys.executable, "GUI\GUI_Master_old.py"])
            else:
                ms.showerror("Error", "Invalid username or password")
    except Exception as e:
        ms.showerror("Error", f"An error occurred:\n{e}")

tk.Label(window, text="Login", font=("Arial", 30, "bold"), bg="lightblue").pack(pady=20)

tk.Label(window, text="Username", font=("Arial", 15), bg="lightblue").pack(pady=5)
username_entry = tk.Entry(window, textvariable=username, font=("Arial", 15))
username_entry.pack(pady=5)
username_entry.focus_set()

tk.Label(window, text="Password", font=("Arial", 15), bg="lightblue").pack(pady=5)
tk.Entry(window, textvariable=password, show="*", font=("Arial", 15)).pack(pady=5)

tk.Button(window, text="Login", font=("Arial", 15), bg="#192841", fg="white", width=15, command=login_user).pack(pady=20)

window.mainloop()
