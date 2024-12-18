# Standard Library Imports
import sqlite3
import time
import os
import smtplib
import string
import ssl
import threading
import logging
from math import sqrt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.message import EmailMessage
import io
import subprocess
# GUI (Tkinter) Related Imports
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox, Frame, simpledialog, Toplevel, Text
from PIL import Image, ImageTk

# Data Handling and ML Libraries
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from io import BytesIO
from utils import log_event

# Security Imports
from security import generate_key, decrypt_data, verify_password, hash_password, encrypt_data, encrypt_data_no_encode

# API and Networking
import requests

# Image Processing
import cv2
import face_recognition

from email_utilities import send_security_alert
from keystroke_utilities import train_model, process_sqlite_data, predict_user_model, compute_keystroke_features, calculate_mean_and_std


# Global idle timer
last_activity_time = time.time()
is_idle = False
is_logged = False
idle_timeout = 10  

current_user = None

# Database setup
db_path = "users.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
security_flag = None

# Create tables for users and keystroke data
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password BLOB NOT NULL,
    email TEXT NOT NULL,
    face_embedding BLOB
)
""")
conn.commit()
#u_default = "admin"
#p_crypted_admin = "admin"
#u_email = "email"
#cursor.execute("""INSERT INTO users(username, password, email)
#                SELECT * FROM (SELECT ?, ?, ?) AS tmp
#                WHERE NOT EXISTS (
#                SELECT username FROM users WHERE username = ?
#                ) LIMIT 1;
#            """, (u_default, p_crypted_admin, u_email, u_default))
 

cursor.execute("""
CREATE TABLE IF NOT EXISTS keystrokes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    ht_mean REAL, ht_std_dev REAL,
    ppt_mean REAL, ppt_std_dev REAL,
    rrt_mean REAL, rrt_std_dev REAL,
    rpt_mean REAL, rpt_std_dev REAL,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")
conn.commit()

cursor.execute("""
CREATE TABLE IF NOT EXISTS models (
    model_blob BLOB
);

""")
conn.commit()

# Set the correct permissions (only creator can access)
os.chmod(db_path, 0o600)  # This gives read-write permissions only to the owner (creator)

def reset_idle_timer(event=None):
    """Reset the idle timer on any user interaction."""
    global last_activity_time
    last_activity_time = time.time()

def check_idle_time():
    """Check if the user has been idle for too long and prompt for re-authentication."""
    global last_activity_time
    global is_idle
    while True:
        current_time = time.time()
        if current_time - last_activity_time > idle_timeout and is_idle == False and is_logged == True:
            is_idle = True
            log_event(f"Entered IDLE mode.")
            print("User has been idle for 2 minutes. Prompting for authentication...")
            app.after(0, idle_prompt)  # Run on main thread to update GUI
            reset_idle_timer()
        time.sleep(1)

def idle_prompt():
    """Prompt the user for password and facial recognition after idle timeout."""
    # Using a Toplevel window for better control
    prompt = Toplevel()
    prompt.title("Session Locked")
    prompt.geometry("300x150")
    
    # Label to show the idle timeout message
    Label(prompt, text="You've been idle for too long.\nPlease re-authenticate.").pack(pady=20)

    # Add buttons for re-authentication
    def authenticate():
        prompt.destroy()
        authenticated = loginAfterIDLE(current_user)
        if authenticated:
            messagebox.showinfo("Authentication completed", "Welcome back!")
            show_home()
        else:
            messagebox.showerror("Error", "Authentication failed. Logging out...")
            logout()

    Button(prompt, text="Re-authenticate", command=authenticate).pack(pady=10)

    # Handle window close event
    def on_close():
        log_event("Session locked dialog closed by user.")
        logout()

    prompt.protocol("WM_DELETE_WINDOW", on_close)  # Redirect to logout if the user closes the window



def send_security_alert_in_background(user_email):
    thread = threading.Thread(target=send_security_alert, args=(user_email,))
    thread.daemon = True   
    thread.start()

def register_face(username, conn=None):
    """Register the user's face securely using their encrypted email."""
    # Ask for permission to access the camera
    response = messagebox.askquestion(
        "Face Authentication",
        "Do you allow the app to access your camera for facial recognition?"
    )
    if response != 'yes':
        messagebox.showerror("Error", "Camera access denied.")
        return False

    print("Please face the camera to register your face.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera. Check permissions.")
        return False

    registered = False
    retries = 0
    max_retries = 10  # Limit retries to avoid resource drain

    try:
        while not registered and retries < max_retries:
            ret, frame = cap.read()
            if not ret:
                print("Error accessing the webcam. Please try again.")
                retries += 1
                continue

            # Process the captured frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if len(face_encodings) == 1:
                # Encrypt and save the face embedding
                face_embedding = np.array(face_encodings[0], dtype=np.float64).tobytes()

                # Use the provided connection to avoid locking
                if conn is None:
                    conn = sqlite3.connect(db_path, timeout=10)

                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE users SET face_embedding=? WHERE username=?',
                    (sqlite3.Binary(face_embedding), username)
                )
                conn.commit()

                print("Face registered successfully!")
                log_event(f"Face registered for user {username}")
                registered = True
            else:
                print("Ensure only your face is visible and retry.")
                retries += 1

            cv2.imshow("Register Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if not registered:
            print("Face registration failed. Please try again.")
            log_event(f"Registration of Face failed")
            messagebox.showerror("Error", "Face registration failed.")
            return False
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return True


def authenticate_face(username):
    """Authenticate the user's face during login with secure handling."""
    # Ask for permission to access the camera
    response = messagebox.askquestion("Face Authentication", "Do you allow the app to access your camera for facial recognition?")
    if response != 'yes':
        messagebox.showerror("Error", "Camera access denied.")
        return None

    print("Authenticating face... Please ensure your face is clearly visible.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera. Check permissions.")
        return None

    start_time = time.time()
    authenticated = False
    retries = 0
    max_retries = 10  # Limit retries for better resource management

    try:
        while not authenticated and retries < max_retries:
            elapsed_time = time.time() - start_time
            if elapsed_time > 8:
                print("Authentication timed out. No face recognized within 8 seconds.")
                cap.release()
                cv2.destroyAllWindows()
                return None

            ret, frame = cap.read()
            if not ret:
                print("Error accessing the webcam.")
                retries += 1
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if len(face_encodings) == 1:
                # Retrieve and decrypt the stored face embedding
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                cursor.execute('SELECT face_embedding, password FROM users WHERE username = ?', (username,))
                result = cursor.fetchone()
                conn.close()

                if result and result[0]:
                    stored_embedding_blob = result[0]
                    stored_embedding = np.frombuffer(stored_embedding_blob, dtype=np.float64)

                    # Compare the face embeddings using Euclidean distance
                    distance = np.linalg.norm(stored_embedding - face_encodings[0])
                    if distance < 0.6:  # Threshold for face matching
                        authenticated = True
                        print(f"Authentication successful for user: {username}")
                        log_event(f"Authentication successfull for user {username}")
                        cap.release()
                        cv2.destroyAllWindows()
                        return username  # Successfully authenticated, return the user's email
                else:
                    print("No face data found for this user.")
                    messagebox.showerror("Error", "No face data registered for this user.")

            else:
                print("Face not recognized. Please adjust your position.")

            cv2.imshow("Authenticate Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if not authenticated:
            print("Authentication failed. Please try again.")
            log_event(f"Authentication failed for user {username}")
            messagebox.showerror("Error", "Authentication failed. Please try again.")
            return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Separate keystroke data for password and confirm password
password_keystrokes = {"press_times": [], "release_times": []}
confirm_password_keystrokes = {"press_times": [], "release_times": []}


def reset_keystroke_data():
    """Reset keystroke data after it's processed."""
    global password_keystrokes, confirm_password_keystrokes
    password_keystrokes["press_times"] = []
    password_keystrokes["release_times"] = []
    confirm_password_keystrokes["press_times"] = []
    confirm_password_keystrokes["release_times"] = []


def on_key_press_password(event):
    """Record the timestamp when a key is pressed for the password field."""
    # Skip modifiers (Shift, Control, Alt) and Backspace/Delete
    if event.keysym != "Tab" and event.keysym not in ["Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Caps_Lock", "BackSpace", "Delete"]:
        password_keystrokes["press_times"].append(time.time())
        print(password_keystrokes)

def on_key_release_password(event):
    """Record the timestamp when a key is released for the password field."""
    # Skip modifiers (Shift, Control, Alt) and Backspace/Delete
    if event.keysym != "Tab" and event.keysym not in ["Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Caps_Lock", "BackSpace", "Delete"]:
        password_keystrokes["release_times"].append(time.time())
        print(password_keystrokes)

def on_key_press_confirm_password(event):
    """Record the timestamp when a key is pressed for the confirm password field."""
    # Skip modifiers (Shift, Control, Alt) and Backspace/Delete
    if event.keysym != "Tab" and event.keysym not in ["Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Caps_Lock", "BackSpace", "Delete"]:
        confirm_password_keystrokes["press_times"].append(time.time())
        print(confirm_password_keystrokes)

def on_key_release_confirm_password(event):
    """Record the timestamp when a key is released for the confirm password field."""
    # Skip modifiers (Shift, Control, Alt) and Backspace/Delete
    if event.keysym != "Tab" and event.keysym not in ["Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Caps_Lock", "BackSpace", "Delete"]:
        confirm_password_keystrokes["release_times"].append(time.time())
        print(confirm_password_keystrokes)

def load_aes_key(host='andre@172.17.0.2', remote_path='/aes_key.key'):
    try:
        # Use ssh and cat to fetch the file content into memory
        result = subprocess.run(
            ['ssh', host, f'cat {remote_path}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        # Return the content of the key as bytes
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error fetching the key: {e.stderr.decode()}")
        return None

def ask_terms_and_conditions():
    """Ask the user to accept the terms and conditions."""
    response = messagebox.askquestion(
        "Terms and Conditions",
        "Do you accept the Terms and Conditions and the Privacy Policy to proceed with registration?\n\n"
        "Terms and Conditions:\n- You must provide accurate information.\n"
        "- You agree not to use the application for illegal activities.\n\n"
        "Privacy Policy:\n- Your data will be securely stored and not shared without your consent.\n"
        "- You can request data deletion at any time."
    )
    return response == 'yes'

def register_user():
    """Register a new user with improved security."""

    user_accepted = ask_terms_and_conditions()
    if not user_accepted:
        messagebox.showinfo("Info", "You must accept the Terms and Conditions to proceed.")
        show_login()
        return
    

    username = reg_username.get().strip()
    password = reg_password.get().strip()
    confirm_password = reg_password_confirm.get().strip()
    email = reg_email.get().strip()

    # Input validation
    if not username or not password or not confirm_password or not email:
        messagebox.showerror("Error", "All fields are required!")
        return
    
    if not username.isdigit():
        messagebox.showerror("Error", "Username must only contain numbers!")
        return

    if password != confirm_password:
        messagebox.showerror("Error", "Passwords do not match!")
        reset_keystroke_data()
        return

    if len(password) < 8:  # Password strength check
        messagebox.showerror("Error", "Password must be at least 8 characters long!")
        return

    # First password entry keystroke dynamics
    reg_password_entry.bind("<KeyPress>", on_key_press_password)
    reg_password_entry.bind("<KeyRelease>", on_key_release_password)
    features_1 = compute_keystroke_features(password_keystrokes)
    if any(v == 0 for v in features_1.values()):
        messagebox.showerror("Error", "Invalid keystroke data for the first input!")
        reset_keystroke_data()
        return

    # Confirm password keystroke dynamics
    reg_password_confirm_entry.bind("<KeyPress>", on_key_press_confirm_password)
    reg_password_confirm_entry.bind("<KeyRelease>", on_key_release_confirm_password)
    features_2 = compute_keystroke_features(confirm_password_keystrokes)
    if any(v == 0 for v in features_2.values()):
        messagebox.showerror("Error", "Invalid keystroke data for the second input!")
        reset_keystroke_data()
        return

    try:
        hashed_password = hash_password(password)

        # Encrypt email and username using the same salt
        encryption_key = load_aes_key()
        encrypted_email = encrypt_data(email, encryption_key)

        # Insert user details into the database (transactional approach)
        with sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()

            # Start a transaction
            cursor.execute(
                "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                (username, hashed_password, encrypted_email)
            )
            user_id = cursor.lastrowid

            # Insert keystroke data for both entries
            cursor.execute("""
                INSERT INTO keystrokes (user_id, ht_mean, ht_std_dev, ppt_mean, ppt_std_dev, rrt_mean, rrt_std_dev, rpt_mean, rpt_std_dev)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, *features_1.values()))
            cursor.execute("""
                INSERT INTO keystrokes (user_id, ht_mean, ht_std_dev, ppt_mean, ppt_std_dev, rrt_mean, rrt_std_dev, rpt_mean, rpt_std_dev)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, *features_2.values()))

            # Attempt to register face
            print(f"Registering face for user {username}.")
            face_registration_success = register_face(username, conn)

            if not face_registration_success:
                # Rollback if face registration fails
                print("Face registration failed, rolling back user registration.")
                log_event(f"Face registration failed, rolling back user registration")
                conn.rollback()
                messagebox.showerror("Error", "Face registration failed. User registration aborted.")
                return

            # Commit the transaction if everything succeeded
            conn.commit()

        # Process and retrain the model after a successful registration
        training_data = process_sqlite_data()
        train_model(training_data, user_id, conn)

        messagebox.showinfo("Success", "Registration successful!")
        log_event(f"Registration successful for user {username}")
        reset_keystroke_data()
        show_login()

    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Username or email already exists!")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        reset_keystroke_data()


def login_user():
    """Handle user login with password, facial recognition, and keystroke verification."""
    global current_user, is_logged, security_flag
    username = login_username.get().strip()
    password = login_password.get().strip()

    if not username or not password:
        messagebox.showerror("Error", "Both fields are required!")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Use encrypted username for the query
            cursor.execute("SELECT * FROM users WHERE username=?", (username,))
            user = cursor.fetchone()
    except sqlite3.IntegrityError:
        print("Error accessing the database")

    if user:
        encryption_key = load_aes_key()
        # Step 1: Authenticate face before checking password
        print("Face authentication started...")
        decrypted_email = decrypt_data(user[3], encryption_key)  # Decrypt email
        if not authenticate_face(username):  # Assuming face is registered with decrypted email
            messagebox.showerror("Error", "Face authentication failed.")
            log_event(f"Error authenticating face for user {username}")
            return

        # Step 2: Check password with bcrypt hash
        hashed_password = user[2]   
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode()  # Convert to bytes
        if verify_password(password, hashed_password):  # Use bcrypt to verify password
            log_event(f"Password match for user {username}")
            print("Password matched. Now checking keystroke data...")

            # Get the user email and user_id
            user_email = decrypted_email
            user_id = user[0]
            current_user = username

            # First password entry keystroke dynamics
            login_password_entry.bind("<KeyPress>", on_key_press_password)
            login_password_entry.bind("<KeyRelease>", on_key_release_password)
            features_1 = compute_keystroke_features(password_keystrokes)
            if any(v == 0 for v in features_1.values()):
                messagebox.showerror("Error", "Invalid keystroke data for the first input!")
                reset_keystroke_data()
                return

            matched = predict_user_model(features_1, conn)
            if matched == user_id:
                security_flag = False
                log_event(f"Keystroke match for user {username}")
                print("There was a match")
            # Optionally, compare features for consistency here (e.g., keystroke analysis)

                try:
                    # Insert keystroke data
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(""" 
                            INSERT INTO keystrokes (user_id, ht_mean, ht_std_dev, ppt_mean, ppt_std_dev, rrt_mean, rrt_std_dev, rpt_mean, rpt_std_dev)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (user_id, *features_1.values()))
                        conn.commit()
                    # Retrain the model
                    training_data = process_sqlite_data()
                    train_model(training_data, user_id, conn)
                except sqlite3.IntegrityError:
                    messagebox.showerror("Error", "Failed to insert keystroke data!")

                reset_keystroke_data()
                is_logged = True
                log_event(f"Login successful for user {username}")
                show_home(username)  # Proceed to the home screen if login is successful
            else:
                security_flag = True
                send_security_alert_in_background(user_email)
                print("keystroke did not match, not saving and turning on security.")
                log_event(f"Failed keystroke authentication for user {username}")
                reset_keystroke_data()
                is_logged = True
                show_home(username)  # Proceed to the home screen if login is successful
        else:
            messagebox.showerror("Error", "Incorrect password.")
            log_event(f"Wrong password for user {username}")
    else:
        messagebox.showerror("Error", "Username not found.")
        log_event(f"Username {username} not found")

def loginAfterIDLE(username):
    """
    Handle user login after idle with facial recognition, password, and keystroke verification.
    
    Args:
        username (str): The username of the user attempting to log in.
    """
    global is_idle, is_logged, security_flag
    # Fetch the user details from the database using the provided username
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username=?", (username,))
            user = cursor.fetchone()
    except sqlite3.IntegrityError:
        print("Error accessing the database")
        return

    if not user:
        messagebox.showerror("Error", "User not found.")
        return

    # Create a new dialog for re-entering the password
    idle_login_window = Tk()
    idle_login_window.title("Re-Authenticate")
    idle_login_window.geometry("400x200")

    Label(idle_login_window, text="Please re-enter your password:").pack(pady=10)
    
    idle_password = StringVar()
    idle_password_entry = Entry(idle_login_window, textvariable=idle_password, show="*")
    idle_password_entry.pack(pady=5)
    idle_password_entry.focus()

    # Initialize keystroke data for the idle login
    idle_keystrokes = {"press_times": [], "release_times": []}

    def on_key_press_idle(event):
        if event.keysym != "Tab":  # Skip recording for Tab key
            idle_keystrokes["press_times"].append(time.time())

    def on_key_release_idle(event):
        if event.keysym != "Tab":  # Skip recording for Tab key
            idle_keystrokes["release_times"].append(time.time())

    # Attach keystroke listeners
    idle_password_entry.bind("<KeyPress>", on_key_press_idle)
    idle_password_entry.bind("<KeyRelease>", on_key_release_idle)

    def handle_idle_login():
        global security_flag, is_idle, is_logged
        entered_password = idle_password_entry.get()
        hashed_password = user[2]  # Assuming the password hash is at index 2
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode()  # Convert to bytes

        if not entered_password:
            messagebox.showerror("Error", "Password input is required.")
            return

        # Verify password
        if not verify_password(entered_password, hashed_password):  # Use bcrypt to verify password
            log_event(f"Incorrect password for user {username}")
            messagebox.showerror("Error", "Incorrect password.")
            on_close()
            return

        is_idle = False

        # Compute keystroke features after user completes typing
        features = compute_keystroke_features(idle_keystrokes)
        if any(v == 0 for v in features.values()):
            messagebox.showerror("Error", "Invalid keystroke data for the input!")
            return
        user_id = user[0]
        matched = predict_user_model(features)
        if matched == user_id:
            security_flag = False
            log_event(f"Keystroke match for user {username} after IDLE")
            print("There was a match")
            try:
                # Insert keystroke data
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(""" 
                        INSERT INTO keystrokes (user_id, ht_mean, ht_std_dev, ppt_mean, ppt_std_dev, rrt_mean, rrt_std_dev, rpt_mean, rpt_std_dev)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (user_id, *features.values()))
                    conn.commit()
                # Retrain the model
                training_data = process_sqlite_data()
                train_model(training_data, user_id, conn)
            except sqlite3.IntegrityError:
                messagebox.showerror("Error", "Failed to insert keystroke data!")

            reset_keystroke_data()
            is_logged = True
            show_home(username)  # Proceed to the home screen if login is successful
        else:
            security_flag = True
            encryption_key = load_aes_key()
            decrypted_email = decrypt_data(user[3], encryption_key)  # Decrypt email
            send_security_alert_in_background(decrypted_email)
            messagebox.showwarning("Warning", "Keystroke data does not match. An email has been sent to your address.")
            log_event(f"Failed keystroke authentication for user {username} after IDLE")
            idle_login_window.destroy()  # Close the window when authentication fails
            return

        # Successful authentication
        print(f"Login successful for user: {username}")
        log_event(f"Login successful for user {username} after IDLE")
        idle_login_window.destroy()  # Close the window after successful login
        show_home(username)  

    # Login button waits for user's action
    Button(idle_login_window, text="Login", command=handle_idle_login).pack(pady=20)

    # Handle window close event: redirect to logout and destroy the window
    def on_close():
        log_event("Password input window closed by user.")
        idle_login_window.destroy()  # Destroy the window first
        logout()  # Then call logout to handle the redirection

    # Bind the close window event to logout
    idle_login_window.protocol("WM_DELETE_WINDOW", on_close)

    idle_login_window.mainloop()


# Security validation function
def validate_physical_matrix(callback):
    security_prompt = Toplevel(app)
    security_prompt.title("Security Check")
    security_prompt.geometry("300x200")
    Label(security_prompt, text="Enter Physical Matrix", font=font_medium).pack(pady=10)
    matrix_input = Entry(security_prompt, font=font_medium)
    matrix_input.pack(pady=10)
    
    def submit_matrix():
        # Placeholder for validation logic
        if matrix_input.get() == "1234": 
            security_prompt.destroy()
            callback()
        else:
            Label(security_prompt, text="Invalid Matrix!", font=font_small, fg="red").pack()
    
    Button(security_prompt, text="Submit", command=submit_matrix, bg=button_color, fg="white").pack(pady=10)

# Function to create new pages dynamically
def show_page(page_title, input_fields):
    global security_flag
    def render_page():
        if security_flag:
            validate_physical_matrix(render_inputs)
        else:
            render_inputs()
    
    def render_inputs():
        page = Toplevel(app)
        page.title(page_title)
        page.geometry("400x300")
        Label(page, text=page_title, font=font_large).pack(pady=10)
        for field in input_fields:
            Label(page, text=field, font=font_medium).pack(pady=5)
            Entry(page, font=font_small).pack(pady=5)
        Button(page, text="Submit", bg=button_color, fg="white", font=font_medium).pack(pady=20)
    
    render_page()

# Dynamic navigation to banking task pages
def open_view_balance():
    show_page("View Balance", ["Account Number"])

def open_transfer_funds():
    show_page("Transfer Funds", ["From Account", "To Account", "Amount"])

def open_transaction_history():
    show_page("Transaction History", ["Account Number", "Date Range"])


# Functionality for page switching
def show_register():
    # Clear registration fields
    reg_username.set("")
    reg_password.set("")
    reg_password_confirm.set("")
    reg_email.set("")
    reset_keystroke_data()
    
    login_frame.pack_forget()
    home_frame.pack_forget()
    reg_frame.pack()

def show_login():
    # Clear login fields
    login_username.set("")
    login_password.set("")
    reset_keystroke_data()
    reg_frame.pack_forget()
    home_frame.pack_forget()
    login_frame.pack()

def show_home(username):
    global is_idle
    is_idle = False

    def render_home():
        reset_keystroke_data()
        reg_frame.pack_forget()
        login_frame.pack_forget()
        home_label.config(text=f"Welcome {username}!")
        home_frame.pack()

    render_home()

def logout():
    global current_user, is_logged
    reg_username.set("")
    reg_password.set("")
    reg_password_confirm.set("")
    reg_email.set("")
    login_username.set("")
    login_password.set("")
    current_user = None
    is_logged = False
    
    # Destroy any dynamic pages if they are open
    for widget in app.winfo_children():
        if isinstance(widget, Toplevel):  # Close all Toplevel windows
            widget.destroy()

    # Return to the login screen
    show_login()

def validate_numeric_input(username):
    if not username.isdigit():
        messagebox.showerror("Error", "Username must only contain numbers!")
        return False
    return True

# GUI setup
app = Tk()
app.title("Bank App")
app.geometry("800x600")  # Enlarged window size
app.configure(bg="#FFFFFF")  # Bank app-themed background color

# Resize the logo using Pillow
original_logo = Image.open("logo.png")  
resized_logo = original_logo.resize((150, 100))  # (width, height)
logo_img = ImageTk.PhotoImage(resized_logo)

# Add the resized logo
logo_label = Label(app, image=logo_img, bg="#FFFFFF")
logo_label.place(x=10, y=10)  # Positioning the logo at the top left corner

# Updated fonts for larger text
font_large = ("Helvetica", 20, "bold")  # Larger font
font_medium = ("Helvetica", 16)
font_small = ("Helvetica", 14)

# Updated button styles for a polished look
primary_color = "#0d47a1"
button_color = "#1565c0"
button_style = {
    "bg": button_color,
    "fg": "white",
    "font": font_medium,
    "relief": "solid",
    "bd": 2
}

def force_quit():
    try:
        # Stop any background threads here if applicable
        app.destroy()  # Forcefully destroy the app
    except Exception as e:
        print(f"Error during quit: {e}")  # Log errors, if needed
    finally:
        app.quit()  # Ensure the app is closed


# Variables
reg_username = StringVar()
reg_password = StringVar()
reg_password_confirm = StringVar()
reg_email = StringVar()

login_username = StringVar()
login_password = StringVar()

def disable_copy_paste(event):
    return "break"  # Prevent the event from propagating
# Updated Registration Frame
reg_frame = Frame(app, bg="#FFFFFF")
Label(reg_frame, text="Register", font=font_large, bg="#FFFFFF").pack(pady=20)

Label(reg_frame, text="Username (Numbers only):", font=font_medium, bg="#FFFFFF").pack()

# Username Entry with Validation
username_feedback = StringVar()  # Feedback message for invalid input
username_feedback.set("")  # Initialize with no feedback

def validate_numeric_input(event=None):
    """Validate that the username contains only numbers."""
    username = reg_username.get()
    if not username.isdigit():
        username_feedback.set("❌ Username must only contain numbers.")
    else:
        username_feedback.set("")  # Clear feedback if valid

# Entry Field for Username
username_entry = Entry(reg_frame, textvariable=reg_username, font=font_medium)
username_entry.pack(pady=5)
username_entry.bind("<KeyRelease>", validate_numeric_input)

# Label for Feedback
Label(reg_frame, textvariable=username_feedback, font=("Arial", 10), fg="red", bg="#FFFFFF").pack()


Label(reg_frame, text="Password:", font=font_medium, bg="#FFFFFF").pack()
reg_password_entry = Entry(reg_frame, textvariable=reg_password, show="*", font=font_medium)
# Disable copy and paste in the registration password field
reg_password_entry.bind("<Control-c>", disable_copy_paste)
reg_password_entry.bind("<Control-v>", disable_copy_paste)
reg_password_entry.bind("<Button-3>", disable_copy_paste)  # Right-click disable
reg_password_entry.pack(pady=5)
reg_password_entry.bind("<KeyPress>", on_key_press_password)
reg_password_entry.bind("<KeyRelease>", on_key_release_password)
Label(reg_frame, text="Confirm Password:", font=font_medium, bg="#FFFFFF").pack()
reg_password_confirm_entry = Entry(reg_frame, textvariable=reg_password_confirm, show="*", font=font_medium)
# Disable copy and paste in the confirmation password field
reg_password_confirm_entry.bind("<Control-c>", disable_copy_paste)
reg_password_confirm_entry.bind("<Control-v>", disable_copy_paste)
reg_password_confirm_entry.bind("<Button-3>", disable_copy_paste)  # Right-click disable
reg_password_confirm_entry.pack(pady=5)
reg_password_confirm_entry.bind("<KeyPress>", on_key_press_confirm_password)
reg_password_confirm_entry.bind("<KeyRelease>", on_key_release_confirm_password)
Label(reg_frame, text="Email:", font=font_medium, bg="#FFFFFF").pack()
Entry(reg_frame, textvariable=reg_email, font=font_medium).pack(pady=5)
Button(reg_frame, text="Register", command=register_user, **button_style).pack(pady=15)
Button(reg_frame, text="Go to Login", command=show_login, **button_style).pack()

# Updated Login Frame
login_frame = Frame(app, bg="#FFFFFF")
Label(login_frame, text="Login", font=font_large, bg="#FFFFFF").pack(pady=20)
Label(login_frame, text="Username:", font=font_medium, bg="#FFFFFF").pack()
Entry(login_frame, textvariable=login_username, font=font_medium).pack(pady=5)
Label(login_frame, text="Password:", font=font_medium, bg="#FFFFFF").pack()
login_password_entry = Entry(login_frame, textvariable=login_password, show="*", font=font_medium)
# Disable copy and paste in the login password field
login_password_entry.bind("<Control-c>", disable_copy_paste)
login_password_entry.bind("<Control-v>", disable_copy_paste)
login_password_entry.bind("<Button-3>", disable_copy_paste)  # Right-click disable
login_password_entry.pack(pady=5)
login_password_entry.bind("<KeyPress>", on_key_press_password)
login_password_entry.bind("<KeyRelease>", on_key_release_password)
Button(login_frame, text="Login", command=login_user, **button_style).pack(pady=15)
Button(login_frame, text="Go to Register", command=show_register, **button_style).pack(pady=5)
Button(login_frame, text="Exit", command=force_quit, bg="#e53935", fg="white", font=font_medium, relief="solid", bd=2).pack(pady=10)

# Updated Home Frame
home_frame = Frame(app, bg="#FFFFFF")
home_label = Label(home_frame, text="", font=font_large, bg="#FFFFFF")
home_label.pack(pady=20)
Button(home_frame, text="View Balance", command=open_view_balance, **button_style).pack(pady=10)
Button(home_frame, text="Transfer Funds", command=open_transfer_funds, **button_style).pack(pady=10)
Button(home_frame, text="Transaction History", command=open_transaction_history, **button_style).pack(pady=10)
Button(home_frame, text="Logout", command=logout, bg="#e53935", fg="white", font=font_medium, relief="solid", bd=2).pack(pady=10)

# Bind events to reset the idle timer
app.bind_all("<Any-KeyPress>", reset_idle_timer)
app.bind_all("<Any-Button>", reset_idle_timer)
app.bind_all("<Motion>", reset_idle_timer)

idle_thread = threading.Thread(target=check_idle_time, daemon=True)
idle_thread.start()

# Start with Login Frame
show_login()

app.mainloop()
