import logging
from utils import log_event
from math import sqrt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from io import BytesIO
import sqlite3
db_path = "users.db"

logging.basicConfig(level=logging.INFO)

def train_model(training_data, user_id, conn=None):
    try:
        logging.info("Starting model training for user_id: %s", user_id)
        
        required_features = [
            'ht_mean', 'ht_std_dev', 'ppt_mean', 'ppt_std_dev', 
            'rrt_mean', 'rrt_std_dev', 'rpt_mean', 'rpt_std_dev'
        ]
        if not all(feature in training_data.columns for feature in required_features):
            raise ValueError(f"Training data is missing one or more required features: {required_features}")
        
        X = training_data[required_features]
        y = training_data['user_id']
        
        logging.info("Feature matrix and target extracted successfully.")

        rf_model = RandomForestClassifier()
        rf_model.fit(X, y)
        logging.info("Random Forest model trained successfully.")

        model_stream = BytesIO()
        joblib.dump(rf_model, model_stream)
        model_stream.seek(0)
        serialized_model = model_stream.read()
        logging.info("Model serialized successfully.")

        if conn is None:
            if db_path is None:
                raise ValueError("Either a valid SQLite connection or a database path must be provided.")
            conn = sqlite3.connect(db_path)
            logging.info("Database connection opened.")
        
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM models
            """)
            
            exists = cursor.fetchone()[0] > 0
            
            if exists:
                cursor.execute("""
                    UPDATE models
                    SET model_blob = ?
                """, (serialized_model,))
                logging.info("Existing model updated in the database successfully.")
            else:
                cursor.execute("""
                    INSERT INTO models (model_blob)
                    VALUES (?)
                """, (serialized_model,))
                logging.info("New model inserted into the database successfully.")
            
            conn.commit()

        log_event(f"ML model trained")

    except Exception as e:
        if conn:
            conn.rollback()
        logging.error("Error during model training or storage: %s", str(e))
        raise e
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")


def process_sqlite_data():
# Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch data from the keystrokes table
    cursor.execute("SELECT * FROM keystrokes")
    rows = cursor.fetchall()

    # Prepare dictionary to store the data
    data = {
        'user_id': [],
        'ht_mean': [],
        'ht_std_dev': [],
        'ppt_mean': [],
        'ppt_std_dev': [],
        'rrt_mean': [],
        'rrt_std_dev': [],
        'rpt_mean': [],
        'rpt_std_dev': [],
    }

    # Process each row from the query result
    for row in rows:
        user_id = row[1]  # user_id is in the second column (index 1)
        ht_mean = row[2]
        ht_std_dev = row[3]
        ppt_mean = row[4]
        ppt_std_dev = row[5]
        rrt_mean = row[6]
        rrt_std_dev = row[7]
        rpt_mean = row[8]
        rpt_std_dev = row[9]

        # Append the calculated values to the data dictionary
        data['user_id'].append(user_id)
        data['ht_mean'].append(ht_mean)
        data['ht_std_dev'].append(ht_std_dev)
        data['ppt_mean'].append(ppt_mean)
        data['ppt_std_dev'].append(ppt_std_dev)
        data['rrt_mean'].append(rrt_mean)
        data['rrt_std_dev'].append(rrt_std_dev)
        data['rpt_mean'].append(rpt_mean)
        data['rpt_std_dev'].append(rpt_std_dev)

    # Convert the dictionary to a DataFrame
    data_df = pd.DataFrame(data)

    # Close the database connection
    conn.close()

    return data_df

def predict_user_model(new_data, conn=None, threshold=0.7):
    """
    Predicts the user input using the stored model in the SQLite database.

    :param new_data: A DataFrame containing the new data for prediction.
    :param conn: An existing SQLite connection object.
    :param threshold: The probability threshold for accepting a prediction.
    :return: Prediction result for the given user_id and new_data, or 0 if below threshold.
    """
    if conn is None:
        if db_path is None:
            raise ValueError("Either a valid SQLite connection or a database path must be provided.")
        conn = sqlite3.connect(db_path)
        logging.info("Database connection opened.")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT model_blob FROM models")
        result = cursor.fetchone()

        if result is None:
            print(f"No model found")
            return False

        # Load the model
        serialized_model = result[0]
        model = joblib.load(BytesIO(serialized_model))

        # Prepare input features
        features = ['ht_mean', 'ht_std_dev', 'ppt_mean', 'ppt_std_dev', 'rrt_mean', 'rrt_std_dev', 'rpt_mean', 'rpt_std_dev']
        X = pd.DataFrame([new_data], columns=features)
        print("Prepared features for prediction:", X)

        # Make prediction
        prediction = model.predict(X)
        probabilities = model.predict_proba(X)
        print("Prediction result:", prediction[0])
        print("Prediction probabilities:", probabilities)

        # Check if the highest probability exceeds the threshold
        max_prob = max(probabilities[0])
        if max_prob >= threshold:
            return prediction[0]
        else:
            return 0
    except Exception as e:
        print(f"Error during prediction: {e}")
        return False
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")




def compute_keystroke_features(keystroke_data):
    press_times = keystroke_data["press_times"]
    release_times = keystroke_data["release_times"]

    ht = [release_times[i] - press_times[i] for i in range(len(press_times))]
    ppt = [press_times[i + 1] - press_times[i] for i in range(len(press_times) - 1)]
    rrt = [release_times[i + 1] - release_times[i] for i in range(len(release_times) - 1)]
    rpt = [press_times[i + 1] - release_times[i] for i in range(len(release_times) - 1)]

    return {
        "ht_mean": calculate_mean_and_std(ht)[0],
        "ht_std_dev": calculate_mean_and_std(ht)[1],
        "ppt_mean": calculate_mean_and_std(ppt)[0],
        "ppt_std_dev": calculate_mean_and_std(ppt)[1],
        "rrt_mean": calculate_mean_and_std(rrt)[0],
        "rrt_std_dev": calculate_mean_and_std(rrt)[1],
        "rpt_mean": calculate_mean_and_std(rpt)[0],
        "rpt_std_dev": calculate_mean_and_std(rpt)[1],
    }

def calculate_mean_and_std(feature_list):
    if not feature_list:
        return 0, 0
    mean = sum(feature_list) / len(feature_list)
    squared_diffs = [(x - mean) ** 2 for x in feature_list]
    variance = sum(squared_diffs) / (len(feature_list) - 1 if len(feature_list) > 1 else 1)
    std_dev = sqrt(variance)
    return mean, std_dev

