import time
# Security Imports
from security import generate_key, decrypt_data, verify_password, hash_password, encrypt_data, encrypt_data_no_encode
import subprocess

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

def log_event(event):
    """Log an event with a timestamp."""
    # Encrypt email and username using the same salt
    log_file = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {event}\n"
    encryption_key = load_aes_key()
    encrypted_log = encrypt_data(log_file, encryption_key)
    with open('system.log', 'a') as log_file:
        log_file.write(encrypted_log)

def calculate_accuracy(total_matches, total_attempts):
    """Calcula a acurácia baseada em correspondências e tentativas."""
    if total_attempts == 0:
        return 0.0
    return (total_matches / total_attempts) * 100
