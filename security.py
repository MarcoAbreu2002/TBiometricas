from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import bcrypt
import os
import base64

# Generate encryption key
def generate_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

# Encrypt data
def encrypt_data(plaintext, key):
    iv = os.urandom(16)  # Generate Initialization Vector
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
    return base64.b64encode(iv + ciphertext).decode()

def encrypt_data_no_encode(plaintext, key):
    iv = os.urandom(16)  # Generate Initialization Vector
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # Ensure plaintext is bytes before encrypting
    if isinstance(plaintext, str):
        plaintext = plaintext.encode('utf-8')  # Convert string to bytes
    
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()  # No need to call .encode()
    return base64.b64encode(iv + ciphertext).decode()


# Decrypt data
def decrypt_data(ciphertext, key):
    raw_data = base64.b64decode(ciphertext.encode())
    iv = raw_data[:16]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(raw_data[16:]) + decryptor.finalize()
    return plaintext.decode()

# Hash a password
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed

# Verify a password
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode(), hashed_password)
