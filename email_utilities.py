import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.message import EmailMessage
import ssl

EMAIL_SENDER = "bancosegurotb@gmail.com"
PASS_APP = "sssi dexj elyg guqu"

def send_security_alert(user_email):
    subject = "Suspicious Login Attempt Detected"
    body = """
        Dear User,

        We noticed a suspicious login attempt to your account. The keystroke data from the recent login does not match the pattern we have on file.

        If you did not initiate this login, please take immediate action to secure your account.

        Thank you,
        Your Security Team
        """
        
    em = EmailMessage()
    em['From'] = EMAIL_SENDER
    em['To'] = user_email
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, PASS_APP)
            smtp.sendmail(EMAIL_SENDER, user_email, em.as_string())
            log_event(f"Security email sent to {user_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")
