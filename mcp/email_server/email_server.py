from fastmcp import FastMCP
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv("./.env")

mcp = FastMCP("gmail-server")

SENDER_EMAIL_ADDRESS = os.getenv("SENDER_EMAIL_ADDRESS")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
print(SENDER_EMAIL_ADDRESS, SENDER_PASSWORD)

@mcp.tool()
def send_email(strToEmailAddress: str, strSubject:str, strBody:str = "Sent from Hackweek demo server") -> int:
    """Sends an email via Gmail to the given email address with the specified subject and content

    Arguments:
        strToEmailAddress (str): Mandatory parameter. This is the recipient's email address.
        strBody(str): Mandatory parameter. This is the email's body. Supports only text. 
        strSubject(str): Optional parameter. If nothing is specified, uses "Sent from Hackweek demo server" as default value.

    Returns:
        Integer value.
        0: Indicates email is sent successfully
        -1: Indicates something went wrong and email delivery failed. 
    """

    msg = MIMEText(strBody)
    msg["Subject"] = strSubject
    msg["From"] = SENDER_EMAIL_ADDRESS
    msg["To"] = strToEmailAddress

    try:
        # Connect to the SMTP server (e.g., Gmail's SMTP server)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server: # Use 587 for TLS
            server.login(SENDER_EMAIL_ADDRESS, SENDER_PASSWORD)
            server.send_message(msg)
        returnValue = 0
    except Exception as e:
        print(f"Error sending email: {e}")
        returnValue = -1

    return returnValue

if __name__ == "__main__":
    mcp.run(transport="sse", #"streamable-http",
            host="127.0.0.1",
            port=4202,
            log_level="debug")
    # print(os.getcwd())