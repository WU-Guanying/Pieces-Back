import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender_email, receiver_email, subject, body, smtp_server, port, password):
    # 创建邮件消息
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # 邮件内容
    msg.attach(MIMEText(body, 'plain'))

    # 连接到 SMTP 服务器并发送邮件
    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()  # 启用加密传输
            server.login(sender_email, password)  # 登录 SMTP 服务器
            server.send_message(msg)  # 发送邮件
            print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

# # 示例调用
# send_email(
#     sender_email="your-email@example.com",
#     receiver_email="recipient@example.com",
#     subject="Test Email",
#     body="This is a test email sent using smtplib.",
#     smtp_server="smtp.gmail.com",
#     port=587,
#     password="your-email-password"
# )
