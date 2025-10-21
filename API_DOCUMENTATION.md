# API Documentation

## `send_email`

# send_email Function

## Description

The `send_email` function sends an email with a specified subject and body to a given recipient. It uses the SMTP protocol to send an email from a sender's address to a recipient's address. The sender's email address and password are optional parameters. If not provided, default values will be used.

## Parameters

- `subject` (_str_): The subject of the email.
- `body` (_str_): The body text of the email.
- `to_address` (_str_): The recipient's email address.
- `from_address` (_str, optional_): The sender's email address. Defaults to 'your-email'.
- `password` (_str, optional_): The sender's email password. Defaults to 'your-password'.

## Returns

The function does not return any value.

## Exceptions

- Raises `smtplib.SMTPException` if an error occurs during the sending process.

## Example

```python
send_email('Hello', 'This is a test email', 'recipient@example.com', 'sender@example.com', 'password123')
```

In the above example, an email with the subject 'Hello' and body 'This is a test email' is sent from 'sender@example.com' to 'recipient@example.com'.

**Best-Practices Review:**
The docstring is almost perfect, but it could be improved by providing more information about the default values of the optional parameters. Here is a revised version:

Function: send_email
Docstring:
"""
Sends an email with a given subject and body to a specified recipient.

This function uses the SMTP protocol to send an email from a specified sender's address to a recipient's address. The sender's email address and password are optional parameters. If not provided, they will default to 'your-email' and 'your-password' respectively.

Parameters:
subject (str): The subject of the email.
body (str): The body text of the email.
to_address (str): The recipient's email address.
from_address (str, optional): The sender's email address. If not provided, defaults to 'your-email'.
password (str, optional): The sender's email password. If not provided, defaults to 'your-password'.

Returns:
None

Raises:
smtplib.SMTPException: If an error occurs during the sending process.
"""

---
