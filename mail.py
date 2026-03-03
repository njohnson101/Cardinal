import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from O365 import Account, FileSystemTokenBackend
from openai import OpenAI


# ===============================================================
# AZURE APP REGISTRATION – HOW TO GET CLIENT ID & SECRET
# ===============================================================
"""
HIGH-LEVEL STEPS (do this once):

1. Go to the Azure Portal
   - Open a browser and log in to `https://portal.azure.com`
   - In the search bar at the top, type "App registrations" and open it.
   - Click "New registration".

2. Register a new application
   - Name: anything you like, e.g. "Personal Outlook Inbox Manager".
   - Supported account types:
       - For personal use with one Microsoft account, "Accounts in any organizational directory and personal Microsoft accounts" is often the easiest.
   - Redirect URI:
       - Type: "Web"
       - Value: e.g. `http://localhost:8000/callback`  (you can choose any localhost URL, but it MUST match what you put in your .env file)
   - Click "Register".

3. Collect your CLIENT ID and TENANT ID
   - After registration, you land on the app's "Overview" page.
   - Copy:
       - "Application (client) ID" -> this is your CLIENT_ID
       - "Directory (tenant) ID" -> this is your TENANT_ID

4. Create a CLIENT SECRET
   - In the left sidebar, click "Certificates & secrets".
   - Under "Client secrets", click "New client secret".
   - Description: e.g. "Inbox manager secret"
   - Expires: choose an appropriate duration.
   - Click "Add".
   - Copy the "Value" shown once – this is your CLIENT_SECRET.
     (You will NOT be able to see it again after you leave the page.)

5. Configure API Permissions for Mail
   - In the left sidebar, click "API permissions".
   - Click "Add a permission" -> "Microsoft Graph" -> "Delegated permissions".
   - Search for and add at least:
       - `Mail.ReadWrite`   (read and modify mail, including moving messages and marking read)
       - `offline_access`   (so your refresh token works and you don't have to log in constantly)
     Optionally, also:
       - `Mail.Send`        (not strictly required since we ONLY create drafts, but useful if you later want to send)

   - After adding, click "Grant admin consent" if available and you have permission.
     For personal Microsoft accounts, consent may be granted during the first OAuth login.

6. Put values into a .env file
   - In the same folder as this script, create a `.env` file with:

       MS_CLIENT_ID="YOUR_CLIENT_ID_HERE"
       MS_CLIENT_SECRET="YOUR_CLIENT_SECRET_HERE"
       MS_TENANT_ID="YOUR_TENANT_ID_HERE"
       MS_REDIRECT_URI="http://localhost:8000/callback"
       MS_ACCOUNT_EMAIL="your_email@outlook.com"

       OPENROUTER_API_KEY="your_openrouter_api_key_here"

   - Ensure the redirect URI here exactly matches what you configured in Azure.

7. First-time authentication
   - When you run this script the first time, it will open a browser window asking you to sign in and consent.
   - After successful login, O365 will store the token locally (in the `./o365_token` folder by default).
   - Subsequent runs will reuse the stored token and work non-interactively.

"""


# ===============================================================
# ENV & CLIENT INITIALIZATION
# ===============================================================

load_dotenv()

MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
MS_TENANT_ID = os.getenv("MS_TENANT_ID")
MS_REDIRECT_URI = os.getenv("MS_REDIRECT_URI")
MS_ACCOUNT_EMAIL = os.getenv("MS_ACCOUNT_EMAIL")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter client via openai Python package
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Model to use on OpenRouter
OPENROUTER_MODEL = "openai/gpt-4o-mini"


# ===============================================================
# AUTHENTICATION WITH MICROSOFT USING O365
# ===============================================================

def get_o365_account() -> Account:
    """
    Create and authenticate an O365 Account using OAuth.

    This uses delegated permissions (your user identity) and stores
    tokens on disk so subsequent runs do not require re-login.
    """
    if not (MS_CLIENT_ID and MS_CLIENT_SECRET and MS_TENANT_ID and MS_REDIRECT_URI):
        raise RuntimeError("Missing one or more required MS_* environment variables.")

    credentials = (MS_CLIENT_ID, MS_CLIENT_SECRET)

    # Where to store tokens so we don't need to re-authenticate every time.
    token_backend = FileSystemTokenBackend(token_path=".", token_filename="o365_token.txt")

    # auth_flow_type="authorization" uses interactive OAuth (browser popup) once.
    account = Account(
        credentials=credentials,
        auth_flow_type="authorization",
        tenant_id=MS_TENANT_ID,
        token_backend=token_backend,
        redirect_uri=MS_REDIRECT_URI,
    )

    # Scopes: read/write mail + offline_access for refresh tokens
    scopes = [
        "offline_access",
        "https://graph.microsoft.com/Mail.ReadWrite",
    ]

    if not account.is_authenticated:
        print("Authenticating with Microsoft... a browser window should open.")
        result = account.authenticate(scopes=scopes)
        if not result:
            raise RuntimeError("Authentication failed. Please check your Azure app and credentials.")
        print("Authentication successful; tokens saved for future runs.")
    else:
        print("Using existing authentication token.")

    return account


# ===============================================================
# FETCH UNREAD EMAILS
# ===============================================================

def fetch_unread_emails(account: Account, limit: int = 5):
    """
    Fetch up to `limit` unread messages from the Inbox.
    """
    mailbox = account.mailbox()
    inbox = mailbox.inbox_folder()

    # Filter by unread emails only
    query = inbox.new_query().on_attribute("is_read").equals(False)

    print(f"Fetching up to {limit} unread email(s) from Inbox...")
    messages = inbox.get_messages(limit=limit, query=query, order_by="receivedDateTime DESC")
    return list(messages)


# ===============================================================
# AI PROCESSING VIA OPENROUTER
# ===============================================================

def build_llm_prompt_for_email(sender: str, subject: str, body: str) -> str:
    """
    Build the content prompt for the LLM, asking for a strict JSON response.
    """
    instructions = {
        "task": "Classify this email and decide on an action.",
        "output_format": {
            "category": "string - one of: Newsletter, Question, Spam, General, or another short label",
            "action": "string - one of: Archive, Draft Reply, Ignore",
            "draft_text": "string or null - concise, polite reply if Draft Reply is needed, otherwise null",
        },
        "constraints": [
            "Return ONLY a single valid JSON object.",
            "Do NOT include any extra text, explanations, or markdown.",
            "If no reply is necessary, set draft_text to null.",
        ],
        "examples": [
            {
                "category": "Newsletter",
                "action": "Archive",
                "draft_text": None,
            },
            {
                "category": "Question",
                "action": "Draft Reply",
                "draft_text": "Hi [Name],\n\nThanks for reaching out. ...\n\nBest regards,\n[Your Name]",
            },
        ],
    }

    email_data = {
        "sender": sender,
        "subject": subject,
        "body": body,
    }

    prompt = (
        "You are an email triage assistant.\n\n"
        "Instructions JSON:\n"
        f"{json.dumps(instructions, indent=2)}\n\n"
        "Email JSON:\n"
        f"{json.dumps(email_data, indent=2)}\n\n"
        "Return ONLY a single JSON object with keys: category, action, draft_text."
    )
    return prompt


def call_openrouter_for_email(sender: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Call the OpenRouter LLM and parse its JSON response safely.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment.")

    prompt = build_llm_prompt_for_email(sender, subject, body)
    print(f"Calling LLM for email from '{sender}' with subject '{subject}'...")

    response = openrouter_client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise email triage assistant. Respond ONLY with valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        # Helps enforce JSON output if supported by OpenRouter backend for this model
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    print("Raw LLM response:", content)

    # Safely parse JSON (strip whitespace and try to load)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: attempt to extract JSON object by first and last braces
        print("LLM response was not directly parseable JSON. Attempting fallback parsing...")
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            raise RuntimeError(f"Could not find JSON object in LLM response: {content}")
        json_str = content[start : end + 1]
        parsed = json.loads(json_str)

    # Normalize keys; ensure missing ones are handled gracefully
    category = parsed.get("category", "Uncategorized")
    action = parsed.get("action", "Ignore")
    draft_text = parsed.get("draft_text", None)

    return {
        "category": category,
        "action": action,
        "draft_text": draft_text,
    }


# ===============================================================
# EMAIL ACTIONS: ARCHIVE, DRAFT REPLY, MARK READ
# ===============================================================

def move_to_archive(message) -> None:
    """
    Move the given message to the Archive folder.
    """
    mailbox = message.parent
    try:
        archive_folder = mailbox.get_folder(folder_name="Archive")
    except Exception:
        # Some tenants/locales might use a different name; you can adjust here if needed.
        print("Archive folder not found by name 'Archive'. Message will NOT be moved.")
        return

    print(f"Moving message '{message.subject}' to Archive...")
    message.move(archive_folder)


def create_draft_reply(message, draft_text: str) -> None:
    """
    Create a draft reply to the given message WITHOUT sending it.

    CRITICAL GUARDRAIL: This function MUST NEVER send the email.
    It only creates a draft for manual review in Outlook.
    """
    if not draft_text:
        print("Draft text is empty or None; skipping draft creation.")
        return

    mailbox = message.parent

    print(f"Creating draft reply for message '{message.subject}'...")

    # Create a new draft message replying to the original
    # We construct a new message so we have explicit control, and then we don't send it.
    draft = mailbox.new_message()
    draft.subject = f"Re: {message.subject}"

    if message.sender and message.sender.address:
        draft.to.add(message.sender.address)

    # Include original email context if you prefer
    original_body = message.body or ""
    combined_body = (
        f"{draft_text}\n\n"
        "----- Original message -----\n"
        f"From: {message.sender.address if message.sender else 'Unknown'}\n"
        f"Subject: {message.subject}\n\n"
        f"{original_body}"
    )

    draft.body = combined_body

    # CRITICAL: Save as draft, do NOT send.
    draft.save_draft()
    print("Draft reply created (not sent). Check your Drafts folder in Outlook.")


def mark_as_read(message) -> None:
    """
    Mark the message as read.
    """
    if not message.is_read:
        print(f"Marking message '{message.subject}' as read...")
        message.is_read = True
        message.save()
    else:
        print(f"Message '{message.subject}' is already marked as read.")


def get_plain_text_body(message) -> str:
    """
    Extract a plain-text approximation of the message body.
    """
    # O365 usually gives `message.body` as HTML or text.
    body = message.body or ""
    # Optionally, you could strip HTML tags here. For now, just return as-is.
    return body


def process_single_email(message) -> None:
    """
    Run the full pipeline for a single email:
    - Extract data
    - Call LLM
    - Perform action (Archive / Draft Reply / Ignore)
    - Mark as read
    """
    sender_address = message.sender.address if message.sender else "Unknown"
    subject = message.subject or "(no subject)"
    body = get_plain_text_body(message)

    print("=" * 60)
    print(f"Processing email from: {sender_address}")
    print(f"Subject: {subject}")

    llm_result = call_openrouter_for_email(sender_address, subject, body)

    category = llm_result["category"]
    action = llm_result["action"]
    draft_text = llm_result["draft_text"]

    print(f"LLM decided -> category: {category}, action: {action}")

    action_lower = (action or "").lower()

    if action_lower == "archive":
        move_to_archive(message)
    elif action_lower == "draft reply":
        create_draft_reply(message, draft_text)
    elif action_lower == "ignore":
        print("Action is 'Ignore'; leaving message in Inbox.")
    else:
        print(f"Unrecognized action '{action}'. No mailbox change will be performed.")

    # Finally, mark as read so we don't process again
    mark_as_read(message)
    print("Finished processing this email.")
    print("=" * 60)


# ===============================================================
# MAIN ENTRYPOINT
# ===============================================================

def main():
    print("Starting Outlook Inbox Manager script...")

    if not MS_ACCOUNT_EMAIL:
        print("Warning: MS_ACCOUNT_EMAIL not set. This script will use whatever default mailbox O365 picks.")
        print("Set MS_ACCOUNT_EMAIL in .env if you want to target a specific mailbox.")

    account = get_o365_account()

    messages = fetch_unread_emails(account, limit=5)
    if not messages:
        print("No unread emails found.")
        return

    print(f"Found {len(messages)} unread email(s).")

    for message in messages:
        try:
            process_single_email(message)
        except Exception as e:
            print(f"Error processing email '{message.subject}': {e}")

    print("All done. Script exiting.")


if __name__ == "__main__":
    main()