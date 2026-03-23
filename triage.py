#!/usr/bin/env python3
"""
Gmail Email Triage Agent
Fetches unread emails, classifies them via Claude Haiku (OpenRouter),
applies Gmail labels, and logs results to triage_log.txt.
"""

import base64
import json
import os
import re
import sys
from datetime import datetime, timezone
from email import message_from_bytes

import argparse

import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "credentials.json")
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "token.json")
LOG_FILE = os.path.join(os.path.dirname(__file__), "triage_log.txt")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4-5"
VALID_CATEGORIES = {"Urgent", "Action Items", "Reading", "Archives", "Newsletters"}


# ── Gmail auth ───────────────────────────────────────────────────────────────
def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                # Stale token (e.g. scope mismatch) — delete and re-auth
                os.remove(TOKEN_FILE)
                creds = None
        if not creds or not creds.valid:
            if not os.path.exists(CREDENTIALS_FILE):
                sys.exit(f"ERROR: {CREDENTIALS_FILE} not found. Download it from Google Cloud Console.")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# ── Gmail helpers ────────────────────────────────────────────────────────────
def fetch_inbox_emails(service, max_results=50):
    result = service.users().messages().list(
        userId="me", q="in:inbox", maxResults=max_results
    ).execute()
    messages = result.get("messages", [])
    emails = []
    for msg in messages:
        full = service.users().messages().get(
            userId="me", id=msg["id"], format="full"
        ).execute()
        emails.append(full)
    return emails


def parse_email(msg):
    headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
    sender = headers.get("From", "Unknown")
    subject = headers.get("Subject", "(no subject)")
    body = extract_body(msg["payload"])
    return {"id": msg["id"], "sender": sender, "subject": subject, "body": body}


def extract_body(payload):
    """Recursively extract plain-text body from a Gmail message payload."""
    mime = payload.get("mimeType", "")
    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    if mime.startswith("multipart/"):
        for part in payload.get("parts", []):
            text = extract_body(part)
            if text:
                return text
    return ""


def get_or_create_label(service, name):
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for label in labels:
        if label["name"].lower() == name.lower():
            return label["id"]
    new_label = service.users().labels().create(
        userId="me", body={"name": name, "labelListVisibility": "labelShow",
                           "messageListVisibility": "show"}
    ).execute()
    return new_label["id"]


def apply_label(service, message_id, label_id, keep_unread=False):
    remove = ["INBOX"] if keep_unread else ["INBOX", "UNREAD"]
    service.users().messages().modify(
        userId="me",
        id=message_id,
        body={"addLabelIds": [label_id], "removeLabelIds": remove},
    ).execute()


# ── OpenRouter / Claude ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert email triage assistant.
Analyze the email and return ONLY valid JSON (no markdown fences) with these fields:
- category: one of "Urgent", "Action Items", "Reading", "Archives", "Newsletters"
- summary: one concise sentence describing the email
- draft_reply: a short, professional reply if a response is warranted, otherwise null

Categorization guide:
- Urgent: requires a response within 24 hours.
- Action Items: requires you to do something — reply, RSVP, submit, accept/decline a calendar invite, or attend an event. Applies to emails from professors, advisors, peers, university offices, and student organizations.
- Reading: informative but no action required — newsletters, digests, event announcements with no RSVP, receipts, order confirmations, and anything good to know but not actionable.
- Archives: nothing needed — automated notifications, system alerts, sign-on notifications, marketing emails, and anything that doesn't need to be seen.
- Newsletters: newsletters and digests from independent writers, publications, or organizations where the primary purpose is substantive content — analysis, research, commentary, or education on topics you've chosen to follow (e.g. AI safety, technology, policy). Typically arrives via Substack, Mailchimp, or similar newsletter platforms. Do NOT use this for newsletters from universities, employers, or institutions that are primarily promoting their own events, programs, jobs, or services — those belong in Reading or Action Items."""


def triage_email_with_claude(sender, subject, body):
    if not OPENROUTER_API_KEY:
        sys.exit("ERROR: OPENROUTER_API_KEY not set in .env")

    # Truncate body to avoid excessive token usage
    body_snippet = body[:2000].strip() if body else "(no body)"

    user_message = f"""From: {sender}
Subject: {subject}

{body_snippet}"""

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/email-triage",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
        },
        timeout=30,
    )
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if the model adds them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


# ── Logging ──────────────────────────────────────────────────────────────────
def append_log(timestamp, sender, subject, category, summary, draft_reply):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"Timestamp : {timestamp}\n")
        f.write(f"Sender    : {sender}\n")
        f.write(f"Subject   : {subject}\n")
        f.write(f"Category  : {category}\n")
        f.write(f"Summary   : {summary}\n")
        if draft_reply:
            f.write(f"Draft Reply:\n{draft_reply}\n")
        f.write("\n")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    print("Authenticating with Gmail...")
    service = get_gmail_service()

    print("Fetching inbox emails...")
    raw_emails = fetch_inbox_emails(service, max_results=args.limit)

    if not raw_emails:
        print("No inbox emails found.")
        return

    print(f"Found {len(raw_emails)} inbox email(s). Triaging...\n")

    # Cache label IDs to avoid redundant API calls
    label_cache: dict[str, str] = {}

    for raw in raw_emails:
        email = parse_email(raw)
        print(f"  → [{email['subject'][:60]}] from {email['sender'][:40]}")

        try:
            result = triage_email_with_claude(
                email["sender"], email["subject"], email["body"]
            )
        except (requests.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"     ERROR during triage: {e}")
            continue

        category = result.get("category", "Action Items")
        if category not in VALID_CATEGORIES:
            category = "Action Items"

        summary = result.get("summary", "")
        draft_reply = result.get("draft_reply")

        # Apply Gmail label
        if category not in label_cache:
            label_cache[category] = get_or_create_label(service, category)
        apply_label(service, email["id"], label_cache[category],
                    keep_unread=category == "Newsletters")

        # Write to log
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        append_log(timestamp, email["sender"], email["subject"],
                   category, summary, draft_reply)

        print(f"     Category  : {category}")
        print(f"     Summary   : {summary}")
        if draft_reply:
            print(f"     Draft     : {draft_reply[:80]}{'...' if len(draft_reply) > 80 else ''}")
        print()

    print(f"Done. Results appended to {LOG_FILE}")


if __name__ == "__main__":
    main()
