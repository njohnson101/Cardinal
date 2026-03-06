"""
Terminal-based AI chat agent with persistent memory.
Uses OpenRouter with Claude Haiku 4.5 via the OpenAI SDK.
"""

"""
Cardinal version 2.0.0
"""

import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MEMORY_FILE = "memory.json"

_today = datetime.now()
_today_date = _today.strftime("%Y-%m-%d")
_today_weekday = _today.strftime("%A")
SYSTEM_PROMPT = (
    "You are my Master AI Orchestrator. Your name is Cardinal. You are helpful, concise, "
    "and have persistent memory of our conversations. "
    f"Today's date is {_today_date} ({_today_weekday})."
)

TOKEN_THRESHOLD = 3000
KEEP_MOST_RECENT_MESSAGES = 6
CHAT_MODEL = "anthropic/claude-haiku-4.5"
SUMMARY_MODEL = "anthropic/claude-haiku-4.5"

# Path to your local Obsidian vault (update this to your real path)
OBSIDIAN_PATH = "/mnt/c/Users/smook/OneDrive/Documents/Selfmaxxing"

# Google Calendar scopes
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_vault",
            "description": (
                "Searches the user's local Obsidian vault (markdown notes) for information. "
                "Use this tool whenever the user asks about their journals, life events, "
                "long-term projects, or anything that might be stored in their personal notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text to look for inside the Obsidian markdown files.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_date_range",
            "description": (
                "Use this tool to read the user's daily journal entries across a specific date range. "
                "Use this when the user asks for a summary of a week, month, or year. "
                "YOU MAY ALSO use this tool autonomously if you feel reading the last few days of "
                "entries will give you better context to answer a general question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (inclusive).",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (inclusive).",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_research",
            "description": (
                "Use this tool to delegate complex questions, deep research, or web/current events "
                "to a specialized sub-agent. Do not try to answer current events yourself; always use "
                "this tool. Pass 'deep_dive' for complex reasoning, 'simple_web' for straightforward "
                "web lookups (facts, definitions, simple queries), or 'current_events' for complex "
                "news/current events and multi-faceted web research."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research question or topic to investigate.",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["deep_dive", "simple_web", "current_events"],
                        "description": "Use 'deep_dive' for complex reasoning, 'simple_web' for simple web lookups, 'current_events' for complex news/research.",
                    },
                },
                "required": ["query", "search_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_todays_schedule",
            "description": (
                "Use this tool whenever the user asks about their schedule, meetings, "
                "or calendar events for today or the near future."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def estimate_tokens(messages: list[dict]) -> int:
    total_chars = 0
    for msg in messages:
        content = msg.get("content") or ""
        total_chars += len(content)
    return total_chars // 4


def search_vault(query: str) -> str:
    """
    Search the local Obsidian vault for the query string (case-insensitive).
    Returns up to 5 matches with filename and a 300-character snippet.
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return "No query provided to search the Obsidian vault."

    if not os.path.isdir(OBSIDIAN_PATH):
        return f"Obsidian vault path not found: {OBSIDIAN_PATH}"

    matches: List[str] = []

    for root, _, files in os.walk(OBSIDIAN_PATH):
        for filename in files:
            if not filename.endswith(".md"):
                continue

            full_path = os.path.join(root, filename)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            content_lower = content.lower()
            idx = content_lower.find(query_lower)
            if idx == -1:
                continue

            start = max(0, idx - 150)
            end = min(len(content), idx + 150)
            snippet = content[start:end].replace("\n", " ").strip()

            rel_name = os.path.relpath(full_path, OBSIDIAN_PATH)
            matches.append(f"- {rel_name}: {snippet}")

            if len(matches) >= 5:
                break
        if len(matches) >= 5:
            break

    if not matches:
        return f"No matches found in the Obsidian vault for query: '{query}'."

    header = f"Top {len(matches)} matches in Obsidian vault for query: '{query}':"
    return header + "\n" + "\n\n".join(matches)


def delegate_research(query: str, search_type: str) -> str:
    """
    Delegate research to a specialized model via OpenRouter.
    search_type: 'deep_dive' -> gemini-2.5-pro, 'simple_web' -> sonar, 'current_events' -> sonar-pro.
    """
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        return "Error: OPENROUTER_API_KEY not found in environment."

    if search_type == "deep_dive":
        model = "google/gemini-2.5-pro"
    elif search_type == "simple_web":
        model = "perplexity/sonar"
    elif search_type == "current_events":
        model = "perplexity/sonar-pro"
    else:
        return f"Invalid search_type '{search_type}'. Use 'deep_dive', 'simple_web', or 'current_events'."

    system_prompt = (
        "You are an expert research agent. Provide a comprehensive, highly detailed, "
        "and factual report on the user's query."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Cardinal Research",
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return (content or "").strip() or "No response from research agent."
    except requests.RequestException as e:
        return f"Research request failed: {e}"
    except (KeyError, IndexError, TypeError) as e:
        return f"Unexpected response format: {e}"


def get_calendar_service():
    """
    Authenticate with Google Calendar API and return a service client.
    Uses token.json for cached credentials; falls back to credentials.json + OAuth flow.
    """
    creds = None
    token_path = "token.json"

    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds or not creds.valid:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError(
                    "credentials.json not found. Please place your Google OAuth client file in the project root."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def get_todays_schedule() -> str:
    """
    Fetch the next 10 upcoming events on the user's primary calendar from now onward.
    Returns a formatted string of Event Name, Start, End.
    """
    try:
        service = get_calendar_service()
    except Exception as e:
        return f"Failed to access Google Calendar: {e}"

    now = datetime.now(timezone.utc).isoformat()

    try:
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
    except Exception as e:
        return f"Error fetching events from Google Calendar: {e}"

    events = events_result.get("items", [])
    if not events:
        return "No upcoming events found for today or the near future."

    lines: list[str] = ["Upcoming events (next 10):"]
    for event in events:
        summary = event.get("summary", "(No title)")
        start_raw = event.get("start", {}).get(
            "dateTime", event.get("start", {}).get("date", "")
        )
        end_raw = event.get("end", {}).get(
            "dateTime", event.get("end", {}).get("date", "")
        )
        lines.append(f"- {summary}\n  Start: {start_raw}\n  End:   {end_raw}")

    return "\n".join(lines)


def read_date_range(start_date: str, end_date: str) -> str:
    """
    Read daily journal entries between start_date and end_date (YYYY-MM-DD, inclusive).
    Looks for files named M-D-YYYY.md inside the '50 Journal' folder in the Obsidian vault.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return (
            "Invalid date format. Please use YYYY-MM-DD for both start_date and end_date."
        )

    if start > end:
        start, end = end, start

    journals_dir = os.path.join(OBSIDIAN_PATH, "50 Journal")
    if not os.path.isdir(journals_dir):
        return f"Obsidian journals folder not found: {journals_dir}"

    current = start
    parts: List[str] = []

    while current <= end:
        # Filenames are M-D-YYYY.md (no leading zeros), e.g. 3-2-2026.md
        filename = f"{current.month}-{current.day}-{current.year}.md"
        full_path = os.path.join(journals_dir, filename)
        if os.path.isfile(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
            except Exception:
                content = ""

            if content:
                parts.append(f"### {current.strftime('%Y-%m-%d')}\n{content}\n")

        current += timedelta(days=1)

    if not parts:
        return "No journal entries found for this date range."

    return "\n".join(parts)


def summarize_memory(client: OpenAI, messages: list[dict]) -> list[dict]:
    """
    Compress older messages into a single rolling summary message.
    Keeps the original system prompt (messages[0]) and the N most recent messages intact.
    """
    if not messages:
        return messages

    if len(messages) <= 1 + KEEP_MOST_RECENT_MESSAGES:
        return messages

    system_prompt = messages[0]
    recent = messages[-KEEP_MOST_RECENT_MESSAGES:]
    older = messages[1:-KEEP_MOST_RECENT_MESSAGES]

    if not older:
        return messages

    summarizer_messages = [
        {
            "role": "system",
            "content": "You are a summarization engine. Output only the summary paragraph.",
        },
        {
            "role": "user",
            "content": (
                "Summarize this conversation history into a dense, highly detailed paragraph. "
                "Retain all personal facts, names, user preferences, and context.\n\n"
                "Conversation history (JSON):\n"
                f"{json.dumps(older, ensure_ascii=False)}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=summarizer_messages,
    )

    summary_text = (response.choices[0].message.content or "").strip()
    summary_message = {
        "role": "system",
        "content": f"Summary of older conversation: {summary_text}",
    }

    return [system_prompt, summary_message, *recent]


def load_memory() -> list[dict]:
    """Load conversation history from memory.json, creating it with system prompt if needed."""
    if not os.path.exists(MEMORY_FILE):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        save_memory(messages)
        return messages

    with open(MEMORY_FILE, "r") as f:
        messages = json.load(f)

    # Ensure system prompt is at the start (for upgrades/corruption recovery)
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        save_memory(messages)

    return messages


def save_memory(messages: list[dict]) -> None:
    """Overwrite memory.json with the current message array."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(messages, f, indent=2)


def main() -> None:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Cardinal CLI",
        },
    )

    messages = load_memory()

    print("AI Chat Agent (type 'exit' or 'quit' to stop)\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Wrap user input in structured JSON including today's date and current time
        current_time = datetime.now().strftime("%H:%M")
        user_payload = {
            "text": user_input,
            "today": _today_date,
            "time": current_time,
        }
        messages.append(
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            }
        )

        if estimate_tokens(messages) > TOKEN_THRESHOLD:
            print("[System: Compressing memory in background...]")
            try:
                messages = summarize_memory(client, messages)
                save_memory(messages)
            except Exception as e:
                print(f"[System: Memory compression failed: {e}]")

        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            print(f"API Error: {e}")
            messages.pop()  # Remove failed user message from history
            continue

        message = response.choices[0].message

        # Handle optional tool calls (OpenAI tool calling via OpenRouter)
        tool_calls = getattr(message, "tool_calls", None) or []

        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            if tool_name == "search_vault":
                query = arguments.get("query", "")
                print(f"[System: Searching Obsidian vault for '{query}'...]")
                tool_result = search_vault(query)
            elif tool_name == "read_date_range":
                start_date = arguments.get("start_date", "")
                end_date = arguments.get("end_date", "")
                print(
                    f"[System: Reading journal entries from {start_date} to {end_date}...]"
                )
                tool_result = read_date_range(start_date, end_date)
            elif tool_name == "delegate_research":
                query = arguments.get("query", "")
                search_type = arguments.get("search_type", "deep_dive")
                print(
                    f"[System: Cardinal is delegating research for '{query}'...]"
                )
                tool_result = delegate_research(query, search_type)
            elif tool_name == "get_todays_schedule":
                print("[System: Cardinal is checking your Google Calendar...]")
                tool_result = get_todays_schedule()
            else:
                tool_result = f"Unknown tool '{tool_name}' requested."

            # Log tool call and result into the message history
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_result,
                }
            )

            try:
                followup = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                )
                message = followup.choices[0].message
            except Exception as e:
                print(f"API Error after tool call: {e}")
                continue

        assistant_message = message.content or ""
        print(f"\nCardinal: {assistant_message}\n")

        messages.append({"role": "assistant", "content": assistant_message})
        save_memory(messages)


if __name__ == "__main__":
    main()
