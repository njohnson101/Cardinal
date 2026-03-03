"""
Cardinal desktop app using customtkinter.
Wraps the existing terminal-based Cardinal agent logic in a GUI.
"""

import json
import threading
import os

import customtkinter as ctk
from dotenv import load_dotenv
from openai import OpenAI

from cardinal import (
    CHAT_MODEL,
    TOKEN_THRESHOLD,
    TOOLS,
    delegate_research,
    estimate_tokens,
    load_memory,
    read_date_range,
    save_memory,
    search_vault,
    summarize_memory,
)


class CardinalApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        # Basic window setup
        self.title("Cardinal - Master AI")
        self.geometry("800x600")

        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Chat display (read-only)
        self.chat_display = ctk.CTkTextbox(self, wrap="word")
        self.chat_display.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        self.chat_display.configure(state="disabled")

        # Bottom frame for input and send button
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        bottom_frame.grid_columnconfigure(0, weight=1)

        self.input_entry = ctk.CTkEntry(bottom_frame)
        self.input_entry.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")
        self.input_entry.bind("<Return>", self._on_enter_pressed)

        self.send_button = ctk.CTkButton(
            bottom_frame,
            text="Send",
            command=self._on_send_clicked,
        )
        self.send_button.grid(row=0, column=1, pady=5)

        # Load environment and client
        load_dotenv()
        api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not api_key:
            self.client = None
            self._append_to_chat(
                "[System] Error: OPENROUTER_API_KEY not found. Add it to your .env and restart."
            )
            self.send_button.configure(state="disabled")
            self.input_entry.configure(state="disabled")
        else:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Cardinal GUI",
                },
            )

        # Load existing memory and render it
        self.messages = load_memory()
        self._render_history()

    # ---------------- UI helpers ----------------

    def _append_to_chat(self, text: str) -> None:
        """Append text to the chat display in a thread-safe way."""
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", text + "\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def _append_safe(self, text: str) -> None:
        self.after(0, self._append_to_chat, text)

    def _render_history(self) -> None:
        """Render existing message history into the chat display."""
        for msg in self.messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            if not content:
                continue

            if role == "system":
                # Show system messages in brackets
                self._append_to_chat(f"[System] {content}")
            elif role == "user":
                self._append_to_chat(f"You: {content}")
            elif role == "assistant":
                self._append_to_chat(f"Cardinal: {content}")
            elif role == "tool":
                name = msg.get("name", "tool")
                self._append_to_chat(f"[Tool {name}]: {content}")

    # ---------------- Event handlers ----------------

    def _on_enter_pressed(self, event) -> None:
        self._on_send_clicked()

    def _on_send_clicked(self) -> None:
        user_text = self.input_entry.get().strip()
        if not user_text:
            return

        # Echo user message in UI and clear entry
        self._append_to_chat(f"You: {user_text}")
        self.input_entry.delete(0, "end")

        # Run the AI call in a background thread
        thread = threading.Thread(
            target=self._handle_message_thread, args=(user_text,), daemon=True
        )
        thread.start()

    # ---------------- Background worker ----------------

    def _handle_message_thread(self, user_text: str) -> None:
        if self.client is None:
            return

        # Reuse the same messages list as the terminal agent
        messages = self.messages

        messages.append({"role": "user", "content": user_text})

        # Rolling summary if needed
        if estimate_tokens(messages) > TOKEN_THRESHOLD:
            self._append_safe("[System: Compressing memory in background...]")
            try:
                messages[:] = summarize_memory(self.client, messages)
                save_memory(messages)
            except Exception as e:
                self._append_safe(f"[System: Memory compression failed: {e}]")

        # First completion (may or may not call tools)
        try:
            response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            self._append_safe(f"[System: API Error: {e}]")
            messages.pop()  # Remove failed user message
            return

        message = response.choices[0].message

        # Tool routing (supports both search_vault and read_date_range)
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
                self._append_safe(
                    f"[System: Searching Obsidian vault for '{query}'...]"
                )
                tool_result = search_vault(query)
            elif tool_name == "read_date_range":
                start_date = arguments.get("start_date", "")
                end_date = arguments.get("end_date", "")
                self._append_safe(
                    f"[System: Reading journal entries from {start_date} to {end_date}...]"
                )
                tool_result = read_date_range(start_date, end_date)
            elif tool_name == "delegate_research":
                query = arguments.get("query", "")
                search_type = arguments.get("search_type", "deep_dive")
                self._append_safe(
                    f"[System: Cardinal is delegating research for '{query}'...]"
                )
                tool_result = delegate_research(query, search_type)
            else:
                tool_result = f"Unknown tool '{tool_name}' requested."
                self._append_safe(f"[System: {tool_result}]")

            # Record the tool call and its result in history
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

            # Follow-up completion so Cardinal can use the tool_result
            try:
                followup = self.client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                )
                message = followup.choices[0].message
            except Exception as e:
                self._append_safe(f"[System: API Error after tool call: {e}]")
                save_memory(messages)
                return

        assistant_message = message.content or ""
        self._append_safe(f"Cardinal: {assistant_message}")

        messages.append({"role": "assistant", "content": assistant_message})
        save_memory(messages)


if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    app = CardinalApp()
    app.mainloop()

