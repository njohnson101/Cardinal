# Cardinal

Documenting my tinkering with a personal AI agent.

Right now, Cardinal acts as a personal assistant that collects persistent memory across interactions. Cardinal currently runs on Claude haiku 4.5 and delegates to other AI's if necessary.

Current capabilities include:
- **Journal access**: can access my Obsidian journal through a couple different search tools. Pulls from my experiences to answer questions
- **Calendar access**: connects to Google Calendar to help with scheduling tasks
- **Research**: can query Gemini 2.5 pro for advanced reasoning tasks and two levels of Perplexity models (sonar and sonar-pro) for web searches

This repository includes both a CLI (`python cardinal.py`) and primitive GUI (`python main.py').

Heavily vibe-coded.

Capabilities that I will add when my Cursor tokens reset:
1. Email filtering and sorting
2. More robust scheduling assistance
3. Goal setting and management
