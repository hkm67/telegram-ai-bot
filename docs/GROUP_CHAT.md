# Group chat mode

This bot is designed to sit in a **Telegram group or supergroup**, quietly capture conversation context on demand, then answer when someone **@mentions** it.

## Adding the bot to a group

1. Create or open your group/supergroup in Telegram.
2. Add the bot as a member (same as any other user).
3. If Telegram asks for permissions, allow the bot to read messages. The bot only reacts to commands, @mentions, and (for photos) captions; it does not read messages for the LLM until you use `/begin` and they are stored in the buffer.

In `.env`, set `BOT_USERNAME` to the **username** you chose in [@BotFather](https://t.me/BotFather), **without** `@`. Mentions in chat must use that exact username (case-insensitive match is fine).

## How capture and memory work

- **Per chat:** Each group (and each DM) has its own buffer. Nothing is shared between groups.
- **In memory only:** The buffer lives in the running process. Restarting `bot.py` clears all buffers.
- **Recording vs paused:**
  - **`/begin`** — New messages and photos are appended to the buffer while recording is on.
  - **`/end`** — Stops recording; the buffer is **kept**. You can still @mention the bot and it will use whatever is already in the buffer.
  - **`/clear`** — Empties the buffer and stops recording.

Messages are only added to the buffer while **`capture_active`** is true (after `/begin`, until `/end` or `/clear`). The bot still **requires a non-empty buffer** before it will call the LLM on an @mention: if you @mention it with an empty buffer, it tells you to use `/begin` first.

## Typical workflow

1. **`/begin`** — Start recording.
2. Chat as usual. The bot does **not** reply to normal messages.
3. Optional: **`/end`** when the discussion phase is over; the transcript stays in the buffer.
4. **`@YourBotUsername`** plus your question or request in the **same message** — The bot sends the buffered transcript (and any captured images) to the agent and replies in the thread.
5. **`/clear`** when you want to discard the session and start over.

Use **`/status`** anytime to see whether recording is on/off, how many lines are buffered, and which LLM model is active.

## Photos in groups

While recording is active, **photos** posted in the group are stored in the buffer (with optional caption). Image bytes are kept for the next agent turn so the model can use vision when supported.

- If the caption **includes** an @mention of the bot, the bot treats that caption as the user request and may reply immediately (still requires a non-empty buffer first).
- If there is no @mention, the photo is only stored for a later @mention on a text message (or a later photo caption that mentions the bot).

## What does *not* work like private chat

| Behavior | Group | Private (DM) |
|----------|-------|----------------|
| Reply without @mention | No — only commands and @mentions trigger bot replies to text | Yes — every text message is processed |
| Needs `/begin` for context | Yes — for using the shared group transcript | Optional — without `/begin`, the bot still answers using “(No captured context…)” as the transcript |
| `@bot` required in message | Yes (for text questions) | N/a |

So “group mode” is **opt-in context capture** plus **@mention to invoke** the agent.

## Admin-only behavior

Users listed in `ADMIN_USER_IDS` can run **`/model`** in the group to change the active LLM at runtime (see the main [README](../README.md) for examples).

## Troubleshooting

- **“No context captured yet”** — Run `/begin`, ensure at least one message (or photo while recording) is in the buffer, then @mention the bot. Use `/status` to confirm the buffer count.
- **Bot ignores my message** — In groups, plain text does not trigger an answer. Include `@YourBotUsername` in the message you want answered.
- **Wrong or missing username** — `BOT_USERNAME` in `.env` must match the bot’s Telegram username exactly.
