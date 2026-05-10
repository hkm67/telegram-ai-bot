# Group Chat Mode

This document explains how the bot behaves inside Telegram **groups** and
**supergroups**: how it captures context, when it speaks, how to address it,
and how photos are handled. For DM behaviour see [Private chat mode](#private-chat-mode-tldr)
at the bottom of this document.

> TL;DR — Add the bot to a group, run `/begin`, chat normally, then tag the
> bot (`@yourbotname what was decided?`) to get an answer that uses the
> captured conversation as context.

---

## How group mode works

The bot sits silently in the group and only replies when it is **explicitly
mentioned**. Between mentions, it can record the conversation into a
per-chat **capture buffer** so it has context to answer with later.

The lifecycle of a group session is:

1. `/begin` — start recording messages into the buffer.
2. People chat normally. The bot stays silent but stores every text message
   (and any photo) sent while capture is active.
3. `/end` — pause recording. The buffer is preserved.
4. `@botname <question>` — the bot answers using the buffered conversation
   plus any photos as context.
5. `/clear` — wipe the buffer and stop recording when you're done.

Capture state is **per chat**. Each group has its own independent buffer and
on/off state, so multiple groups using the same bot don't interfere with
each other.

### State diagram

```
              /begin                       /end
   ┌──────────────────────┐      ┌──────────────────────┐
   │                      ▼      │                      ▼
[ OFF, buffer kept ] ──► [ ON, recording ] ──► [ OFF, buffer kept ]
                                                        │
                                                        │ /clear
                                                        ▼
                                              [ OFF, buffer empty ]
```

- `/begin` only flips capture **on**. It does not clear the buffer, so you
  can pause with `/end`, run a query, then `/begin` again to keep adding.
- `/end` only flips capture **off**. The buffer is preserved so you can
  still tag the bot after recording has stopped.
- `/clear` is the only command that empties the buffer.

---

## Commands available in a group

| Command         | Who   | Description |
|-----------------|-------|-------------|
| `/begin`        | anyone | Start recording messages into the capture buffer. |
| `/end`          | anyone | Pause recording. The buffer is kept. |
| `/clear`        | anyone | Wipe the buffer and stop recording. |
| `/status`       | anyone | Show whether capture is on, how many messages are buffered, and the active model. |
| `/model <spec>` | admin  | Switch the active LLM at runtime (see below). |

Admin status is controlled by the `ADMIN_USER_IDS` env var (comma-separated
Telegram numeric user IDs).

### `/status` output

`/status` is the quickest way to debug "why didn't the bot answer?".

```
Capture: ON
Buffered messages: 42
Active model: ollama:gemma4:26b
```

- `Capture: OFF` means new messages are **not** being added to the buffer.
- `Buffered messages: 0` means there is no context for the bot to use yet —
  tagging the bot will return a "no context captured" reply.

### `/model` (admin)

```
/model gemini-2.0-flash
/model gemini-1.5-pro
/model ollama:gemma4:26b
/model ollama:qwen3-coder-next:latest
```

The bot always starts on the local Ollama model from `OLLAMA_MODEL`. If an
Ollama call fails at request time and `GEMINI_API_KEY` is set, the bot
automatically retries with `GEMINI_FALLBACK_MODEL` and tells the chat which
model actually answered.

---

## How to address the bot

The bot only responds to messages that contain its `@username` mention. The
mention can appear anywhere in the message; everything around it (except
the mention itself) is treated as the request.

Examples — assuming `BOT_USERNAME=mybot`:

| Message                                  | Bot reacts? | Request sent to LLM         |
|------------------------------------------|-------------|-----------------------------|
| `hello everyone`                         | No (just buffered) | — |
| `@mybot summarise the discussion`        | Yes         | `summarise the discussion`  |
| `hey @mybot what did we decide?`         | Yes         | `hey what did we decide?`   |
| `@MyBot`                                 | Yes         | (empty → bot prints help)   |
| `email me at hi@mybot.example`           | Yes (matches the prefix) | The bot will reply — pick a `BOT_USERNAME` that won't appear in normal text. |

A few details worth knowing:

- Mention matching is **case-insensitive** but uses a word boundary, so
  `@mybotextra` does **not** trigger the bot.
- A message with **only** the mention and nothing else triggers a short help
  reply (so people who tag the bot without a question get guidance).
- If capture is off and the buffer is empty when you tag the bot, it will
  ask you to run `/begin` first instead of answering with no context.

### Replies use HTML formatting

Bot replies are sent with `parse_mode="HTML"`, so the LLM is expected to
return Telegram-compatible HTML (`<b>`, `<i>`, `<code>`, `<pre>`, etc.). If
HTML parsing fails, the bot will fall back to a plain-text error message
rather than crashing the handler.

---

## What gets captured

While capture is **on**, every text message in the group is appended to the
buffer as `{username, text}`. Commands (anything starting with `/`) are
**not** captured — they are handled as commands and never reach the message
handler.

### Photos

Photos posted in the group are handled separately:

- The bot always **downloads** the highest-resolution version of each photo
  it sees in the group, regardless of capture state, so it is available
  later if you tag the bot.
- If capture is **on**, the photo is also recorded in the buffer as
  `[photo: <caption if any>]` with the raw bytes stored alongside.
- If the photo's **caption** mentions the bot (`@yourbotname what is this?`),
  the bot replies immediately, using the buffered conversation plus all
  buffered images as context.
- If there is no mention in the caption, the bot stays silent — you can tag
  it later in a follow-up message and it will still have the image.

If you don't want photos in the context, simply `/end` before sending them,
or `/clear` afterwards.

---

## Typical group flow

```text
You:   /begin
Bot:   Recording started. Messages will be captured until /end.

Alice: We need to pick a release date.
Bob:   I'd say next Friday. Tuesday is too tight.
Alice: Agreed, Friday works for me.

You:   /end
Bot:   Recording paused. 3 message(s) in buffer.

You:   @mybot summarise what we decided about the release date.
Bot:   Alice and Bob agreed the release will ship next Friday.

You:   /clear
Bot:   Buffer cleared. Use /begin to start a new session.
```

You can interleave `/begin` and `/end` as many times as you like before
clearing — the buffer accumulates everything captured during the "on"
windows.

---

## Permissions and group setup

For the bot to actually see group messages, Telegram requires either:

- **Privacy mode disabled** in [BotFather](https://t.me/BotFather)
  (`/setprivacy` → `Disable`), so the bot receives all group messages, or
- **Promoting the bot to admin** in the group, which has the same effect.

If neither is done, the bot will only receive messages that explicitly
mention it. Commands like `/begin` and `/status` will still work, but the
buffer will stay empty because the bot never sees the conversation in
between.

You also want to make sure the bot can:

- Read messages (covered by the privacy setting above).
- Send messages and reply to messages (granted by default when added to a
  group).

The bot does **not** need permissions to delete messages, pin messages, or
manage users.

---

## Multiple groups, multiple admins

- Each group's capture state and buffer are independent. `/begin` in group A
  does not start capture in group B.
- The `/model` switch is **global** to the bot process. Switching models in
  one group changes the model used to answer in every group.
- `ADMIN_USER_IDS` is also global. Anyone whose Telegram user ID is in that
  list can run `/model` in any group the bot is in.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Bot doesn't react to `/begin` in a group. | Bot is not actually in the group, or commands are being delivered to a different bot. | Re-add the bot; double-check `BOT_USERNAME` matches the bot you're addressing. |
| `/status` shows `Capture: ON` but `Buffered messages: 0` even though people are chatting. | Telegram privacy mode is enabled, so the bot only sees mentions. | In BotFather, run `/setprivacy` → `Disable`, **or** promote the bot to group admin. Re-add the bot to the group afterwards. |
| Tagging the bot replies with "No context captured yet." | Either `/begin` was never run, or `/clear` wiped the buffer. | Run `/begin`, chat, then tag the bot. |
| Bot replies "Sorry, only admins can change the model." | Your Telegram user ID isn't in `ADMIN_USER_IDS`. | Add your numeric user ID to the `ADMIN_USER_IDS` env var and restart the bot. |
| Bot answers but says it used a different model than expected. | The primary backend (Ollama) failed and the Gemini fallback was used. | Check the bot logs for the Ollama error; verify `OLLAMA_HOST` is reachable from wherever the bot runs. |
| Bot replies with a formatting error or a plain "Sorry, something went wrong" message. | The model returned HTML that Telegram couldn't parse. | This is recovered automatically; if it keeps happening, try a different model with `/model`. |

---

## Private chat mode (TL;DR)

Group mode and private (DM) mode are deliberately different:

- In **groups**, the bot only answers when mentioned, and only has context
  if `/begin` was used.
- In **DMs**, every message you send is treated as a request to the bot. You
  don't need to mention it. If you've run `/begin` in the DM, your messages
  are also added to the same capture buffer and used as context; otherwise
  the bot answers from just your latest message.

Everything else in this document (commands, `/model`, photos, fallback
behaviour) works the same way in a DM.
