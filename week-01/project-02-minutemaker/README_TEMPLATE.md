# Project 2 - MinuteMaker

One or two lines: what this service does and who it is for.

## Overview
Takes a raw meeting transcript, returns structured minutes (attendees,
decisions, action items), and emails them to a recipient. Built on the
ReleaseBot shape: a forced tool call whose arguments are the structured output,
plus a separate streaming endpoint.

## Setup
- Python 3.10+ in a virtual environment
- `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and fill in:
  - `OPENAI_API_KEY`
  - `SMTP_SENDER` (your Gmail address)
  - `SMTP_PASSWORD` (a Gmail app password, 2-Step Verification required)
- `.env` is gitignored. Never commit secrets.

## Run
```
uvicorn app.main:app --reload      # http://localhost:8000
```
(You can also run it in Google Colab with the same launcher pattern as
ReleaseBot's notebook; the own-machine path above is what this repo assumes.)

## API
- `POST /minutes` - transcript + recipient in; forces send_minutes, emails the
  minutes, returns structured JSON. One round trip.
- `POST /minutes-stream` - streams the minutes as plain text over SSE, no tool.

## Architecture
Replace this with your own sketch (ASCII is fine). Show the request path: caller
-> endpoint (422 if empty) -> forced tool call (one round trip) -> email side
effect + Pydantic validation -> JSON back.
```
[ your diagram here ]
```

## Design decisions
- Schema: how MeetingMinutes is shaped, and why owner and due are nullable but
  still required under strict mode.
- Forced tool call: how tool_choice makes the arguments the structured output.
- Failure handling: what happens on a bad model output, an SMTP failure, and an
  empty transcript.
- Prompt: how you instruct the model to use empty lists and null when unsure.

## How I verified it (checkpoints)
Note which of the six checkpoints you hit, and how you tested the five
transcripts (especially the traps: 03 empty, 04 -> 422, 05 null owners).

## Known limitations
Be honest. What does it not handle well yet?

## AI assistance
Which AI tools you used and how (this is expected and fine; say what you did).

## Demo
Unlisted video link (2 to 3 minutes, one take, showing it work).
