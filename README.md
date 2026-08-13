# Intelligent Customer Signal Detector

AI-powered early-warning system that identifies at-risk customers by fusing
behavioural, billing, and conversational signals into a ranked, explained
priority list for a customer operations team.

**Status:** in development — POC build.

## Problem

Customer ops teams are reactive: by the time a complaint is escalated or a
cancellation is requested, the chance to intervene has passed. Signals exist
across support chats, billing records, and satisfaction scores, but they are
siloed and reviewed manually.

## Approach

Multi-agent pipeline. Deterministic rule agents read structured data; an LLM
agent reads conversation transcripts and returns structured signals with
verbatim evidence. Signals are merged and scored by a transparent weighted
model, then a second LLM agent writes a plain-English explanation and a
recommended retention action.

Scoring is deterministic arithmetic; the LLM only produces language. Same
input always yields the same risk score.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add your Gemini API key
```


