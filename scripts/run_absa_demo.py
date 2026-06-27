#!/usr/bin/env python3
"""
Minimal demo: run ABSA on a single hospital review using GPT-4o.

This script demonstrates how the prompt template is combined with a
patient review and sent to the OpenAI API. It processes ONE review
and prints the structured JSON result.

Usage:
    export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
    python run_absa_demo.py --review "醫生很專業但掛號等太久"
    python run_absa_demo.py --file my_review.txt

Requirements:
    pip install openai
"""

import argparse
import json
from pathlib import Path

from openai import OpenAI

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "stage1_closed_absa_prompt_zh-TW.txt"
MODEL = "gpt-4o"


def run_absa(review_text: str) -> dict:
    """Send one review through the ABSA prompt and return parsed JSON."""
    client = OpenAI()
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt.replace("{review_text}", review_text)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(description="ABSA demo — single review")
    parser.add_argument("--review", type=str, help="Review text (Chinese)")
    parser.add_argument("--file", type=Path, help="Read review from a text file")
    args = parser.parse_args()

    if args.file:
        review_text = args.file.read_text(encoding="utf-8").strip()
    elif args.review:
        review_text = args.review
    else:
        review_text = "醫生很專業很有耐心，但是掛號等了快兩個小時，櫃檯人員態度也不太好。"

    print(f"Review: {review_text[:80]}{'...' if len(review_text) > 80 else ''}")
    print()

    result = run_absa(review_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
