#!/usr/bin/env python

import base64
import requests

from dotenv import load_dotenv
import os
import argparse
import time

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to analyze the image
def analyze_image(
    image=None,
    image_path=None,
    prompt=None,
    model=None,
    max_tokens=100,
    api_key=api_key,
):
    prompt = prompt or "What's in this image?"
    model = model or "gpt-4o"

    if image_path:
        base64_image = encode_image(image_path)
    elif image:
        base64_image = image
    else:
        raise ValueError("Either image or image_path must be provided")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    start_time = time.time()
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    end_time = time.time()

    elapsed_time = end_time - start_time

    return response.json(), elapsed_time


# Function to analyze text prompt only
def analyze_prompt(prompt, model=None, max_tokens=100, api_key=api_key):
    model = model or "gpt-4o"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": max_tokens,
    }

    start_time = time.time()
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    end_time = time.time()

    elapsed_time = end_time - start_time

    return response.json(), elapsed_time


# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(
        description="Analyze an image or prompt using OpenAI API"
    )
    parser.add_argument("input", type=str, help="Path to the image file or prompt text")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to describe the image",
        default="What's in this image?",
    )
    parser.add_argument(
        "--base64",
        action="store_true",
        help="Indicate if the image is provided as a base64 string",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for analysis",
        default=None,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="Maximum number of tokens to generate",
        default=100,
    )

    args = parser.parse_args()

    if args.base64:
        result, elapsed_time = analyze_image(
            image=args.input,
            prompt=args.prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            api_key=api_key,
        )
    elif os.path.isfile(args.input):
        result, elapsed_time = analyze_image(
            image_path=args.input,
            prompt=args.prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            api_key=api_key,
        )
    else:
        result, elapsed_time = analyze_prompt(
            prompt=args.input,
            model=args.model,
            max_tokens=args.max_tokens,
            api_key=api_key,
        )

    print("Response:")
    print(result)
    print(f"\nElapsed Time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
