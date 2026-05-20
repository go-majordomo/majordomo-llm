#!/usr/bin/env python3

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from majordomo_llm import get_llm_instance
from majordomo_llm.logging import FileStorageAdapter, LoggingLLM, MixpanelAdapter

load_dotenv()

DB_PATH = Path(__file__).parent / "llm_logs.db"


async def main():
    # Create your LLM instance
    llm = get_llm_instance("anthropic", "claude-haiku-4-5-20251001")

    # add mixpanel token for your project
    db = await MixpanelAdapter.create("your_mixpanel_token")

    # Local file storage for request/response bodies
    storage = await FileStorageAdapter.create("./request_logs")

    # Wrap your LLM with logging
    logged_llm = LoggingLLM(llm, db, storage)

    # Use as normal - all requests are logged automatically
    response = await logged_llm.get_response("Hello!")

    print(f"  Content: {response.content}")

    # Don't forget to close connections when done
    await logged_llm.close()

if __name__ == "__main__":
    asyncio.run(main())
