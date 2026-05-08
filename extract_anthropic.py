import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import print as rprint

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = Anthropic(api_key=api_key)


class ActionItem(BaseModel):
    description: str = Field(description="What needs to be done? Be specific.")
    owner: str = Field(description="Who is responsible for this?")
    deadline: str = Field(
        description="""When does this need to be done by?
                   Use exact dates if available
                   (e.g., 'Monday EOD', 'Tuesday'), otherwise 'ASAP' or 'TBD'."""
    )


class EmailSummary(BaseModel):
    sender: str
    urgency: str = Field(
        description="""Must be one of: 'low', 'medium',
        'high', or 'critical'.Only these exact values."""
    )
    main_topic: str = Field(description="The main topic of the email")
    action_items: list[ActionItem]
    requires_reply: bool = Field(
        description="""True if the email requires a reply,
                                 False otherwise. Binary choice only."""
    )


EMAIL = """
From: Maria Sanchez <maria@company.com>
Subject: URGENT — Q2 demo prep

Hi team,

The board demo is on Tuesday and we still don't have a working RAG prototype.
Erica, can you push the eval suite to staging by Monday EOD?
Tom, please prepare the slides — focus on the cost numbers.
Let me know if anyone is blocked. We can sync tomorrow at 10am if needed.

Thanks,
Maria
"""

# Generate JSON schema from the Pydantic model
schema = json.dumps(EmailSummary.model_json_schema(), indent=2)

# Give Claude the JSON schema and ask for valid JSON only.
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
    system=f"""You extract structured data from emails. Respond ONLY with
    valid JSON matching this schema. No prose, no markdown fences, just JSON.
    Schema: {schema}""",
    messages=[{"role": "user", "content": EMAIL}],
)
raw_json = response.content[0].text

# Extract JSON from markdown code blocks if present
if raw_json.strip().startswith("```"):
    # Remove markdown code blocks
    raw_json = raw_json.strip().split("```")[1]
    if raw_json.startswith("json"):
        raw_json = raw_json[4:]  # Remove "json" prefix
    raw_json = raw_json.strip()

result = EmailSummary.model_validate_json(raw_json)

rprint("[bold magenta]Extracted EmailSummary (Claude):[/bold magenta]")
rprint(result.model_dump())
