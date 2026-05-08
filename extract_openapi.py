import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich import print as rprint

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=api_key)

# 1. Define the shape that you want back


class ActionItem(BaseModel):
    description: str = Field(description="What needs to be done? Be specific.")
    owner: str = Field(description="Who is responsible for this?")
    deadline: str = Field(
        description="""When does this need to be done by? Use exact dates
        if available(e.g., 'Monday EOD', 'Tuesday'), otherwise 'ASAP'
        or 'TBD'."""
    )


class EmailSummary(BaseModel):
    sender: str
    urgency: str = Field(
        description="""Must be one of: 'low', 'medium', 'high', or 'critical'.
        Only these exact values."""
    )
    main_topic: str = Field(description="The main topic of the email")
    action_items: list[ActionItem]
    requires_reply: bool = Field(
        description="True if the email requires a reply, False otherwise. Binary choice only."
    )


# 2. Give it a piece of unstructured text to process
EMAIL = """
From: Maria Sanchez <maria@company.com>
Subject: URGENT — Q2 demo prep

Hi team,

The board demo is on Tuesday and we still don't have a working RAG prototype.
Erica, can you push the eval suite to staging by Monday EOD?
Tom, please prepare the slides — focus on the cost numbers.
Let me know if anyone is blocked. We can sync tomorrow at 10am if needed.

Thanks,
Maria"""

EMAIL2 = """From: Erica erica@example.com
To: Alex alex@example.com
Subject: Plans for next week and a few updates

Hey Alex,

Hope everything’s been good on your side. Things have been pretty busy here lately,
but in a good way.

A couple quick updates:

I finally finished the backend refactor I mentioned last month.
We deployed the new API version on Tuesday.
I’m still debugging one annoying authentication issue, but it’s mostly working now.
I also started going to the gym again this week, which has been surprisingly motivating.

For next week, here’s what I’m planning:

Finish the remaining test cases for the Pydantic validation flow.
Clean up the logging and monitoring dashboards.
Prepare slides for Thursday’s internal demo.
Schedule a meeting with the analytics team.
Review the open pull requests before Friday.

Also, are you still interested in collaborating on that side project idea? I was thinking
we could start with a small prototype first instead of trying to build everything at once.

By the way, I found two good resources you might like:

A FastAPI tutorial focused on async APIs
A guide on structured output parsing with Pydantic models

If you want, I can send the links later.

Anyway, let me know if you’re free sometime this weekend. Maybe coffee on Saturday
afternoon or dinner Sunday evening?

Talk soon,

Erica"""


# 3. Use OpenAI's structured-output API

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": """Extract structured data from emails with consistency.
            For urgency, use only: low, medium, high, or critical.
            For deadlines, use specific dates when available
            (e.g., 'Monday EOD', 'Tuesday'), otherwise 'ASAP' or 'TBD'.
            For requires_reply, answer with true or false.""",
        },
        {"role": "user", "content": EMAIL2},
    ],
    response_format=EmailSummary,
)

result: EmailSummary = response.choices[0].message.parsed
if not result:
    raise ValueError("Failed to extract email summary")

# 4. You now have a real Python object. Use it.
rprint("[bold green]Extracted EmailSummary:[/bold green]")
rprint(result.model_dump())

rprint("\n[bold yellow]Summary:[/bold yellow]")
rprint(f"[cyan]Urgency:[/cyan] {result.urgency}")
rprint(f"[cyan]Requires reply:[/cyan] {result.requires_reply}")
rprint(f"[cyan]Number of action items:[/cyan] {len(result.action_items)}")

rprint("\n[bold magenta]Action Items:[/bold magenta]")
for i, item in enumerate(result.action_items, 1):
    rprint(f"  {i}. [yellow]{item.description}[/yellow]")
    rprint(f"     Owner: {item.owner} | Due: {item.deadline}")
