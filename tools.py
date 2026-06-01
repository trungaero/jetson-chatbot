"""
Tool definitions for the LangGraph ReAct agent.
Add new tools here — they are automatically picked up by agent.py.
"""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import dateparser

import reminder_scheduler


# ── Internet Search ──────────────────────────────────────────────────────────

_search = DuckDuckGoSearchRun()


@tool
def search_internet(query: str) -> str:
    """Search the internet for current information.
    Use this for: news, weather, sports scores, recent events, live prices,
    or any question that requires up-to-date data not in training knowledge.

    Args:
        query: A concise search query string.

    Returns:
        A text summary of the top search results.
    """
    try:
        result = _search.run(query)
        return result if result else "No results found for that query."
    except Exception as e:
        return f"Search failed: {e}"


# ── Reminders ─────────────────────────────────────────────────────

@tool
def set_reminder(task: str, when: str) -> str:
    """Set a reminder to notify the user about a task at a specified time.
    Use this whenever the user asks to be reminded about something.

    Args:
        task: What to remind the user about (e.g. 'take medication', 'call John').
        when: Natural language time expression (e.g. 'in 10 minutes',
              'tomorrow at 9am', 'Friday at 3pm', '2026-06-02 14:00').

    Returns:
        Confirmation message with the reminder ID and scheduled time.
    """
    due = dateparser.parse(
        when,
        settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False},
    )
    if due is None:
        return (
            f"Could not understand the time '{when}'. "
            "Please try a clearer expression like 'in 30 minutes' or 'tomorrow at 9am'."
        )
    if due <= datetime.now():
        return f"The time '{when}' appears to be in the past. Please specify a future time."

    reminder_id = reminder_scheduler.add_reminder(task, due)
    return (
        f"Reminder set! ID: {reminder_id}. "
        f"I will remind you about '{task}' at {due.strftime('%A, %B %d at %H:%M')}."
    )


@tool
def list_reminders_tool() -> str:
    """List all pending reminders that have not yet fired.

    Returns:
        A formatted list of pending reminders, or a message if none exist.
    """
    pending = reminder_scheduler.list_reminders()
    if not pending:
        return "You have no pending reminders."
    lines = []
    for r in pending:
        due = datetime.fromisoformat(r["due"])
        lines.append(
            f"- ID {r['id']}: '{r['task']}' at {due.strftime('%A, %B %d at %H:%M')}"
        )
    return "Pending reminders:\n" + "\n".join(lines)


@tool
def cancel_reminder_tool(reminder_id: str) -> str:
    """Cancel a pending reminder by its ID.

    Args:
        reminder_id: The short ID returned when the reminder was created.

    Returns:
        Confirmation of cancellation, or an error if not found.
    """
    success = reminder_scheduler.cancel_reminder(reminder_id)
    if success:
        return f"Reminder {reminder_id} has been cancelled."
    return (
        f"No reminder found with ID '{reminder_id}'. "
        "Use list_reminders_tool to see active reminders."
    )


# ── Tool Registry ────────────────────────────────────────────────────────────

def get_tools() -> list:
    """Return all available tools for the agent."""
    return [
        search_internet,
        set_reminder,
        list_reminders_tool,
        cancel_reminder_tool,
    ]
