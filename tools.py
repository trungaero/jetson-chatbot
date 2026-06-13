"""
Tool definitions for the LangGraph ReAct agent.
Add new tools here — they are automatically picked up by agent.py.
"""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import dateparser
import subprocess

import reminder_scheduler
from camera_tool import describe_image_with_local_llm, capture_image_b64

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
    print("   ⏳ Parsing reminder time...")
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
    print(
        f"   ✓ Reminder set! ID: {reminder_id}. Scheduled for {due.strftime('%A, %B %d at %H:%M')}."
    )
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
    print("   ⏳ Fetching pending reminders...")
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
    print(f"   ⏳ Attempting to cancel reminder ID {reminder_id}...")
    success = reminder_scheduler.cancel_reminder(reminder_id)
    if success:
        print(f"   ✓ Reminder {reminder_id} has been cancelled.")
        return f"Reminder {reminder_id} has been cancelled."
    return (
        f"No reminder found with ID '{reminder_id}'. "
        "Use list_reminders_tool to see active reminders."
    )


# ── System Control ────────────────────────────────────────────────────────────


@tool
def shutdown_device(delay_seconds: int = 0) -> str:
    """Shutdown the Jetson Orin device.
    Use this when the user explicitly requests to power down or shutdown the device.

    Args:
        delay_seconds: Delay before shutdown in seconds (default: 0 for immediate).
                      Useful for graceful shutdown with a warning.

    Returns:
        Confirmation message or error if shutdown fails.
    """
    try:
        print(f"   🔴 Initiating device shutdown (delay: {delay_seconds}s)...")
        if delay_seconds > 0:
            subprocess.run(
                [
                    "sudo",
                    "shutdown",
                    "-h",
                    f"+{delay_seconds // 60}",
                    f"Device shutting down in {delay_seconds} seconds.",
                ],
                check=True,
                timeout=10,
            )
        else:
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True, timeout=10)
        return "Device shutdown command sent successfully. The Jetson Orin is shutting down."
    except subprocess.CalledProcessError as e:
        return f"Shutdown command failed: {e}. Make sure the application has sudo privileges."
    except Exception as e:
        return f"Error initiating shutdown: {e}"


# ── Camera Vision ────────────────────────────────────────────────────────────


@tool
def look_around_with_camera(user_question: str) -> str:
    """Capture what the camera currently sees and return it for visual analysis.
    Use this tool whenever the user asks what you see, what is in front of you,
    what is happening around you, to describe the environment, or any similar
    request that requires vision or visual observation.

    Args:
        user_question: The user's request or question about the environment.

    Returns:
        Description of the current camera view, or an error message if capture/analysis fails.
    """
    try:
        image_data = capture_image_b64()
        base64_jpeg = image_data.get("base64_jpeg", None)
        if not base64_jpeg:
            error_msg = image_data.get("error", "Unknown error during image capture.")
            return f"Failed to capture image: {error_msg}"
        else:
            print("   ⏳ Analyzing image with local LLM...")
            description = describe_image_with_local_llm(base64_jpeg, user_question)
            return description
    except Exception as e:
        return f"Error during camera capture or analysis: {e}"


# ── Tool Registry ────────────────────────────────────────────────────────────


def get_tools() -> list:
    """Return all available tools for the agent."""
    return [
        search_internet,
        set_reminder,
        list_reminders_tool,
        cancel_reminder_tool,
        shutdown_device,
        look_around_with_camera,
    ]
