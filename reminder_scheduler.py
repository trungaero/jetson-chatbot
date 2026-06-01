"""
Reminder Scheduler — background thread that fires reminders when due.

Flow:
  1. set_reminder tool writes to reminders.json
  2. ReminderScheduler polls every POLL_INTERVAL seconds
  3. When a reminder is due, it puts a trigger string into reminder_queue
  4. main.py checks reminder_queue each loop iteration and injects it as
     an agent prompt — the agent then speaks the reminder aloud via TTS
"""

import json
import threading
import queue
import uuid
import os
from datetime import datetime


# Shared queue: scheduler → main loop
reminder_queue: queue.Queue = queue.Queue()

REMINDERS_FILE = os.path.join(os.path.dirname(__file__), "reminders.json")
POLL_INTERVAL = 20  # seconds between checks


# ── Persistence helpers ───────────────────────────────────────────────────────

def _load_reminders() -> list[dict]:
    """Load reminders from disk."""
    if not os.path.isfile(REMINDERS_FILE):
        return []
    try:
        with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_reminders(reminders: list[dict]):
    """Persist reminders to disk."""
    with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(reminders, f, indent=2, default=str)


# ── Public API (called from tools.py) ────────────────────────────────────────

def add_reminder(task: str, due: datetime) -> str:
    """Add a new reminder. Returns the reminder ID."""
    reminders = _load_reminders()
    reminder_id = str(uuid.uuid4())[:8]
    reminders.append({
        "id": reminder_id,
        "task": task,
        "due": due.isoformat(),
        "fired": False,
    })
    _save_reminders(reminders)
    return reminder_id


def list_reminders() -> list[dict]:
    """Return all pending (not yet fired) reminders."""
    return [r for r in _load_reminders() if not r["fired"]]


def cancel_reminder(reminder_id: str) -> bool:
    """Cancel a reminder by ID. Returns True if found and removed."""
    reminders = _load_reminders()
    original_count = len(reminders)
    reminders = [r for r in reminders if r["id"] != reminder_id]
    if len(reminders) < original_count:
        _save_reminders(reminders)
        return True
    return False


# ── Background scheduler ──────────────────────────────────────────────────────

class ReminderScheduler:
    """Background thread that checks reminders and fires them when due."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="ReminderScheduler",
            daemon=True,
        )

    def start(self):
        self._thread.start()
        pending = list_reminders()
        print(f"✓ Reminder scheduler started. {len(pending)} reminder(s) loaded.")

    def stop(self):
        self._stop_event.set()

    def _run(self):
        while not self._stop_event.is_set():
            self._check_reminders()
            self._stop_event.wait(timeout=POLL_INTERVAL)

    def _check_reminders(self):
        now = datetime.now()
        reminders = _load_reminders()
        changed = False

        for reminder in reminders:
            if reminder["fired"]:
                continue
            due = datetime.fromisoformat(reminder["due"])
            if now >= due:
                trigger = f"[REMINDER] Please remind the user about: {reminder['task']}"
                reminder_queue.put(trigger)
                reminder["fired"] = True
                changed = True
                print(f"\n   🔔 Reminder fired: {reminder['task']}")

        if changed:
            _save_reminders(reminders)
