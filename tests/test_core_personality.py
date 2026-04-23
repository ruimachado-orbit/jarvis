from jarvis.core.personality import SYSTEM_PROMPT, proactive_alert


def test_system_prompt_present():
    assert "Jarvis" in SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT) > 200


def test_proactive_alert_calendar():
    msg = proactive_alert("calendar", "call with Sarah in 25 minutes", "send her the deck link")
    assert "Sarah" in msg
    assert "deck link" in msg
    assert "?" in msg


def test_proactive_alert_email():
    msg = proactive_alert("email", "urgent email from CEO", "draft a reply")
    assert "CEO" in msg
    assert "draft" in msg


def test_proactive_alert_no_action():
    msg = proactive_alert("calendar", "dentist appointment tomorrow", None)
    assert "dentist" in msg
    assert msg.strip()
