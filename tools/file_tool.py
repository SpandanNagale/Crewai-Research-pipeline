import os
from datetime import datetime
from config.settings import settings


def save_report(content: str, topic: str) -> str:
    """Save the final report to the outputs directory. Returns the file path."""
    os.makedirs(settings.OUTPUTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = topic[:40].replace(" ", "_").replace("/", "-")
    filename = f"{timestamp}_{safe_topic}.md"
    filepath = os.path.join(settings.OUTPUTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Research Report: {topic}\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write("---\n\n")
        f.write(content)
    return filepath
