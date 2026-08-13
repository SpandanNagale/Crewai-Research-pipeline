import sys
import argparse
from crew.research_crew import ResearchCrew
from eval.scorer import score_report

# Windows consoles often default to a legacy codepage (cp1252) that can't
# encode characters research reports commonly contain (em dashes, curly
# quotes, subscripts like CO2 written as CO2 with a subscript 2, etc.).
# Reconfigure stdout/stderr to UTF-8 so a report never crashes printing.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-agent research pipeline powered by Gemma 4 + CrewAI"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default="The impact of AI on drug discovery in 2024",
        help="Research topic to investigate",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Score the final report after generation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the knowledge-reuse check and run full research even if a similar "
        "question was already answered before",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    topic = args.topic

    print(f"\n{'='*60}")
    print(f"  Research Pipeline — Gemma 4 + CrewAI")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    crew = ResearchCrew()
    result = crew.run_with_meta(topic, force_rerun=args.force)
    report = result["report"]

    if result.get("reused"):
        print(f"\n[Knowledge reuse hit — similarity={result.get('similarity')}. "
              f"Re-run with --force for fresh research.]")
    else:
        print(f"\n[question_id={result['question_id']}  "
              f"findings={result['findings_count']}  "
              f"contradictions={result['contradictions_count']}]")

    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(report)

    if args.eval:
        print("\n" + "="*60)
        print("EVALUATION SCORES")
        print("="*60)
        result = score_report(report, topic)
        print(result)


if __name__ == "__main__":
    main()
