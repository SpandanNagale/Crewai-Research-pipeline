import sys
import argparse
from crew.research_crew import ResearchCrew
from eval.scorer import score_report


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
    return parser.parse_args()


def main():
    args = parse_args()
    topic = args.topic

    print(f"\n{'='*60}")
    print(f"  Research Pipeline — Gemma 4 + CrewAI")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    crew = ResearchCrew()
    report = crew.run(topic)

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
