# ================================================================
#  evaluate.py  —  Test accuracy before demo
#  Run:  uv run python Evaluation/evaluate.py
# ================================================================

import sys
sys.path.insert(0, ".")
from src.rag_engine import get_answer

TESTS = [
    {"sem": "4", "subject": "operating_systems",
     "q": "What is a deadlock?",
     "keywords": ["deadlock", "process", "resource", "wait"]},

    {"sem": "4", "subject": "operating_systems",
     "q": "What are the conditions for deadlock?",
     "keywords": ["mutual exclusion", "hold", "circular", "wait"]},

    {"sem": "4", "subject": "database_management_system",
     "q": "What is normalisation in DBMS?",
     "keywords": ["normal form", "redundancy", "dependency"]},

    {"sem": "4", "subject": "computer_networks",
     "q": "Explain the OSI model",
     "keywords": ["layer", "physical", "transport", "application"]},

    {"sem": "4", "subject": "artificial_intelligence",
     "q": "What is a heuristic function?",
     "keywords": ["heuristic", "search", "estimate", "goal"]},

    {
        "sem": "6",
        "subject": "software_engineering",
        "q": "What is Software Engineering?",
        "keywords": ["software", "engineering", "discipline", "systematic", "development"]
    },
    {
        "sem": "6",
        "subject": "software_engineering",
        "q": "What is the SDLC (Software Development Life Cycle)?",
        "keywords": ["process", "phases", "planning", "development", "testing", "maintenance"]
    },
    {
        "sem": "6",
        "subject": "software_engineering",
        "q": "Explain the difference between Verification and Validation.",
        "keywords": ["verification", "validation", "product", "right", "requirements"]
    },
    {
        "sem": "6",
        "subject": "software_engineering",
        "q": "What is software configuration management (SCM)?",
        "keywords": ["change", "control", "version", "baseline", "tracking"]
    },
    {
        "sem": "6",
        "subject": "software_engineering",
        "q": "What is the difference between Cohesion and Coupling?",
        "keywords": ["cohesion", "coupling", "dependence", "module", "intra", "inter"]
    }
    # Add more from your actual TU past papers
]


def run():
    print("=" * 60)
    print("  TU BSc CSIT AI Study Buddy — Evaluation")
    print("=" * 60)
    passed = 0

    for i, t in enumerate(TESTS, 1):
        result  = get_answer(t["q"], sem=t["sem"], subject=t["subject"])
        answer  = result["answer"].lower()
        found   = [kw for kw in t["keywords"] if kw.lower() in answer]
        score   = len(found) / len(t["keywords"])
        ok      = score >= 0.6
        if ok:
            passed += 1

        print(f"\n[{i}] {'✅' if ok else '❌'}  ({score*100:.0f}%)  {t['q']}")
        if not ok:
            print(f"     Missing: {[kw for kw in t['keywords'] if kw not in answer]}")

    acc = passed / len(TESTS) * 100
    print(f"\n{'='*60}")
    print(f"  Score: {passed}/{len(TESTS)}  ({acc:.1f}%)")
    print("  🎉 Ready!" if acc >= 80 else "  ⚠️  Add more epub data or tune chunk size.")
    print("=" * 60)


if __name__ == "__main__":
    run()
