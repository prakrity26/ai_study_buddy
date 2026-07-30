"""Discover study-material folders that are actually present in Data/."""

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
SEMESTERS = {
    "Semester IV": "4", "Semester V": "5", "Semester VI": "6",
    "Semester VII": "7", "Semester VIII": "8",
}
SLUG_OVERRIDES = {
    "design and analysis of algorithms": "design_and_analysis_of_algorithms",
    "system analysis and design": "system_analysis_and_design",
    "simulation and modeling": "simulation_and_modeling",
    "net centric computing": "dotnet_centric_computing",
    "e-commerce": "ecommerce",
    "data warehousing and data mining": "data_warehousing_and_mining",
    "network and system administration": "network_and_system_administration",
    "geographical information system": "geographical_information_system",
    "introduction to cloud computing": "cloud_computing",
    "advanced networking with ipv6": "advanced_networking_ipv6",
    "distributed and object oriented database": "distributed_and_object_oriented_database",
}


def _slug(name: str) -> str:
    normalized = name.strip().lower()
    return SLUG_OVERRIDES.get(normalized, re.sub(r"[^a-z0-9]+", "_", normalized).strip("_"))


def available_subjects() -> list[dict]:
    """Return only folders containing EPUB or PDF study material."""
    subjects = []
    for semester_folder, sem in SEMESTERS.items():
        semester_path = DATA_DIR / semester_folder
        if not semester_path.exists():
            continue
        for folder in sorted(path for path in semester_path.iterdir() if path.is_dir()):
            files = sorted(path for path in folder.iterdir()
                           if path.is_file() and path.suffix.lower() in {".epub", ".pdf"})
            if not files:
                continue
            match = re.match(r"^([A-Z]+\d+)\s*-\s*(.+)$", folder.name)
            code, name = match.groups() if match else ("", folder.name)
            subjects.append({"sem": sem, "slug": _slug(name),
                             "label": f"{name} ({code})" if code else name,
                             "path": folder, "files": files})
    return subjects
