"""Shared configuration for the dataset builder pipeline."""

from pathlib import Path

# Directories
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
RAW_ARTICLES_DIR = DATASET_DIR / "raw_articles"
CHUNKED_PASSAGES_DIR = DATASET_DIR / "chunked_passages"

# Ensure directories exist
for d in [RAW_ARTICLES_DIR, CHUNKED_PASSAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Seed articles — all post-October 2023
SEED_ARTICLES = [
    # Film & Awards
    "97th Academy Awards",
    "98th Academy Awards",
    "2024 Golden Globe Awards",
    "2025 Golden Globe Awards",
    "2024 BAFTA Awards",
    "2025 BAFTA Awards",
    "2024 in film",
    "2025 in film",
    # Music
    "66th Grammy Awards",
    "67th Grammy Awards",
    "2024 in music",
    # Sports
    "2024 Summer Olympics",
    "UEFA Euro 2024",
    "2024 FIFA Club World Cup",
    "2025 Australian Open",
    "2024 Wimbledon Championships",
    "2024 Tour de France",
    # Science & Technology
    "2024 Nobel Prize in Physics",
    "2024 Nobel Prize in Medicine",
    "2024 Nobel Prize in Chemistry",
    # Politics & World Events
    "2024 United States presidential election",
    "2025 in politics",
    # Literature & Culture
    "2024 Booker Prize",
    "2024 Pulitzer Prize",
    "2025 in literature",
]
