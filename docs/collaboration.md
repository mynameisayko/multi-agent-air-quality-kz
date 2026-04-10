# Collaboration Setup

## Recommended storage split

Use `GitHub private repo` for code and writing.

Use `Google Drive`, `OneDrive`, or institutional cloud storage for:

- raw air quality data
- hospital or health-related data
- large intermediate files
- trained model checkpoints larger than a few hundred MB

## Branching

- `main` stays stable
- each contributor works in a feature branch
- merge through pull requests

Suggested branch naming:

- `feature/data-pipeline`
- `feature/tft-model`
- `feature/manuscript-intro`

## File ownership suggestion

- Person 1: data ingestion, preprocessing, feature engineering
- Person 2: modeling, evaluation, manuscript integration

## Minimal team rules

1. Do not commit raw sensitive data.
2. Keep notebooks readable and name them clearly.
3. Store reproducible logic in `src/`, not only in notebooks.
4. Document each dataset in `data/README.md`.
5. Add a short note for every major experiment in `reports/` or `docs/`.
