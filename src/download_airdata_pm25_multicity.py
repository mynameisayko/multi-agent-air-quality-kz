from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "external" / "airdata_kz_hourly_pm25"

CITY_FILES = {
    "almaty": "pm25.csv.gz",
    "astana": "pm25.csv.gz",
    "karaganda": "pm25.csv.gz",
    "rest_of_kz": "pm2_5.csv.gz",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/qazybekb/AirDatakz-OpenData/main"
    for city, filename in CITY_FILES.items():
        city_dir = OUT_DIR / city
        city_dir.mkdir(parents=True, exist_ok=True)
        url = f"{base}/{city}/{filename}"
        target = city_dir / "pm25.csv.gz"
        print(f"Downloading {city}: {url}")
        urlretrieve(url, target)
        print(f"Saved {target}")


if __name__ == "__main__":
    main()
