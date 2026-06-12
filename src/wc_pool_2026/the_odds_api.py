import os
import pandas as pd
import requests

from dotenv import load_dotenv

try:
    from .paths import ENV_FILE
except ImportError:
    from paths import ENV_FILE

load_dotenv(ENV_FILE)

API_KEY = os.getenv("api_key")


def main() -> None:
    sports = requests.get(
        "https://api.the-odds-api.com/v4/sports/",
        params={"apiKey": API_KEY, "all": "true"},
        timeout=30,
    ).json()

    df = pd.DataFrame(sports)

    print(
        df[
            df["key"].str.contains("soccer", case=False, na=False)
            & df["has_outrights"].fillna(False)
        ][["key", "title", "description", "active"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
