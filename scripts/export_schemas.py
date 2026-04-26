"""Dump Pydantic schemas to contracts/schemas.json for non-Python consumers."""
import json
from pathlib import Path

from backend.app.schemas import FeedbackRequest, PredictRequest, PredictResponse

OUT = Path(__file__).resolve().parents[1] / "contracts" / "schemas.json"


def main() -> None:
    schemas = {
        "PredictRequest": PredictRequest.model_json_schema(),
        "PredictResponse": PredictResponse.model_json_schema(),
        "FeedbackRequest": FeedbackRequest.model_json_schema(),
    }
    OUT.write_text(json.dumps(schemas, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
