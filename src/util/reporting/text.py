from typing import Dict, Tuple
import pandas as pd


def history_to_csv(history: Dict[str, Tuple[float]], path: str) -> None:
    df = pd.DataFrame(history)
    cols = pd.DataFrame({"epoch": list(range(1, len(df) + 1))})
    history = pd.concat([cols, df], axis=1)
    history.to_csv(path, index=False)
