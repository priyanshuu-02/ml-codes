from pathlib import Path
import sys
import json

# Allow importing from src/
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.io_vnbd_dataset import IOVNBDSynchronizedDataset


SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


def split_sessions(sessions):

    if len(sessions) != 32:
        print(
            f"Warning: expected 32 sessions, "
            f"but found {len(sessions)}."
        )

    # Deterministic ordering
    sessions = sorted(
        sessions,
        key=lambda x: x["session_id"]
    )

    # Deterministic shuffle
    import random

    rng = random.Random(SEED)
    rng.shuffle(sessions)

    n = len(sessions)

    # Allocate closest possible 70/20/10 split
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)

    # Everything remaining goes to test
    n_test = n - n_train - n_val

    train = sessions[:n_train]
    val = sessions[n_train:n_train + n_val]
    test = sessions[n_train + n_val:]

    return train, val, test


def main():

    dataset = IOVNBDSynchronizedDataset(
        "data/IO-VNBD/synchronized"
    )

    sessions = dataset.get_sessions()

    train, val, test = split_sessions(sessions)

    print("=" * 70)
    print("IO-VNBD SESSION-LEVEL SPLIT")
    print("=" * 70)

    print(f"\nTotal sessions: {len(sessions)}")

    print(
        f"\nTRAIN: {len(train)} sessions "
        f"({len(train) / len(sessions) * 100:.2f}%)"
    )

    print(
        f"VALIDATION: {len(val)} sessions "
        f"({len(val) / len(sessions) * 100:.2f}%)"
    )

    print(
        f"TEST: {len(test)} sessions "
        f"({len(test) / len(sessions) * 100:.2f}%)"
    )

    print("\n" + "=" * 70)
    print("TRAIN SESSIONS")
    print("=" * 70)

    for session in train:
        print(
            session["session_id"],
            "|",
            session["category"]
        )

    print("\n" + "=" * 70)
    print("VALIDATION SESSIONS")
    print("=" * 70)

    for session in val:
        print(
            session["session_id"],
            "|",
            session["category"]
        )

    print("\n" + "=" * 70)
    print("TEST SESSIONS")
    print("=" * 70)

    for session in test:
        print(
            session["session_id"],
            "|",
            session["category"]
        )

    # ---------------------------------------------------------
    # Save split
    # ---------------------------------------------------------

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    split_data = {
        "seed": SEED,
        "total_sessions": len(sessions),
        "train": [
            s["session_id"] for s in train
        ],
        "validation": [
            s["session_id"] for s in val
        ],
        "test": [
            s["session_id"] for s in test
        ]
    }

    output_file = output_dir / "session_split.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split_data,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print("SPLIT SAVED")
    print("=" * 70)

    print(output_file.resolve())


if __name__ == "__main__":
    main()