"""
river_split.py

Provides a single, deterministic train/test split of the ARA24 rivers, so that
training and evaluation always agree on which rivers are reserved for testing.
Without this, both training and evaluation sample from the same full pool of
1,073 rivers, meaning evaluation isn't testing genuine generalisation to
unseen conditions (see Dr. Bane's feedback on following standard evaluation
procedure).

Usage:
    from river_split import get_train_test_split
    train_ids, test_ids = get_train_test_split()
"""
import pandas as pd
import numpy as np

DEFAULT_TEST_FRACTION = 0.12  # ~129 of 1,073 rivers reserved for testing
DEFAULT_SPLIT_SEED = 123      # fixed seed -- must NOT change once training starts,
                               # or train/test sets will silently shift between runs


def get_train_test_split(csv_path="ARA24_Clean_Master_Enhanced.csv",
                          test_fraction=DEFAULT_TEST_FRACTION,
                          seed=DEFAULT_SPLIT_SEED):
    """
    Returns (train_river_ids, test_river_ids), both lists of strings, with no
    overlap, deterministically split from the full set of river IDs in the
    given CSV. Calling this multiple times with the same arguments always
    produces the identical split.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    all_ids = sorted(df["River ID"].astype(str).unique().tolist(), key=int)

    rng = np.random.RandomState(seed)  # RandomState (not default_rng) for
                                        # guaranteed identical output across
                                        # numpy versions -- deterministic
                                        # reproducibility matters more here
                                        # than using the newer API.
    shuffled = all_ids.copy()
    rng.shuffle(shuffled)

    n_test = int(round(len(shuffled) * test_fraction))
    test_ids = sorted(shuffled[:n_test], key=int)
    train_ids = sorted(shuffled[n_test:], key=int)

    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise RuntimeError(f"Train/test split produced overlapping rivers -- this should "
                            f"never happen: {overlap}")

    return train_ids, test_ids


if __name__ == "__main__":
    train_ids, test_ids = get_train_test_split()
    print(f"Total rivers: {len(train_ids) + len(test_ids)}")
    print(f"Train: {len(train_ids)} rivers")
    print(f"Test:  {len(test_ids)} rivers")
    print(f"Overlap check: {len(set(train_ids) & set(test_ids))} (should be 0)")
    print(f"First 10 test river IDs: {test_ids[:10]}")
