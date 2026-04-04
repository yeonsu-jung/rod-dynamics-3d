import pandas as pd

EXTREME_RODS_CSV = '/Users/yeonsu/GitHub/rod-dynamics-3d/assets/extreme_rods.csv'
VALID_METRICS = {"MinFSA", "MinFTA", "MaxFSA", "MaxFTA"}

df = pd.read_csv(EXTREME_RODS_CSV)


def _normalize_seed_id(random_seeds: str) -> str:
    return random_seeds.replace(",", "_")

def look_up_extreme_rod(N, AR, random_seeds, metric):
    if not isinstance(N, int) or not isinstance(AR, int):
        raise TypeError("N and AR must both be integers")
    if not isinstance(random_seeds, str):
        raise TypeError("random_seeds must be a string like '355_359_829'")
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}")
    seed_id = _normalize_seed_id(random_seeds)

    the_row = df[
        (df["N"] == N)
        & (df["AR"] == AR)
        & (df["ID"] == seed_id)
        & (df["Metric"] == metric)
    ]
    if the_row.empty:
        raise LookupError(
            f"No extreme rod found for N={N}, AR={AR}, ID={seed_id}, metric={metric}"
        )
    if len(the_row) != 1:
        raise LookupError(
            f"Expected exactly one row for N={N}, AR={AR}, ID={seed_id}, metric={metric}; got {len(the_row)}"
        )

    row = the_row.iloc[0]
    rod_index = int(row["RodIndex"])
    value = float(row["Value"])
    return rod_index, value


if __name__ == "__main__":
    
    rod_index, value = look_up_extreme_rod(1000,200,"355_359_829", "MaxFTA")
    print(rod_index, value)
