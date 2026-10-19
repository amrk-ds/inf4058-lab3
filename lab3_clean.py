"""Lab 3 - INF4058: Look, Profile, Find problems, Clean. (Stops before Postgres.)"""
import pandas as pd

PATH = "yellow_tripdata_2021-01.csv.gz"   # download it; do NOT commit it to git
CHUNK = 100_000
DATE_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
DTYPES = {
    "VendorID": "Int64", "passenger_count": "Int64", "RatecodeID": "Int64",
    "PULocationID": "Int64", "DOLocationID": "Int64", "payment_type": "Int64",
    "store_and_fwd_flag": "string",
}
JAN_START, FEB_START = pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-01")


def read_chunks(path=PATH, chunksize=CHUNK):
    return pd.read_csv(path, dtype=DTYPES, parse_dates=DATE_COLS, chunksize=chunksize)


# ---------- Step 1: Look ----------
def look(path=PATH):
    df = pd.read_csv(path, nrows=100)
    print(df.head())
    print(df.dtypes)


# ---------- Step 2: Profile (chunked, read-only) ----------
def profile(path=PATH):
    n, nulls, mins, maxs, dtypes = 0, None, {}, {}, None
    for ch in read_chunks(path):
        n += len(ch)
        dtypes = ch.dtypes
        nulls = ch.isna().sum() if nulls is None else nulls + ch.isna().sum()
        for c in ch.columns:
            if ch[c].dtype == "string":
                continue
            lo, hi = ch[c].min(), ch[c].max()
            if pd.isna(lo):          # all-null column in this chunk (e.g. passenger_count)
                continue
            mins[c] = lo if c not in mins else min(mins[c], lo)
            maxs[c] = hi if c not in maxs else max(maxs[c], hi)
    out = pd.DataFrame({
        "dtype": dtypes.astype(str),
        "nulls": nulls,
        "nulls_%": (nulls / n * 100).round(2),
        "min": pd.Series(mins),
        "max": pd.Series(maxs),
    })
    return n, out


# ---------- Step 3: Find problems (counts only, no changes) ----------
def find_problems(path=PATH):
    c = dict.fromkeys([
        "total_rows", "pickup_outside_jan_2021", "dropoff_before_pickup",
        "fare_amount<=0", "total_amount<=0", "trip_distance<=0",
        "passenger_count==0", "passenger_count_null", "duplicate_rows",
    ], 0)
    seen = None
    for ch in read_chunks(path):
        c["total_rows"] += len(ch)
        c["pickup_outside_jan_2021"] += int((~ch.tpep_pickup_datetime.between(JAN_START, FEB_START, inclusive="left")).sum())
        c["dropoff_before_pickup"] += int((ch.tpep_dropoff_datetime < ch.tpep_pickup_datetime).sum())
        c["fare_amount<=0"] += int((ch.fare_amount <= 0).sum())
        c["total_amount<=0"] += int((ch.total_amount <= 0).sum())
        c["trip_distance<=0"] += int((ch.trip_distance <= 0).sum())
        c["passenger_count==0"] += int((ch.passenger_count == 0).sum())
        c["passenger_count_null"] += int(ch.passenger_count.isna().sum())
        # duplicates (within and across chunks) via row hashes
        h = pd.util.hash_pandas_object(ch, index=False)
        dup_in = int(h.duplicated().sum())
        h = h.drop_duplicates()
        dup_across = int(h.isin(seen).sum()) if seen is not None else 0
        c["duplicate_rows"] += dup_in + dup_across
        seen = h if seen is None else pd.concat([seen, h]).drop_duplicates()
    return pd.Series(c, name="count")


# ---------- Step 5: Clean ----------
def clean(df: pd.DataFrame):
    """Return (cleaned_df, {rule: rows_removed}). The input df is never modified."""
    df = df.copy()
    report = {}

    def drop(rule, mask):
        nonlocal df
        report[rule] = int(mask.sum())
        df = df[~mask]

    drop("duplicate_rows", df.duplicated())
    drop("pickup_outside_jan_2021",
         ~df.tpep_pickup_datetime.between(JAN_START, FEB_START, inclusive="left"))
    drop("dropoff_before_pickup", df.tpep_dropoff_datetime < df.tpep_pickup_datetime)
    drop("fare_amount<=0", df.fare_amount <= 0)
    drop("trip_distance<=0", df.trip_distance <= 0)

    # FIX (not drop): passenger_count 0 / missing -> <NA> so the row stays but the bad value is not trusted
    bad_pc = (df.passenger_count == 0) | df.passenger_count.isna()
    report["passenger_count_fixed_to_NA"] = int(bad_pc.sum())   # fixed, not removed
    df["passenger_count"] = df["passenger_count"].mask(df["passenger_count"] == 0, pd.NA)

    # FLAG: trips longer than 24h are kept but marked for the owner of the rule
    df["flag_long_trip"] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime) > pd.Timedelta("24h")
    report["flagged_trip_over_24h"] = int(df.flag_long_trip.sum())  # flagged, not removed
    return df, report


def clean_all(path=PATH):
    """Apply clean() chunk by chunk (never load the whole file), summing the reports.
    Duplicates are caught within a chunk only here; cross-chunk duplicates counted in find_problems()."""
    total, kept, rep = 0, 0, {}
    for ch in read_chunks(path):
        total += len(ch)
        out, r = clean(ch)
        kept += len(out)
        for k, v in r.items():
            rep[k] = rep.get(k, 0) + v
    return total, kept, rep


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("=== STEP 1: LOOK ===");   look()
    print("\n=== STEP 2: PROFILE ==="); n, prof = profile(); print("rows:", n); print(prof)
    print("\n=== STEP 3: PROBLEMS ==="); print(find_problems())
    print("\n=== STEP 5: CLEAN ===")
    total, kept, rep = clean_all()
    print("rows before:", total); print("rows after :", kept)
    print("rows removed:", total - kept)
    print(pd.Series(rep, name="count"))
