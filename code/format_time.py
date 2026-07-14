import pandas as pd


def add_milliseconds(input_file, output_file=None):
    """
    Adds milliseconds to the start_time and end_time columns using the
    fractional part of start_time_sec and end_time_sec.

    Parameters
    ----------
    input_file : str
        Path to the input tab-delimited detection file.
    output_file : str, optional
        Path to save the updated file. If None, overwrites the input file.
    """

    df = pd.read_csv(input_file, sep="\t")

    # Convert to datetime
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # Add fractional seconds
    df["start_time"] += pd.to_timedelta(df["start_time_sec"] % 1, unit="s")
    df["end_time"] += pd.to_timedelta(df["end_time_sec"] % 1, unit="s")

    # Format with milliseconds
    df["start_time"] = df["start_time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    df["end_time"] = df["end_time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]

    if output_file is None:
        output_file = input_file

    df.to_csv(output_file, sep="\t", index=False)

    print(f"Saved updated file to {output_file}")


if __name__ == "__main__":
    input_file = r"F:\detections\Relabeled_cleaned\HAT_A_06_WMVZ_verified.txt"
    add_milliseconds(input_file)