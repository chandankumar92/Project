import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RATE_PER_UNIT = 6.0


def load_data(path):
    """Load the CSV, validate it and return a clean DataFrame (Month, Units)."""
    try:
        raw = pd.read_csv(path, header=None, names=["Month", "Units"],
                          skipinitialspace=True)
    except FileNotFoundError:
        print(f"Error: file '{path}' not found.")
        print(f"Looked in: {os.path.dirname(os.path.abspath(path))}")
        folder = os.path.dirname(os.path.abspath(path))
        if os.path.isdir(folder):
            print("Files in that folder:")
            for name in sorted(os.listdir(folder)):
                print("   ", name)
        print("Tip: the file must be named exactly 'usage.csv' (not usage.csv.txt).")
        return None
    except Exception as exc:
        print(f"Error: could not read '{path}' ({exc}).")
        return None

    first = pd.to_numeric(raw.iloc[0], errors="coerce")
    if first.isna().all():
        raw = raw.iloc[1:].reset_index(drop=True)

    df = raw.apply(pd.to_numeric, errors="coerce")

    bad_rows = int(df.isna().any(axis=1).sum())
    if bad_rows:
        print(f"Warning: {bad_rows} row(s) with missing or non-numeric values "
              f"were removed.")
        df = df.dropna().reset_index(drop=True)

    return df


def show_summary(df):
    print("\n--- Data Summary ---")
    print(f"Rows           : {len(df)}")
    print(f"Months         : {int(df['Month'].min())} to {int(df['Month'].max())}")
    print(f"Units (min)    : {df['Units'].min():.2f}")
    print(f"Units (max)    : {df['Units'].max():.2f}")
    print(f"Units (mean)   : {df['Units'].mean():.2f}")
    print("--------------------")


def fit_model(df):
    """Fit Units = slope * Month + intercept and return slope, intercept, R2."""
    x = df["Month"].to_numpy(dtype=float)
    y = df["Units"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    predicted = slope * x + intercept
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return slope, intercept, r2


def plot_data(df, slope, intercept, future_month, show=True,
              filename="usage_plot.png"):
    x = df["Month"].to_numpy(dtype=float)
    y = df["Units"].to_numpy(dtype=float)
    x_line = np.linspace(min(x.min(), future_month), max(x.max(), future_month), 100)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="tab:blue", label="Actual usage", zorder=3)
    plt.plot(x_line, slope * x_line + intercept, color="tab:red",
             label="Regression line (predicted)")
    plt.scatter([future_month], [slope * future_month + intercept],
                color="tab:green", marker="*", s=150,
                label=f"Prediction (Month {future_month:g})", zorder=4)
    plt.title("Actual vs Predicted Electricity Usage")
    plt.xlabel("Month")
    plt.ylabel("Units consumed")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    if show:
        plt.show()


def predict(df):
    if len(df) < 2:
        print("Error: At least 2 months of data are required for prediction.")
        return

    raw = input("Enter month number to predict: ").strip()
    try:
        month = float(raw)
    except ValueError:
        print("Error: month must be a number.")
        return

    slope, intercept, r2 = fit_model(df)
    units = slope * month + intercept
    bill = units * RATE_PER_UNIT

    print(f"Slope: {slope:.2f} | Intercept: {intercept:.2f} | R2: {r2:.2f}")
    print(f"Predicted units (Month {month:g}): {units:.2f}")
    print(f"Estimated bill: Rs. {bill:.2f}")
    print("Line chart 'Actual vs Predicted' displayed")
    plot_data(df, slope, intercept, month)


TEST_CASES = [
    ("Test Case 1: Normal Input",
     [(1, 200), (2, 220), (3, 240), (4, 260)], 5),
    ("Test Case 2: Boundary / Minimum Input",
     [(1, 100), (2, 150)], 3),
    ("Test Case 3: Invalid Input",
     [(1, 100)], 2),
    ("Test Case 4: Different Valid Scenario",
     [(1, 100), (2, 120), (3, 150), (4, 170)], 6),
]


def run_all_tests():
    """Run all 4 built-in test cases (no CSV file needed)."""
    plotted = False
    for number, (title, rows, month) in enumerate(TEST_CASES, start=1):
        print(f"\n=== {title} ===")
        df = pd.DataFrame(rows, columns=["Month", "Units"])
        print(f"usage data: {rows} | Predict: Month {month}")

        if len(df) < 2:
            print("Error: At least 2 months of data are required for prediction.")
            continue

        slope, intercept, r2 = fit_model(df)
        units = slope * month + intercept
        print(f"Slope: {slope:.2f} | Intercept: {intercept:.2f} | R2: {r2:.2f}")
        print(f"Predicted units (Month {month}): {units:.2f}")
        print(f"Estimated bill: Rs. {units * RATE_PER_UNIT:.2f}")
        print("Line chart 'Actual vs Predicted' displayed")
        plot_data(df, slope, intercept, month, show=False,
                  filename=f"usage_plot_test{number}.png")
        plotted = True

    if plotted:
        plt.show()


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "usage.csv")
    df = load_data(path)
    if df is not None:
        show_summary(df)
    else:
        print("\n(No CSV loaded. You can still use option 3 to run the "
              "built-in test cases.)")

    while True:
        print("\nMenu:\n1. Predict\n2. Exit\n3. Run all 4 test cases")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            if df is None:
                print("Error: no data loaded. Put usage.csv next to this script.")
            else:
                predict(df)
        elif choice == "2":
            print("Goodbye!")
            break
        elif choice == "3":
            run_all_tests()
        else:
            print("Invalid option. Please enter 1, 2 or 3.")


if __name__ == "__main__":
    main()
