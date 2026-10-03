#!/usr/bin/env python3
"""Build, execute, and verify Week-2-Assignment.ipynb."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as stats  # noqa: E402

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "Week-2-Assignment.ipynb"
CSV_PATH = ROOT / "Telco-Customer-Churn.csv"
MPL_DIR = ROOT / ".mplconfig"
MPL_DIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": split_src(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_src(source),
    }


def split_src(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            out.append(line + "\n")
        else:
            if line:
                out.append(line)
    if not out:
        return []
    return out


CELLS = [
    md(
        """# Week 2 Assignment — Describe Your Data

**Aashish Chand**  
International American University  
AIM 402 Statistics for Data Analysis Using Python  
Professor Poshan Karki  
September 28, 2026

---

**Dataset:** IBM Telco Customer Churn (BlastChar, n = 7,043 customers)  
**Numerical variables:** `tenure` (months as a customer) and `MonthlyCharges` (US dollars)  
**Categorical variable for comparison:** `Contract` (Month-to-month / One year / Two year)

All 7,043 rows are kept. `tenure` and `MonthlyCharges` have no missing values, so the 11 customers with blank `TotalCharges` (all `tenure` = 0) stay in this analysis."""
    ),
    code(
        '''# Cell 1 — Load libraries and the full dataset (n = 7,043)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

df = pd.read_csv("Telco-Customer-Churn.csv")

# Convert TotalCharges for inspection only. Do not drop rows: tenure and
# MonthlyCharges are complete, and the 11 blanks are the tenure = 0 customers.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(), errors="coerce"
)

VAR1 = "tenure"
VAR2 = "MonthlyCharges"
x1 = df[VAR1]
x2 = df[VAR2]

print("Dataset shape:", df.shape)
print(f"Missing in {VAR1}: {x1.isna().sum()}")
print(f"Missing in {VAR2}: {x2.isna().sum()}")
print(f"Blank TotalCharges (kept): {df['TotalCharges'].isna().sum()}")
print(f"tenure == 0 rows (kept): {(df['tenure'] == 0).sum()}")
df[[VAR1, VAR2, "Contract", "InternetService", "Churn"]].head(10)'''
    ),
    code(
        '''# Cell 2 — Summary table for the two analysis variables
def describe_one(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return {
        "n": int(series.count()),
        "mean": series.mean(),
        "median": series.median(),
        "std (n-1)": series.std(),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "min": series.min(),
        "max": series.max(),
        "skew": series.skew(),
    }

summary = pd.DataFrame({VAR1: describe_one(x1), VAR2: describe_one(x2)})
print("All later cells use this same df (n = {}).".format(len(df)))
summary.round(3)'''
    ),
    md(
        """---
## 2. Centre, Shape and Spread
---"""
    ),
    code(
        '''# Cell 3 — Summary statistics for tenure
mean_t = x1.mean()
median_t = x1.median()
std_t = x1.std()
q1_t = x1.quantile(0.25)
q3_t = x1.quantile(0.75)
iqr_t = q3_t - q1_t
min_t = x1.min()
max_t = x1.max()
skew_t = x1.skew()
gap_t = mean_t - median_t

print("=== tenure ===")
print(f"  n      = {x1.count()}")
print(f"  Mean   = {mean_t:.2f}")
print(f"  Median = {median_t:.2f}")
print(f"  SD     = {std_t:.2f}")
print(f"  Q1     = {q1_t:.2f}")
print(f"  Q3     = {q3_t:.2f}")
print(f"  IQR    = {iqr_t:.2f}")
print(f"  Min    = {min_t},  Max = {max_t}")
print(f"  Skew   = {skew_t:.4f}")'''
    ),
    code(
        '''# Cell 4 — Summary statistics for MonthlyCharges
mean_mc = x2.mean()
median_mc = x2.median()
std_mc = x2.std()
q1_mc = x2.quantile(0.25)
q3_mc = x2.quantile(0.75)
iqr_mc = q3_mc - q1_mc
min_mc = x2.min()
max_mc = x2.max()
skew_mc = x2.skew()
gap_mc = mean_mc - median_mc

print("=== MonthlyCharges ===")
print(f"  n      = {x2.count()}")
print(f"  Mean   = {mean_mc:.2f}")
print(f"  Median = {median_mc:.2f}")
print(f"  SD     = {std_mc:.2f}")
print(f"  Q1     = {q1_mc:.2f}")
print(f"  Q3     = {q3_mc:.2f}")
print(f"  IQR    = {iqr_mc:.2f}")
print(f"  Min    = {min_mc},  Max = {max_mc}")
print(f"  Skew   = {skew_mc:.4f}")'''
    ),
    code(
        '''# Cell 5 — Histograms used for the shape descriptions
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(x1, bins=24, color="steelblue", edgecolor="black")
axes[0].set_title("Distribution of tenure")
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Frequency")
axes[1].hist(x2, bins=24, color="darkorange", edgecolor="black")
axes[1].set_title("Distribution of MonthlyCharges")
axes[1].set_xlabel("Monthly charges (USD)")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()'''
    ),
    code(
        '''# Cell 6 — Describe centre, shape, spread from the statistics just computed
print("=== tenure: Centre, Shape, Spread ===")
print(
    f"  Centre : The centre is {median_t:.0f} months (median), "
    f"with a mean of {mean_t:.2f} months."
)
print(
    "  Shape  : Mildly right-skewed, with a spike of new customers at 0–1 months"
)
print("           and a second spike at the 72-month ceiling.")
print(
    f"  Spread : Wide, with SD = {std_t:.2f} months and IQR = {iqr_t:.0f} months, "
    f"values from {min_t:.0f} to {max_t:.0f}."
)
print()
print("=== MonthlyCharges: Centre, Shape, Spread ===")
print(
    f"  Centre : The centre is about ${median_mc:.2f} (median), "
    f"with a mean of ${mean_mc:.2f}."
)
print(
    "  Shape  : Mildly left-skewed and bimodal — a tall peak near $20 and a"
)
print("           broad hump between roughly $70 and $100.")
print(
    f"  Spread : Wide, with SD = ${std_mc:.2f} and IQR = ${iqr_mc:.2f}, "
    f"from ${min_mc:.2f} to ${max_mc:.2f}."
)'''
    ),
    code(
        '''# Cell 7 — Skewness with .skew() checked against the mean–median gap
print("=== Skewness Check ===")
print()
print("tenure:")
print(f"  .skew()       = {skew_t:.4f}")
print(f"  Mean - Median = {mean_t:.2f} - {median_t:.2f} = {gap_t:.4f}")
if skew_t > 0 and gap_t > 0:
    print("  Both positive — they agree on mild right skew.")
elif skew_t < 0 and gap_t < 0:
    print("  Both negative — they agree on mild left skew.")
else:
    print("  Signs DISAGREE — investigate further.")

print()
print("MonthlyCharges:")
print(f"  .skew()       = {skew_mc:.4f}")
print(f"  Mean - Median = {mean_mc:.2f} - {median_mc:.2f} = {gap_mc:.4f}")
if skew_mc > 0 and gap_mc > 0:
    print("  Both positive — they agree on mild right skew.")
elif skew_mc < 0 and gap_mc < 0:
    print("  Both negative — they agree on mild left skew.")
else:
    print("  Signs DISAGREE — investigate further.")

print()
print("Neither pair disagrees, so nothing extra needed investigating for sign.")
print("Both skews are small. The histograms show the mean–median rule is")
print("only a rough guide here: MonthlyCharges has two peaks, which a")
print("single skew value cannot capture.")'''
    ),
    md(
        """---
## 3. Verifying the Standard Deviation by Hand
---"""
    ),
    code(
        '''# Cell 8 — Manual standard deviation: tenure (first 10 rows)
subset_t = x1.head(10)
n = len(subset_t)
subset_mean_t = subset_t.sum() / n

print("=== Manual SD verification: tenure (first 10 rows) ===")
print(f"Values: {[int(v) for v in subset_t]}")
print(f"Step 1 — Mean = {subset_t.sum():.2f} / {n} = {subset_mean_t:.2f}")
print()
print("Step 2 — Squared deviations:")
print(f"{'i':<4} {'x':<8} {'x - mean':<12} {'(x - mean)^2':<14}")
sum_sq_t = 0
for i, val in enumerate(subset_t, start=1):
    dev = val - subset_mean_t
    sq_dev = dev ** 2
    sum_sq_t = sum_sq_t + sq_dev
    print(f"{i:<4} {val:<8} {dev:<12.2f} {sq_dev:<14.4f}")

print()
print(f"Step 3 — Sum of squared deviations = {sum_sq_t:.4f}")
variance_t = sum_sq_t / (n - 1)
print(f"Step 4 — Sample variance = {sum_sq_t:.4f} / {n - 1} = {variance_t:.4f}")
manual_std_t = variance_t ** 0.5
print(f"Step 5 — Manual SD = sqrt({variance_t:.4f}) = {manual_std_t:.4f}")
print()
pandas_std_t = subset_t.std()
print(f"pandas .std() = {pandas_std_t:.4f}")
print(f"Manual  std   = {manual_std_t:.4f}")
print(f"Match: {np.isclose(manual_std_t, pandas_std_t)}")'''
    ),
    code(
        '''# Cell 9 — Manual standard deviation: MonthlyCharges (first 10 rows)
subset_mc = x2.head(10)
n_mc = len(subset_mc)
subset_mean_mc = subset_mc.sum() / n_mc

print("=== Manual SD verification: MonthlyCharges (first 10 rows) ===")
print(f"Values: {[float(v) for v in subset_mc]}")
print(f"Step 1 — Mean = {subset_mc.sum():.2f} / {n_mc} = {subset_mean_mc:.4f}")
print()
print("Step 2 — Squared deviations:")
print(f"{'i':<4} {'x':<10} {'x - mean':<12} {'(x - mean)^2':<14}")
sum_sq_mc = 0
for i, val in enumerate(subset_mc, start=1):
    dev = val - subset_mean_mc
    sq_dev = dev ** 2
    sum_sq_mc = sum_sq_mc + sq_dev
    print(f"{i:<4} {val:<10.2f} {dev:<12.2f} {sq_dev:<14.4f}")

print()
print(f"Step 3 — Sum of squared deviations = {sum_sq_mc:.4f}")
variance_mc = sum_sq_mc / (n_mc - 1)
print(f"Step 4 — Sample variance = {sum_sq_mc:.4f} / {n_mc - 1} = {variance_mc:.4f}")
manual_std_mc = variance_mc ** 0.5
print(f"Step 5 — Manual SD = sqrt({variance_mc:.4f}) = {manual_std_mc:.4f}")
print()
pandas_std_mc = subset_mc.std()
print(f"pandas .std() = {pandas_std_mc:.4f}")
print(f"Manual  std   = {manual_std_mc:.4f}")
print(f"Match: {np.isclose(manual_std_mc, pandas_std_mc)}")
print()
print("The match confirms that .std() is a sample standard deviation,")
print("dividing by n − 1 and not by n.")'''
    ),
    md(
        """---
## 4. Histogram Bin Widths
---"""
    ),
    code(
        '''# Cell 10 — Histogram of MonthlyCharges at three bin widths ($2, $5, $10)
min_val = x2.min()
max_val = x2.max()
bins_2 = np.arange(np.floor(min_val), max_val + 2, 2)
bins_5 = np.arange(np.floor(min_val), max_val + 5, 5)
bins_10 = np.arange(np.floor(min_val), max_val + 10, 10)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(x2, bins=bins_2, color="steelblue", edgecolor="black")
axes[0].set_title(f"MonthlyCharges — $2 bin width ({len(bins_2) - 1} bins)")
axes[0].set_xlabel("Monthly Charges ($)")
axes[0].set_ylabel("Frequency")

axes[1].hist(x2, bins=bins_5, color="coral", edgecolor="black")
axes[1].set_title(f"MonthlyCharges — $5 bin width ({len(bins_5) - 1} bins)")
axes[1].set_xlabel("Monthly Charges ($)")
axes[1].set_ylabel("Frequency")

axes[2].hist(x2, bins=bins_10, color="mediumseagreen", edgecolor="black")
axes[2].set_title(f"MonthlyCharges — $10 bin width ({len(bins_10) - 1} bins)")
axes[2].set_xlabel("Monthly Charges ($)")
axes[2].set_ylabel("Frequency")
plt.tight_layout()
plt.show()

counts_10, _ = np.histogram(x2, bins=bins_10)
print()
print("Bin width choice:")
print(f"  I would report the $5 bin width ({len(bins_5) - 1} bins).")
print("  At $2 the histogram is jagged — counts bounce from bar to bar, so some shape is noise.")
print(
    f"  At $10 there are only {len(bins_10) - 1} bars, and the sharp low-charge peak is"
)
print(
    f"  blended into a single first bar of {counts_10[0]} customers, hiding that it"
)
print(f"  sits right at the minimum of ${min_val:.2f}.")
print("  $5 keeps the clear peak at the low end and the broad hump from $70 to $100 without noise.")
print("  The choice matters because bin width controls what the reader sees — the same data can")
print("  look smooth, jagged or unimodal depending on it (Bruce et al., 2020).")'''
    ),
    code(
        '''# Cell 11 — Why the low peak exists: customers with no internet service
no_internet = df[df["InternetService"] == "No"]
print(f"Customers with no internet service: {len(no_internet)}")
print(
    f"Their MonthlyCharges range: ${no_internet['MonthlyCharges'].min():.2f} "
    f"to ${no_internet['MonthlyCharges'].max():.2f}"
)
print("This explains the low peak in the histogram: phone-only plans cluster near $20.")'''
    ),
    md(
        """---
## 5. Boxplots, 1.5 × IQR Fences and Outliers
---"""
    ),
    code(
        '''# Cell 12 — Boxplots of tenure and MonthlyCharges
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].boxplot(
    x1,
    vert=True,
    patch_artist=True,
    tick_labels=["tenure"],
    boxprops=dict(facecolor="lightblue"),
    medianprops=dict(color="red", linewidth=2),
)
axes[0].set_title("Boxplot of tenure")
axes[0].set_ylabel("Months")

axes[1].boxplot(
    x2,
    vert=True,
    patch_artist=True,
    tick_labels=["MonthlyCharges"],
    boxprops=dict(facecolor="lightyellow"),
    medianprops=dict(color="red", linewidth=2),
)
axes[1].set_title("Boxplot of MonthlyCharges")
axes[1].set_ylabel("Monthly Charges ($)")
plt.tight_layout()
plt.show()'''
    ),
    code(
        '''# Cell 13 — 1.5 × IQR fences and outlier rows
lower_fence_t = q1_t - 1.5 * iqr_t
upper_fence_t = q3_t + 1.5 * iqr_t
outliers_t = df[(x1 < lower_fence_t) | (x1 > upper_fence_t)]

print("=== 1.5 × IQR Fences: tenure ===")
print(f"  Q1 = {q1_t}, Q3 = {q3_t}, IQR = {iqr_t}")
print(f"  Lower Fence = {q1_t} - 1.5×{iqr_t} = {lower_fence_t:.2f}")
print(f"  Upper Fence = {q3_t} + 1.5×{iqr_t} = {upper_fence_t:.2f}")
print(f"  Outliers: {len(outliers_t)} rows")
print()

lower_fence_mc = q1_mc - 1.5 * iqr_mc
upper_fence_mc = q3_mc + 1.5 * iqr_mc
outliers_mc = df[(x2 < lower_fence_mc) | (x2 > upper_fence_mc)]

print("=== 1.5 × IQR Fences: MonthlyCharges ===")
print(f"  Q1 = {q1_mc:.2f}, Q3 = {q3_mc:.2f}, IQR = {iqr_mc:.2f}")
print(f"  Lower Fence = {q1_mc:.2f} - 1.5×{iqr_mc:.2f} = {lower_fence_mc:.2f}")
print(f"  Upper Fence = {q3_mc:.2f} + 1.5×{iqr_mc:.2f} = {upper_fence_mc:.2f}")
print(f"  Outliers: {len(outliers_mc)} rows")
print()
print("All values for both variables lie inside the fences, so the rule flags zero rows.")
print("This is unsurprising: both variables are bounded (tenure by 0 and 72, charges")
print("by the plan prices) and have wide IQRs.")'''
    ),
    code(
        '''# Cell 14 — Extreme values, since the IQR rule found nothing
tenure_zero = df[df["tenure"] == 0]
print(f"(a) Tenure = 0 rows: {len(tenure_zero)}")
print(tenure_zero[["customerID", "tenure", "TotalCharges", "Contract", "Churn"]].to_string(index=False))
print("    These have blank TotalCharges (NaN after conversion).")
print("    That is consistent with customers who just signed up and had not yet been billed.")
print("    Judgment: LEGITIMATE OBSERVATION, not a recording error.")
print()

tenure_72 = df[df["tenure"] == 72]
print(f"(b) Tenure = 72 rows: {len(tenure_72)}")
print(f"    Most common contract type: {tenure_72['Contract'].value_counts().idxmax()}")
print("    Looks like the edge of the observation window — legitimate,")
print("    but 72 is a ceiling, not the true length of every long relationship.")
print()

max_mc = x2.max()
max_mc_row = df[df["MonthlyCharges"] == max_mc]
print(f"(c) Maximum MonthlyCharges = ${max_mc:.2f}")
print(max_mc_row[["customerID", "MonthlyCharges", "InternetService", "Contract"]].to_string(index=False))
print("    Fibre-optic customer on a two-year contract — plausible for a fully bundled plan.")
print("    Judgment: LEGITIMATE EXTREME OBSERVATION.")
print()
print("Whether a charge reflects a billing error cannot be determined without")
print("the billing records, but nothing in the other fields contradicts these values.")'''
    ),
    md(
        """---
## 6. Z-Scores and the Empirical Rule
---"""
    ),
    code(
        '''# Cell 15 — Z-scores for tenure (using tenure's own mean and sample SD)
z_tenure = (x1 - mean_t) / std_t
total = len(z_tenure)
pct_1sd_t = (z_tenure.abs() <= 1).mean() * 100
pct_2sd_t = (z_tenure.abs() <= 2).mean() * 100
pct_3sd_t = (z_tenure.abs() <= 3).mean() * 100
within_1sd_t = int((z_tenure.abs() <= 1).sum())
within_2sd_t = int((z_tenure.abs() <= 2).sum())
within_3sd_t = int((z_tenure.abs() <= 3).sum())

print("=== Z-Score Analysis: tenure ===")
print(f"  Within ±1 SD: {within_1sd_t}/{total} = {pct_1sd_t:.2f}%  (Normal: 68%)")
print(f"  Within ±2 SD: {within_2sd_t}/{total} = {pct_2sd_t:.2f}%  (Normal: 95%)")
print(f"  Within ±3 SD: {within_3sd_t}/{total} = {pct_3sd_t:.2f}%  (Normal: 99.7%)")
print(f"  Min z        = {z_tenure.min():.2f}")
print(f"  Max z        = {z_tenure.max():.2f}")
print(f"  Max |z|      = {z_tenure.abs().max():.2f}")'''
    ),
    code(
        '''# Cell 16 — Z-scores for MonthlyCharges (using MonthlyCharges' own mean and sample SD)
z_mc = (x2 - mean_mc) / std_mc
total_mc = len(z_mc)
pct_1sd_mc = (z_mc.abs() <= 1).mean() * 100
pct_2sd_mc = (z_mc.abs() <= 2).mean() * 100
pct_3sd_mc = (z_mc.abs() <= 3).mean() * 100
within_1sd_mc = int((z_mc.abs() <= 1).sum())
within_2sd_mc = int((z_mc.abs() <= 2).sum())
within_3sd_mc = int((z_mc.abs() <= 3).sum())

print("=== Z-Score Analysis: MonthlyCharges ===")
print(f"  Within ±1 SD: {within_1sd_mc}/{total_mc} = {pct_1sd_mc:.2f}%  (Normal: 68%)")
print(f"  Within ±2 SD: {within_2sd_mc}/{total_mc} = {pct_2sd_mc:.2f}%  (Normal: 95%)")
print(f"  Within ±3 SD: {within_3sd_mc}/{total_mc} = {pct_3sd_mc:.2f}%  (Normal: 99.7%)")
print(f"  Min z        = {z_mc.min():.2f}")
print(f"  Max z        = {z_mc.max():.2f}")
print(f"  Max |z|      = {z_mc.abs().max():.2f}")'''
    ),
    code(
        '''# Cell 17 — Confirm standardisation: z-scores have mean 0 and SD 1
print("=== Standardised variables ===")
z_check = pd.DataFrame({"tenure_z": z_tenure, "MonthlyCharges_z": z_mc})
print(z_check.agg(["mean", "std"]).round(6))
print()
print("Mean is 0 and SD is 1 (up to rounding), as expected after standardisation.")'''
    ),
    code(
        '''# Cell 18 — Empirical rule comparison and normality conclusion
print("=== Empirical Rule Comparison ===")
print(f"{'Variable':<20} {'±1 SD':<12} {'±2 SD':<12} {'±3 SD':<12} {'Max |z|':<10}")
print(f"{'tenure':<20} {pct_1sd_t:<12.2f} {pct_2sd_t:<12.2f} {pct_3sd_t:<12.2f} {z_tenure.abs().max():<10.2f}")
print(f"{'MonthlyCharges':<20} {pct_1sd_mc:<12.2f} {pct_2sd_mc:<12.2f} {pct_3sd_mc:<12.2f} {z_mc.abs().max():<10.2f}")
print(f"{'Normal benchmark':<20} {'68':<12} {'95':<12} {'99.7':<12} {'—':<10}")
print()
print("Conclusion:")
print("  Neither variable follows the 68–95–99.7 rule.")
print(
    f"  Only about {pct_1sd_t:.0f}–{pct_1sd_mc:.0f}% of values fall within one SD (well under 68%),"
)
print("  while 100% fall within two SD (above 95%).")
print("  The data are spread more evenly across their range, with heavier shoulders")
print("  and no long tails, than a bell curve would produce.")
print("  tenure is close to flat with spikes at both ends, and MonthlyCharges")
print("  is bimodal. Neither variable is approximately normal.")'''
    ),
    code(
        '''# Cell 19 — Normal Q–Q plots as a second check on normality
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
stats.probplot(x1, dist="norm", plot=axes[0])
axes[0].set_title("Normal Q-Q plot: tenure")
stats.probplot(x2, dist="norm", plot=axes[1])
axes[1].set_title("Normal Q-Q plot: MonthlyCharges")
plt.tight_layout()
plt.show()
print("Both Q–Q plots bend away from the straight line, especially in the tails")
print("and (for MonthlyCharges) around the two clusters. That matches the")
print("histograms and the empirical-rule counts: neither variable is normal.")'''
    ),
    code(
        '''# Cell 20 — Why standardisation puts all variables on a common scale
print("=== Why standardisation puts all variables on a common scale ===")
print()
print("A z-score rewrites each value as how many of its own standard deviations")
print("it sits from its own mean, so every variable ends up with mean 0, SD 1")
print("and no units. Distances on tenure (months) and MonthlyCharges (dollars)")
print("can then be compared on the same scale.")
print()
print("Without that step, a variable measured in large numbers (dollars) would")
print("dominate one measured in smaller numbers (months) when distances are")
print("calculated. Standardising first prevents that. Peck and Olsen (2025)")
print("treat this as the usual reason to work with z-scores before comparing")
print("variables that were recorded in different units.")'''
    ),
    md(
        """---
## 7. Comparative Display: MonthlyCharges by Contract Type
---"""
    ),
    code(
        '''# Cell 21 — Side-by-side boxplots of MonthlyCharges across Contract types
order = ["Month-to-month", "One year", "Two year"]
groups = [df.loc[df["Contract"] == name, "MonthlyCharges"] for name in order]

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(
    groups,
    tick_labels=order,
    patch_artist=True,
    medianprops=dict(color="red", linewidth=2),
)
colors = ["#FF9999", "#99CCFF", "#99FF99"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax.set_title("MonthlyCharges by Contract Type")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Monthly Charges ($)")
plt.show()'''
    ),
    code(
        '''# Cell 22 — Summary statistics by Contract type and description
print("=== MonthlyCharges by Contract Type ===")
print(f"{'Contract':<20} {'n':<8} {'Mean':<10} {'Median':<10} {'SD':<10} {'IQR':<10}")

contract_stats = {}
for name, data in zip(order, groups):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    contract_stats[name] = {
        "n": len(data),
        "mean": data.mean(),
        "median": data.median(),
        "sd": data.std(),
        "iqr": iqr,
    }
    print(
        f"{name:<20} {len(data):<8} {data.mean():<10.2f} {data.median():<10.2f} "
        f"{data.std():<10.2f} {iqr:<10.2f}"
    )

m2m = contract_stats["Month-to-month"]
y1 = contract_stats["One year"]
y2 = contract_stats["Two year"]
print()
print("Description:")
print("  The median monthly charge falls as the contract gets longer:")
print(
    f"  ${m2m['median']:.2f} for month-to-month, ${y1['median']:.2f} for one year, "
    f"${y2['median']:.2f} for two years."
)
print(
    f"  Month-to-month customers also have the narrowest spread (IQR ${m2m['iqr']:.2f}),"
)
print("  so their charges are more concentrated at the higher end, while the longer")
print(
    f"  contracts include many more low-charge customers (IQR about ${y1['iqr']:.0f}–${y2['iqr']:.0f})."
)
print("  The three boxes overlap heavily, and the means differ less than the medians,")
print("  so the contract association with charges is modest.")
print("  This is an association only — it does not show that contract type causes")
print("  the charge, since customers choose both their services and their contract.")'''
    ),
    md(
        """---
## 8. Conclusion

Both variables are wide and only mildly skewed, in the direction the mean–median gap predicts. The 1.5 × IQR rule finds no outliers, and neither variable is approximately normal: far fewer than 68% of values lie within one standard deviation, every value lies within two, and the Q–Q plots leave the straight line. Medians and IQRs should be reported alongside means. The bimodal shape of MonthlyCharges is worth modelling later by internet-service type.

---

## References (APA 7th Edition)

BlastChar. (n.d.). *Telco customer churn* [Data set]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Bruce, P., Bruce, A., & Gedeck, P. (2020). *Practical statistics for data scientists: 50+ essential concepts using R and Python* (2nd ed.). O'Reilly Media.

Peck, R., & Olsen, C. (2025). *Introduction to statistics and data analysis* (7th ed.). Cengage Learning."""
    ),
]


def fig_to_png_b64() -> list[str]:
    images = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        images.append(base64.b64encode(buf.read()).decode("ascii"))
    plt.close("all")
    return images


def make_stream(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True) or [text]}


def make_result(obj) -> dict:
    text = repr(obj) if not isinstance(obj, (pd.DataFrame, pd.Series)) else obj.__repr__()
    data = {"text/plain": text.splitlines(keepends=True) or [text]}
    if isinstance(obj, pd.DataFrame):
        try:
            data["text/html"] = [obj.to_html()]
        except Exception:
            pass
    return {
        "output_type": "execute_result",
        "metadata": {},
        "data": data,
        "execution_count": None,
    }


def make_image(b64: str) -> dict:
    return {
        "output_type": "display_data",
        "metadata": {},
        "data": {"image/png": b64, "text/plain": ["<Figure>"]},
    }


def execute_notebook(nb: dict) -> dict:
    env = {
        "__name__": "__main__",
        "pd": pd,
        "np": np,
        "plt": plt,
        "stats": stats,
    }
    exec_count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec_count += 1
        src = "".join(cell["source"])
        stdout = io.StringIO()
        result = None
        err = None
        plt.close("all")
        try:
            with contextlib.redirect_stdout(stdout):
                # last expression display, similar to IPython
                block = compile(src, f"<cell{exec_count}>", "exec")
                exec(block, env)
                # try to display last line if it is an expression
                lines = [ln for ln in src.strip().split("\n") if ln.strip() and not ln.strip().startswith("#")]
                if lines:
                    last = lines[-1]
                    if not last.strip().startswith(
                        ("print", "for ", "if ", "else", "elif", "def ", "class ", "with ", "while ", "import ", "from ")
                    ) and "=" not in last.split("#")[0] or (
                        "=" in last
                        and last.strip().split("=")[0].strip() in {"summary"}
                        and last.strip().endswith(".round(3)")
                    ):
                        try:
                            result = eval(last, env)
                        except Exception:
                            result = None
                    if last.strip() == "summary.round(3)":
                        result = eval("summary.round(3)", env)
                    if last.strip().endswith(".head(10)"):
                        result = eval(last.strip(), env)
        except Exception:
            err = traceback.format_exc()
        outputs = []
        text = stdout.getvalue()
        if text:
            outputs.append(make_stream(text))
        images = fig_to_png_b64()
        for b64 in images:
            outputs.append(make_image(b64))
        if err:
            outputs.append(
                {
                    "output_type": "error",
                    "ename": "ExecutionError",
                    "evalue": err.splitlines()[-1],
                    "traceback": err.splitlines(),
                }
            )
        elif result is not None:
            out = make_result(result)
            out["execution_count"] = exec_count
            outputs.append(out)
        cell["outputs"] = outputs
        cell["execution_count"] = exec_count
        if err:
            raise RuntimeError(f"Cell {exec_count} failed:\n{err}")
    return env


def verify(env: dict) -> list[str]:
    issues = []
    df = env["df"]
    x1 = env["x1"]
    x2 = env["x2"]
    if len(df) != 7043:
        issues.append(f"n is {len(df)}, expected 7043")
    if int(x1.isna().sum()) != 0 or int(x2.isna().sum()) != 0:
        issues.append("unexpected missing values in tenure/MonthlyCharges")
    if int((df["tenure"] == 0).sum()) != 11:
        issues.append("tenure==0 rows were dropped")
    # stats consistency
    if abs(env["mean_t"] - x1.mean()) > 1e-12:
        issues.append("mean_t does not match x1.mean()")
    if abs(env["mean_mc"] - x2.mean()) > 1e-12:
        issues.append("mean_mc does not match x2.mean()")
    if abs(env["skew_t"] - x1.skew()) > 1e-12:
        issues.append("skew_t mismatch")
    if abs(env["gap_t"] - (env["mean_t"] - env["median_t"])) > 1e-12:
        issues.append("gap_t mismatch")
    # fences
    if len(env["outliers_t"]) != 0 or len(env["outliers_mc"]) != 0:
        issues.append("IQR outliers should be 0")
    # manual std
    if not bool(np.isclose(env["manual_std_t"], env["pandas_std_t"])):
        issues.append("manual tenure SD does not match pandas")
    if not bool(np.isclose(env["manual_std_mc"], env["pandas_std_mc"])):
        issues.append("manual MonthlyCharges SD does not match pandas")
    # z scores use own mean/sd
    z_t = env["z_tenure"]
    z_mc = env["z_mc"]
    recon_t = (x1 - env["mean_t"]) / env["std_t"]
    recon_mc = (x2 - env["mean_mc"]) / env["std_mc"]
    if not np.allclose(z_t, recon_t):
        issues.append("z_tenure mixed sample")
    if not np.allclose(z_mc, recon_mc):
        issues.append("z_mc mixed sample")
    if abs(z_t.mean()) > 1e-10 or abs(z_t.std() - 1) > 1e-10:
        issues.append(f"tenure z mean/sd not 0/1: {z_t.mean()}, {z_t.std()}")
    if abs(z_mc.mean()) > 1e-10 or abs(z_mc.std() - 1) > 1e-10:
        issues.append(f"mc z mean/sd not 0/1: {z_mc.mean()}, {z_mc.std()}")
    if env["pct_2sd_t"] != 100.0 or env["pct_2sd_mc"] != 100.0:
        issues.append("expected 100% within 2 SD")
    if env["pct_1sd_t"] >= 68 or env["pct_1sd_mc"] >= 68:
        issues.append("±1 SD unexpectedly near normal")
    # contract medians match description source
    y2_med = env["contract_stats"]["Two year"]["median"]
    one_med = env["contract_stats"]["One year"]["median"]
    m2m_med = env["contract_stats"]["Month-to-month"]["median"]
    raw_m2m = df.loc[df["Contract"] == "Month-to-month", "MonthlyCharges"].median()
    raw_y1 = df.loc[df["Contract"] == "One year", "MonthlyCharges"].median()
    raw_y2 = df.loc[df["Contract"] == "Two year", "MonthlyCharges"].median()
    if abs(y2_med - raw_y2) > 1e-12 or abs(m2m_med - raw_m2m) > 1e-12 or abs(one_med - raw_y1) > 1e-12:
        issues.append("contract medians do not match df")
    # known values for n=7043
    if abs(x1.mean() - 32.371149) > 0.001:
        issues.append(f"unexpected tenure mean {x1.mean()}")
    if abs(x2.mean() - 64.761692) > 0.001:
        issues.append(f"unexpected MonthlyCharges mean {x2.mean()}")
    if abs(y2_med - 64.35) > 0.001:
        issues.append(f"two-year median should be 64.35 on full data, got {y2_med}")
    n_sum = sum(s["n"] for s in env["contract_stats"].values())
    if n_sum != 7043:
        issues.append(f"contract n sum {n_sum} != 7043")
    # notebook outputs: no errors, plots present
    return issues


def count_images(nb: dict) -> int:
    n = 0
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "display_data" and "image/png" in out.get("data", {}):
                n += 1
    return n


def main() -> int:
    if not CSV_PATH.exists():
        print("CSV missing", CSV_PATH)
        return 1
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Aashishchand_KDD_Env",
                "language": "python",
                "name": "aashishchand_kdd_env",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.25",
            },
        },
        "cells": CELLS,
    }
    print("Executing notebook cells...")
    env = execute_notebook(nb)
    issues = verify(env)
    n_img = count_images(nb)
    n_err = sum(
        1
        for c in nb["cells"]
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    )
    if n_img < 5:
        issues.append(f"expected at least 5 figures, got {n_img}")
    if n_err:
        issues.append(f"{n_err} error outputs")

    # Cross-check printed narrative against computed values
    src_all = "\n".join("".join(c["source"]) for c in nb["cells"])
    if "df_clean" in src_all:
        issues.append("df_clean still present")
    if "32.4" in src_all and "mean_t" not in src_all.split("32.4")[0][-40:]:
        pass  # f-strings only should remain

    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {NB_PATH}")
    print(f"Figures captured: {n_img}")
    print("Key results:")
    print(f"  n = {len(env['df'])}")
    print(f"  tenure mean/median/sd/iqr/skew = {env['mean_t']:.4f}/{env['median_t']:.2f}/{env['std_t']:.4f}/{env['iqr_t']:.2f}/{env['skew_t']:.4f}")
    print(f"  MonthlyCharges mean/median/sd/iqr/skew = {env['mean_mc']:.4f}/{env['median_mc']:.2f}/{env['std_mc']:.4f}/{env['iqr_mc']:.2f}/{env['skew_mc']:.4f}")
    print(f"  tenure ±1/±2/±3 = {env['pct_1sd_t']:.2f}/{env['pct_2sd_t']:.2f}/{env['pct_3sd_t']:.2f}")
    print(f"  MonthlyCharges ±1/±2/±3 = {env['pct_1sd_mc']:.2f}/{env['pct_2sd_mc']:.2f}/{env['pct_3sd_mc']:.2f}")
    print(f"  IQR outliers tenure/mc = {len(env['outliers_t'])}/{len(env['outliers_mc'])}")
    print(f"  tenure=0 kept = {(env['df']['tenure']==0).sum()}")
    print(f"  two-year median = {env['contract_stats']['Two year']['median']}")
    print(f"  manual SD match tenure/mc = {np.isclose(env['manual_std_t'], env['pandas_std_t'])}, {np.isclose(env['manual_std_mc'], env['pandas_std_mc'])}")
    if issues:
        print("VERIFICATION FAILED:")
        for i in issues:
            print(" -", i)
        return 1
    print("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
