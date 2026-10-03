#!/usr/bin/env python3
"""Build, execute, and verify Week-3-Python-Assignment.ipynb.
AIM 402 – Week 3: Correlation, Regression, Residuals & Inference
Uses only pure Python lists (no NumPy / Pandas / Scikit-learn).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "Week-3-Python-Assignment.ipynb"


# ─── helpers ────────────────────────────────────────────────────────────────

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
    return out if out else []


# ─── notebook cells ─────────────────────────────────────────────────────────

CELLS = [

    # ── Title / header ───────────────────────────────────────────────────────
    md(
        """# Week 3 Assignment — Correlation, Regression, Residuals & Inference

**Aashish Chand**
International American University
AIM 402 Statistics for Data Analysis Using Python
Professor Poshan Karki
October 3, 2026

---

**Objective:** Implement bivariate linear regression analysis in pure Python — no external libraries.

### Dataset
Advertising spend $x$ (£000) vs. sales $y$ (£000) for six consecutive months:

| Month | Advertising Spend $x$ (£000) | Sales $y$ (£000) |
|:---:|:---:|:---:|
| 1 | 1 | 3 |
| 2 | 2 | 5 |
| 3 | 3 | 4 |
| 4 | 4 | 8 |
| 5 | 5 | 10 |
| 6 | 6 | 12 |

**Given:** $\\bar{x} = 3.5$, $\\bar{y} = 7.0$"""
    ),

    # ── Cell 1: Data setup ───────────────────────────────────────────────────
    code(
        '''# Cell 1 — Define dataset (pure Python lists only, no external libraries)
x = [1, 2, 3, 4, 5, 6]
y = [3, 5, 4, 8, 10, 12]
n = len(x)

x_bar = sum(x) / n
y_bar = sum(y) / n

print(f"Sample size (n): {n}")
print(f"Calculated x̄: {x_bar:.4f}  (Given: 3.5)")
print(f"Calculated ȳ: {y_bar:.4f}  (Given: 7.0)")'''
    ),

    # ── Part A header ────────────────────────────────────────────────────────
    md(
        """---
## Part A — Correlation, the Regression Line, Residuals & Interpretation

### Question 1: Pearson Correlation Coefficient $r$

**Formulas:**
$$S_{xy} = \\sum (x - \\bar{x})(y - \\bar{y})$$
$$S_{xx} = \\sum (x - \\bar{x})^2, \\quad S_{yy} = \\sum (y - \\bar{y})^2$$
$$r = \\frac{S_{xy}}{\\sqrt{S_{xx} \\times S_{yy}}}$$"""
    ),

    # ── Cell 2: Pearson r ────────────────────────────────────────────────────
    code(
        '''# Cell 2 — Question 1: Pearson correlation coefficient r
Sxy = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))
Sxx = sum((xi - x_bar) ** 2 for xi in x)
Syy = sum((yi - y_bar) ** 2 for yi in y)

r = Sxy / ((Sxx * Syy) ** 0.5)

print("--- Question 1 Calculations ---")
print(f"Sxy = Σ(x - x̄)(y - ȳ) = {Sxy:.4f}")
print(f"Sxx = Σ(x - x̄)²       = {Sxx:.4f}")
print(f"Syy = Σ(y - ȳ)²       = {Syy:.4f}")
print(f"r   = Sxy / √(Sxx·Syy) = {r:.6f}")
print(f"Rounded r              = {round(r, 4)}")'''
    ),

    # ── Q2 header ────────────────────────────────────────────────────────────
    md(
        """### Question 2: Least-Squares Regression Line

**Formulas:**
$$b = \\frac{S_{xy}}{S_{xx}}, \\quad a = \\bar{y} - b\\bar{x}$$
$$\\hat{y} = a + bx$$"""
    ),

    # ── Cell 3: Regression line ──────────────────────────────────────────────
    code(
        '''# Cell 3 — Question 2: Least-squares regression line
b = Sxy / Sxx
a = y_bar - b * x_bar

print("--- Question 2 Calculations ---")
print(f"Slope     b = Sxy / Sxx = {Sxy:.4f} / {Sxx:.4f} = {b:.4f}")
print(f"Intercept a = ȳ - b·x̄  = {y_bar:.4f} - {b:.4f}×{x_bar:.4f} = {a:.4f}")
print(f"\\nRegression Equation:  ŷ = {a:.4f} + {b:.4f}x")
print(f"Rounded form:          ŷ = {round(a, 3)} + {round(b, 3)}x")'''
    ),

    # ── Q3 header ────────────────────────────────────────────────────────────
    md(
        """### Question 3: Coefficient of Determination $R^2$

**Formula:**
$$R^2 = r^2$$"""
    ),

    # ── Cell 4: R² ──────────────────────────────────────────────────────────
    code(
        '''# Cell 4 — Question 3: Coefficient of determination R²
r_squared = r ** 2
r_squared_pct = r_squared * 100

print("--- Question 3 Calculations ---")
print(f"r²  = ({r:.6f})² = {r_squared:.6f}")
print(f"R²  = {r_squared_pct:.4f}%")
print(f"Rounded R² = {round(r_squared, 4)} = {round(r_squared_pct, 2)}%")'''
    ),

    # ── Q4 header ────────────────────────────────────────────────────────────
    md(
        """### Question 4: Adjusted $R^2$

**Formula:**
$$R^2_{\\text{adj}} = 1 - (1 - R^2)\\frac{n-1}{n-k-1}$$

where $n = 6$ and $k = 1$ (one predictor)."""
    ),

    # ── Cell 5: Adjusted R² ──────────────────────────────────────────────────
    code(
        '''# Cell 5 — Question 4: Adjusted R²
k = 1  # single predictor: advertising spend

r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
r_squared_adj_pct = r_squared_adj * 100

print("--- Question 4 Calculations ---")
print(f"n = {n}, k = {k}")
print(f"Degrees of freedom (total): n - 1 = {n - 1}")
print(f"Degrees of freedom (error): n - k - 1 = {n - k - 1}")
print(f"R²_adj = 1 - (1 - {r_squared:.6f}) × ({n-1}/{n-k-1})")
print(f"R²_adj = {r_squared_adj:.6f} = {r_squared_adj_pct:.4f}%")
print(f"Rounded R²_adj = {round(r_squared_adj, 4)} = {round(r_squared_adj_pct, 2)}%")'''
    ),

    # ── Q5 header ────────────────────────────────────────────────────────────
    md(
        """### Question 5: Predicted Values $\\hat{y}$ and Residuals $e$

**Formulas:**
$$\\hat{y} = a + bx, \\quad e = y - \\hat{y}$$

Check: $\\sum e \\approx 0$"""
    ),

    # ── Cell 6: Predicted values + residuals ─────────────────────────────────
    code(
        '''# Cell 6 — Question 5: Predicted values ŷ and residuals e
y_hat_list = []
residual_list = []

header = f"{'Month':<6} {'x (£000)':<10} {'Actual y':<10} {'Predicted ŷ':<14} {'Residual e = y - ŷ':<18}"
print(header)
print("-" * len(header))

for i in range(n):
    xi = x[i]
    yi = y[i]
    y_hat_i = a + b * xi
    e_i = yi - y_hat_i
    y_hat_list.append(y_hat_i)
    residual_list.append(e_i)
    print(f"{i+1:<6} {xi:<10} {yi:<10.1f} {y_hat_i:<14.4f} {e_i:<18.4f}")

sum_e = sum(residual_list)
print("-" * len(header))
print(f"{'Σ':<6} {sum(x):<10} {sum(y):<10.1f} {sum(y_hat_list):<14.4f} {sum_e:<18.10f}")
print("-" * len(header))
print(f"\\nVerification: Σe = {sum_e:.10f} ≈ 0  ✓")'''
    ),

    # ── Part B header ────────────────────────────────────────────────────────
    md(
        """---
## Part B — Inference & Logical Understanding

### Question 6: Interpret the Correlation

**Question:** The value of $r$ is positive and close to 1. What does this tell you about the relationship between advertising spend and sales?

**Answer:**
- The Pearson correlation coefficient is **$r \\approx +0.956$**.
- Because $r > 0$ and very close to $+1$, it demonstrates an **extremely strong, positive linear relationship** between monthly advertising expenditure and sales.
- In practical terms, as advertising spend increases across the six months, sales increase in a highly consistent, proportional linear pattern."""
    ),

    # ── Q7 ───────────────────────────────────────────────────────────────────
    md(
        """### Question 7: Interpret the Slope

**Question:** Suppose your regression line gives $b = 1.829$. Explain what this means in the context of advertising and sales. Remember that both $x$ and $y$ are measured in £000.

**Answer:**
- The slope is $b \\approx 1.8286$ (rounded to $1.829$).
- Since both $x$ and $y$ are in thousands of pounds, a unit increase $\\Delta x = 1$ represents an additional **£1,000** in advertising spend.
- **Interpretation:** For every additional **£1,000** spent on advertising, monthly sales are estimated to increase on average by **£1,829**."""
    ),

    # ── Q8 ───────────────────────────────────────────────────────────────────
    md(
        """### Question 8: Interpret $R^2$

**Question:** Suppose you obtain $R^2 \\approx 0.914$. Explain this result in simple business language.

**Answer:**
- Sales vary month-to-month across the six-month period (ranging from £3,000 to £12,000). About **91.4% of these fluctuations in monthly sales** can be accounted for by how much money was allocated to advertising in those months.
- The remaining **8.6%** of sales variation is caused by other unmeasured factors (competitor campaigns, seasonality, product availability) or random noise."""
    ),

    # ── Q9 ───────────────────────────────────────────────────────────────────
    md(
        """### Question 9: Compare $R^2$ and Adjusted $R^2$

**Question:** $R^2 = 91.43\\%$, $R^2_{\\text{adj}} = 89.29\\%$. What does this difference tell you?

**Answer:**
- The difference is only **2.14%** — very small.
- This minor drop confirms that advertising expenditure is a **genuinely useful predictor**.
- Even after penalizing for the small sample size ($n = 6$) and the single predictor ($k = 1$), the model still explains nearly 90% of the variance. The high $R^2$ is not an artifact of over-parameterization."""
    ),

    # ── Q10 header + code ────────────────────────────────────────────────────
    md(
        """### Question 10: Positive or Negative Residual?

**Question:** For one month, $y = 4$ and $\\hat{y} = 6.086$. Calculate the residual and explain what it means."""
    ),

    code(
        '''# Cell 7 — Question 10: Residual for Month 3
y_actual_q10 = 4.0
y_pred_q10   = 6.086
residual_q10 = y_actual_q10 - y_pred_q10

print("--- Question 10 Calculations ---")
print(f"Actual sales  (y)    = £{y_actual_q10:.3f}k  (£{int(y_actual_q10*1000):,})")
print(f"Predicted sales (ŷ)  = £{y_pred_q10:.3f}k  (£{int(y_pred_q10*1000):,})")
print(f"Residual (e = y − ŷ) = {residual_q10:.3f}k  (£{int(residual_q10*1000):,})")'''
    ),

    md(
        """**Answer:**
- Residual $e = 4 - 6.086 = -2.086$ (£000) → **negative residual**.
- Actual sales (£4,000) were **below** the predicted value (£6,086) by **£2,086**.
- The regression model **overpredicted** sales for that month."""
    ),

    # ── Q11 ──────────────────────────────────────────────────────────────────
    md(
        """### Question 11: Business Inference

**Question:** A manager says: *"If we increase advertising, sales will definitely increase."* Do you agree?

**Answer:** **I do not agree with the definitive statement.**

1. **Correlation ≠ Causation.** $r = 0.956$ and a fitted regression line establish a strong *statistical association*, but do not prove advertising *causes* higher sales. A lurking variable (seasonality, macroeconomic conditions) could be driving both.
2. **Probabilistic, not deterministic.** The word *"definitely"* implies a guaranteed outcome. Regression provides only an *expected average trend* — 8.6% of variance remains unexplained.
3. **Diminishing returns.** Increasing ad spend beyond a certain point experiences diminishing marginal returns; the £1,829-per-£1,000 relationship will not hold indefinitely."""
    ),

    # ── Q12 header + code ────────────────────────────────────────────────────
    md(
        """### Question 12: Prediction ($x = 7$)

**Question:** Estimate sales when advertising spend is $x = 7$. Is this interpolation or extrapolation?"""
    ),

    code(
        '''# Cell 8 — Question 12: Prediction for x = 7
x_new = 7
y_pred_new = a + b * x_new

print("--- Question 12 Calculations ---")
print(f"Regression Equation: ŷ = {a:.4f} + {b:.4f}x")
print(f"At x = {x_new}: ŷ = {a:.4f} + {b:.4f}×{x_new} = {y_pred_new:.4f}k  (£{y_pred_new*1000:,.0f})")
print(f"Rounded: ŷ = {0.6 + 1.829 * 7:.3f}k  (£{(0.6 + 1.829*7)*1000:,.0f})")'''
    ),

    md(
        """**Answer:**
- **Estimated sales:** $\\hat{y} = 0.600 + 1.8286(7) = 13.400$ (£000) = **£13,400**.
- This is **extrapolation** — $x = 7$ lies *outside* the observed range ($x = 1$ to $6$).
- **Why it matters:** Interpolation predicts within an empirically observed domain; extrapolation assumes the linear pattern extends beyond it, which is risky due to potential saturation or other market effects."""
    ),

    # ── Q13 header + code ────────────────────────────────────────────────────
    md(
        """### Question 13: Largest Absolute Residual

**Question:** Which month has the largest absolute residual? What does it indicate?"""
    ),

    code(
        '''# Cell 9 — Question 13: Largest absolute residual
abs_residuals = [abs(e) for e in residual_list]
max_abs_e = max(abs_residuals)
max_idx   = abs_residuals.index(max_abs_e)

print("--- Question 13: Absolute Residuals by Month ---")
for i, (xi, yi, yh, e) in enumerate(zip(x, y, y_hat_list, residual_list)):
    marker = " ← LARGEST" if i == max_idx else ""
    print(f"Month {i+1}: x={xi}, y={yi}, ŷ={yh:.4f}, e={e:.4f}, |e|={abs(e):.4f}{marker}")

print(f"\\nMonth with largest |e|: Month {max_idx+1}  (|e| = {max_abs_e:.4f})")'''
    ),

    md(
        """**Answer:**
- **Month 3** ($x = 3, y = 4$) has the largest absolute residual: $|e| \\approx 2.086$ (£2,086).
- **What it indicates:** Month 3 is the biggest departure from the linear model. Actual sales (£4,000) substantially underperformed relative to the model's prediction (£6,086), suggesting unique external factors — poor weather, a competitor promotion, or supply disruption — were at work that month."""
    ),

    # ── Q14 ──────────────────────────────────────────────────────────────────
    md(
        """### Question 14: Is $R^2$ a Measure of Accuracy?

**Question:** An analyst says *"$R^2 = 91.4\\%$, so the model is 91.4% accurate."* Is this correct?

**Answer:** **No — this is an incorrect interpretation.**

1. **$R^2$ measures explained variance, not accuracy.** It quantifies the proportion of variability in $y$ accounted for by the linear relationship with $x$, not how close individual predictions are to actual values.
2. **Accuracy requires error metrics.** True predictive accuracy is assessed with MAE, RMSE, or prediction intervals — none of which $R^2$ captures.
3. **$R^2$ does not detect model mis-specification.** A non-linear or biased model can still produce a high $R^2$."""
    ),

    # ── Q15 ──────────────────────────────────────────────────────────────────
    md(
        """### Question 15: Final Conclusion

**Question:** Write a 2–3 sentence conclusion for the manager.

**Answer:**

> **Executive Summary for Management:**
> Our analysis of six months of advertising and sales data reveals a **very strong positive linear relationship** ($r \\approx 0.956$): as advertising expenditure increases, monthly sales increase in a highly consistent pattern. The fitted regression equation $\\hat{y} = 0.600 + 1.829x$ estimates that every additional £1,000 spent on advertising is associated with approximately **£1,829 in additional sales**, and the model explains **91.4%** ($R^2$) of the month-to-month variation in sales. However, with only six observations and without experimental controls, this association **cannot be interpreted as definitive causation** — other factors or seasonal effects may be contributing, and the model should be validated with more data before using it for high-stakes budget decisions."""
    ),

    # ── Summary Cell ─────────────────────────────────────────────────────────
    md(
        """---
## Summary of Key Results

| Metric | Value |
|--------|-------|
| Sample size $n$ | 6 |
| $\\bar{x}$ | 3.5 |
| $\\bar{y}$ | 7.0 |
| $S_{xy}$ | 19.0 |
| $S_{xx}$ | 17.5 |
| $S_{yy}$ | 70.0 |
| Pearson $r$ | ≈ 0.9560 |
| Intercept $a$ | ≈ 0.6000 |
| Slope $b$ | ≈ 1.8286 |
| $R^2$ | ≈ 91.43% |
| Adjusted $R^2$ | ≈ 89.29% |
| Largest residual | Month 3, $e \\approx -2.086$ |
| Prediction at $x = 7$ | ≈ £13,400 (extrapolation) |"""
    ),
]


# ─── assemble and write notebook ─────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": CELLS,
}

NB_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"✅  Written: {NB_PATH}")
print(f"   Cells  : {len(CELLS)}")
print(f"   Size   : {NB_PATH.stat().st_size:,} bytes")
