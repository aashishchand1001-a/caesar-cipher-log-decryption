# AIM 402 — Statistics for Data Science Using Python
## Assignments · International American University

**Student:** Aashish Chand  
**Professor:** Poshan Karki  

---

## Repository Contents

| File | Description |
|------|-------------|
| `decrypt_logs.py` | **Week 1** — Caesar cipher log decryption program |
| `raw_logs.txt` | Encrypted log input for Week 1 |
| `decrypted_master.txt` | Decrypted output from Week 1 |
| `security_alerts.txt` | Filtered security alerts from Week 1 |
| `Week-1-Assignment.ipynb` | Week 1 Jupyter notebook submission |
| `data.py` | Shared data helpers |
| `Week-2-Assignment.ipynb` | **Week 2** — Describe Your Data (Telco Churn dataset) |
| `_build_week2.py` | Build script that generates and verifies the Week 2 notebook |
| `Assignment2_Describe_Your_Data.ipynb` | Week 2 annotated assignment notebook |
| `Telco-Customer-Churn.csv` | IBM Telco Customer Churn dataset (n = 7,043) |
| `Week-3-Assignment.ipynb` | **Week 3** — Assignment notebook |
| `Week-3-Python-Assignment.ipynb` | Week 3 Python assignment notebook |
| `class-Activity.ipynb` | In-class activity notebook |

---

## Week 1 — Caesar Cipher Log Decryption

A Python script (`decrypt_logs.py`) that:
- Reads encrypted log entries from `raw_logs.txt`
- Applies a Caesar cipher decryption with key shift
- Writes all decrypted entries to `decrypted_master.txt`
- Filters and saves security-related alerts to `security_alerts.txt`

**Concepts used:** `dict`, `list`, `tuple`, loops, `if/elif/else`, `.get()`, `.append()`

---

## Week 2 — Describe Your Data (AIM 402)

Analysis of the IBM Telco Customer Churn dataset using descriptive statistics and visualizations.

**Dataset:** `Telco-Customer-Churn.csv`  
**Variables analyzed:**
- `tenure` — months as a customer (numerical)
- `MonthlyCharges` — monthly bill in USD (numerical)
- `Contract` — contract type (categorical, for group comparison)

**Key findings:**
- `tenure` is mildly right-skewed with a bimodal distribution (spikes at 0–1 months and at the 72-month ceiling)
- `MonthlyCharges` is roughly uniform with a slight left skew
- Month-to-month customers have significantly higher churn rates

**Build script:** `_build_week2.py` — programmatically builds, executes, and verifies `Week-2-Assignment.ipynb`

---

## Setup

```bash
# Install dependencies
pip install pandas numpy matplotlib scipy jupyter

# Run the Week 2 build script
python _build_week2.py

# Launch Jupyter
jupyter notebook
```

---

## References

- BlastChar (2019). *Telco Customer Churn* [Dataset]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Behrman, K. (2022). *Foundational Python for data science* (1st ed.). Pearson.
- Lambert, K. A. (2023). *Fundamentals of Python: First programs* (3rd ed.). Cengage Learning.
- Python Software Foundation. (n.d.). *Data structures*. https://docs.python.org/3/tutorial/datastructures.html
- VanderPlas, J. (2023). *Python data science handbook* (2nd ed.). O'Reilly Media.
