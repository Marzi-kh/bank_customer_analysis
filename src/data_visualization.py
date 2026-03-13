import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.load_file import load_data

sns.set_theme(style="whitegrid")

TARGET = "deposit"
AGE_COL = "age"
JOB_COL = "job"
POUTCOME_COL = "poutcome"


def save_plot(name):
    project_root = Path(__file__).resolve().parents[1]
    out = project_root / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out / name, dpi=250, bbox_inches="tight")
    plt.close()


def deposit_to_bin(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().map({"yes": 1, "no": 0})


def annotate_bar_values(ax, fmt="{:.1f}%"):

    if ax.patches and ax.patches[0].get_width() > ax.patches[0].get_height():  # barh
        for p in ax.patches:
            w = p.get_width()
            ax.text(w + 0.3, p.get_y() + p.get_height() / 2, fmt.format(w), va="center")
    else:
        for p in ax.patches:
            h = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2, h + 0.8, fmt.format(h), ha="center")


# -------------------- PLOT 1: Target (%) --------------------
def plot_deposit_rate_pct(df):
    plt.figure(figsize=(5.5, 4))
    y = deposit_to_bin(df[TARGET].dropna()).dropna()

    pct = (y.value_counts(normalize=True).sort_index() * 100)
    ax = pct.plot(kind="bar")
    ax.set_title("Customer Subscription Rate (%)")
    ax.set_xlabel("Deposit (No, Yes)")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, max(10, pct.max() + 10))
    ax.set_xticklabels(["No", "Yes"], rotation=45)
    annotate_bar_values(ax)
    save_plot("01_deposit_rate_pct.png")


# -------------------- PLOT 2: Subscription by age group (%) --------------------
def plot_deposit_rate_by_age_group(df):
    plt.figure(figsize=(7.5, 4))
    d = df[[AGE_COL, TARGET]].dropna().copy()
    d["deposit_bin"] = deposit_to_bin(d[TARGET])
    d = d.dropna(subset=["deposit_bin"])

    d["age_group"] = pd.cut(d[AGE_COL], bins=[18, 30, 40, 50, 60, 70, 100], right=False)
    rate = d.groupby("age_group")["deposit_bin"].mean().mul(100)

    ax = rate.plot(kind="bar")
    ax.set_title("Subscription Rate by Age Group (%)")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Subscription rate (%)")
    ax.set_ylim(0, max(10, rate.max() + 10))
    plt.xticks(rotation=0)
    annotate_bar_values(ax)
    save_plot("02_subscription_by_age_group.png")


# -------------------- PLOT 3: Top jobs (%) (readable) --------------------
def plot_top_jobs_pct(df, top_n=10):
    plt.figure(figsize=(7.5, 4.2))
    s = df[JOB_COL].dropna().astype(str)

    pct = (s.value_counts(normalize=True) * 100).head(top_n).sort_values()
    ax = pct.plot(kind="barh")
    ax.set_title(f"Top {top_n} Jobs (% of customers)")
    ax.set_xlabel("Percentage")
    ax.set_ylabel("Job")
    ax.set_xlim(0, max(10, pct.max() + 5))
    annotate_bar_values(ax)
    save_plot("03_top_jobs_pct.png")


# -------------------- PLOT : poutcome vs subscription --------------------
def plot_subscription_by_poutcome(df):

    plt.figure(figsize=(7.5, 4.2))
    d = df[[POUTCOME_COL, TARGET]].dropna().copy()
    d["deposit_bin"] = deposit_to_bin(d[TARGET])
    d = d.dropna(subset=["deposit_bin"])

    rate = d.groupby(POUTCOME_COL)["deposit_bin"].mean().mul(100).sort_values(ascending=True)
    ax = rate.plot(kind="barh")
    ax.set_title("Subscription Rate by Previous Campaign Outcome (%)")
    ax.set_xlabel("Subscription rate (%)")
    ax.set_ylabel("Previous outcome (poutcome)")
    ax.set_xlim(0, max(10, rate.max() + 5))
    annotate_bar_values(ax)
    save_plot("04_subscription_by_poutcome.png")




def main():
    df = load_data()

    plot_deposit_rate_pct(df)
    plot_deposit_rate_by_age_group(df)
    plot_top_jobs_pct(df)
    plot_subscription_by_poutcome(df)


if __name__ == "__main__":
    main()