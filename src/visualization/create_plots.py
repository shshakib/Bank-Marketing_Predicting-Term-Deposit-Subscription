"""Generate reusable EDA figures for the bank marketing project."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_deposit_distribution(df, output_dir):
    """Save the target class distribution plot."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="deposit", hue="deposit", palette="Set2", ax=ax)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_title("Distribution of Deposit")
    ax.set_xlabel("Subscription Status")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "deposit_distribution.png", dpi=160)
    plt.close(fig)


def save_numerical_histograms(df, output_dir):
    """Save histograms for all numeric fields to inspect skew and spread."""
    numerical = df.select_dtypes(include="number")
    axes = numerical.hist(figsize=(12, 9), bins=30, color="steelblue")
    for ax in axes.flatten():
        ax.set_title(f"Histogram of {ax.get_title()}")
    plt.tight_layout()
    plt.savefig(output_dir / "numerical_histograms.png", dpi=160)
    plt.close()


def save_correlation_heatmap(df, output_dir):
    """Save a numeric correlation heatmap for quick relationship checks."""
    numerical = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numerical.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)


def save_boxplots(df, output_dir):
    """Save numeric boxplots to make outliers and long tails visible."""
    numerical = df.select_dtypes(include="number")
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes = axes.flatten()
    for idx, column in enumerate(numerical.columns):
        sns.boxplot(y=df[column], color="lightblue", fliersize=3, ax=axes[idx])
        axes[idx].set_title(f"Boxplot of {column}")
        axes[idx].set_xlabel("")
    for idx in range(len(numerical.columns), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "numerical_boxplots.png", dpi=160)
    plt.close(fig)


def save_categorical_proportions(df, output_dir):
    """Save subscription proportions for each categorical predictor."""
    categorical = [column for column in df.select_dtypes(include="object").columns if column != "deposit"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for idx, column in enumerate(categorical):
        proportions = (
            df.groupby(column)["deposit"]
            .value_counts(normalize=True)
            .rename("proportion")
            .reset_index()
        )
        sns.barplot(data=proportions, x=column, y="proportion", hue="deposit", ax=axes[idx])
        axes[idx].set_title(f"Proportion of Deposit by {column}")
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("Proportion")
    for idx in range(len(categorical), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "categorical_deposit_proportions.png", dpi=160)
    plt.close(fig)


def save_age_by_subscription(df, output_dir):
    """Save an age distribution split by subscription outcome."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="age", hue="deposit", binwidth=5, multiple="dodge", ax=ax)
    ax.set_title("Distribution of Age by Subscription")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "age_by_subscription.png", dpi=160)
    plt.close(fig)


def save_campaign_subscription(df, output_dir):
    """Save subscription proportions by campaign contact count."""
    campaign_df = (
        df.groupby("campaign")["deposit"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=campaign_df, x="campaign", y="proportion", hue="deposit", ax=ax)
    ax.set_title("Number of Contacts Impact on Subscription")
    ax.set_xlabel("Contacts Number")
    ax.set_ylabel("Proportion")
    fig.tight_layout()
    fig.savefig(output_dir / "campaign_subscription_proportion.png", dpi=160)
    plt.close(fig)


def create_all_plots(input_file, output_dir):
    """Read the raw dataset and write all EDA figures to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype(str).str.strip().str.lower()

    save_deposit_distribution(df, output_path)
    save_numerical_histograms(df, output_path)
    save_correlation_heatmap(df, output_path)
    save_boxplots(df, output_path)
    save_categorical_proportions(df, output_path)
    save_age_by_subscription(df, output_path)
    save_campaign_subscription(df, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Create EDA plots for the bank marketing analysis.")
    parser.add_argument("--input", required=True, help="Path to raw bank.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_all_plots(args.input, args.output_dir)
