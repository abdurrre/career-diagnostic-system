import pandas as pd


def build_normalized_market_skill_comparison(jobs_df, top_n=15):
    rows = []

    market_totals = jobs_df.groupby("market_source").size().to_dict()

    for _, row in jobs_df.dropna(subset=["cleaned_skills"]).iterrows():
        market = row["market_source"]

        skills = set(
            s.strip().lower()
            for s in row["cleaned_skills"].split(",")
            if s.strip()
        )

        for skill in skills:
            rows.append({
                "market": market,
                "skill": skill
            })

    skill_df = pd.DataFrame(rows)

    if skill_df.empty:
        return pd.DataFrame()

    count_df = (
        skill_df.groupby(["skill", "market"])
        .size()
        .unstack(fill_value=0)
    )

    if "Global" not in count_df.columns:
        count_df["Global"] = 0

    if "Indonesia" not in count_df.columns:
        count_df["Indonesia"] = 0

    count_df["Global_pct"] = (
        count_df["Global"] / market_totals.get("Global", 1) * 100
    )

    count_df["Indonesia_pct"] = (
        count_df["Indonesia"] / market_totals.get("Indonesia", 1) * 100
    )

    count_df["avg_pct"] = (
        count_df["Global_pct"] + count_df["Indonesia_pct"]
    ) / 2

    count_df["gap"] = (
        count_df["Indonesia_pct"] - count_df["Global_pct"]
    )

    result = (
        count_df
        .sort_values("avg_pct", ascending=False)
        .head(top_n)
        .reset_index()
    )

    return result.head(top_n)

def build_market_skill_comparison(jobs_df, top_n=15):

    rows = []

    for _, row in jobs_df.dropna(subset=["cleaned_skills"]).iterrows():

        market = row["market_source"]

        skills = set(
            s.strip().lower()
            for s in row["cleaned_skills"].split(",")
            if s.strip()
        )

        for skill in skills:
            rows.append({
                "market": market,
                "skill": skill
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame()

    pivot = (
        df.groupby(["skill", "market"])
        .size()
        .unstack(fill_value=0)
    )

    if "Global" not in pivot.columns:
        pivot["Global"] = 0

    if "Indonesia" not in pivot.columns:
        pivot["Indonesia"] = 0

    pivot["total"] = pivot["Global"] + pivot["Indonesia"]

    pivot = pivot.sort_values("total", ascending=False).head(top_n)

    return pivot.reset_index()

def get_unique_skill_count(skill_text):
    if pd.isna(skill_text):
        return 0

    return len(set(
        s.strip().lower()
        for s in str(skill_text).split(",")
        if s.strip()
    ))


def build_skill_role_matrix(jobs_data, top_n_skills=20):
    rows = []

    for _, row in jobs_data.dropna(subset=["cleaned_skills"]).iterrows():
        role = row["job_category"]

        skills = set(
            s.strip().lower()
            for s in row["cleaned_skills"].split(",")
            if s.strip()
        )

        for skill in skills:
            rows.append({
                "job_category": role,
                "skill": skill
            })

    skill_role_df = pd.DataFrame(rows)

    if skill_role_df.empty:
        return pd.DataFrame()

    top_skills = (
        skill_role_df["skill"]
        .value_counts()
        .head(top_n_skills)
        .index
    )

    matrix = (
        skill_role_df[skill_role_df["skill"].isin(top_skills)]
        .groupby(["job_category", "skill"])
        .size()
        .reset_index(name="frequency")
        .pivot(index="job_category", columns="skill", values="frequency")
        .fillna(0)
    )

    return matrix