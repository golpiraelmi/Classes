import os

import pandas as pd
import numpy as np
from tableone import TableOne

from scipy.stats import (
        mannwhitneyu, wilcoxon, ttest_ind, ttest_rel,
        chi2_contingency, fisher_exact, shapiro,
        f_oneway, kruskal
    )
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

#############################



def table1(
    df,
    columns=None,
    categorical=None,
    groupby=None,
    alpha_normality=0.05,
    exact_threshold=5,
    **kwargs
):
    

    # -------------------- Setup --------------------
    columns = columns or df.columns.tolist()
    categorical = categorical or []

    df = df.copy()

    if groupby is None:
        raise ValueError("You must provide 'groupby'.")

    group_levels = df[groupby].dropna().unique()

    if len(group_levels) < 2:
        raise ValueError("Need at least 2 groups.")



    new_p_values = {}
    test_used = {}
 
    # -------------------- Helpers --------------------
    def is_binary(series):
        return series.dropna().nunique() == 2

    def is_normal(x):
        if len(x) < 3:
            return False
        try:
            return shapiro(x).pvalue > alpha_normality
        except:
            return False

   
    # -------------------- Main loop --------------------
    for var in columns:

        # ==================== CATEGORICAL ====================
        if var in categorical:

            table = pd.crosstab(df[groupby], df[var])

            if table.size == 0:
                new_p_values[var] = None
                continue

            try:
                chi2, p, dof, expected = chi2_contingency(table)

                if (expected < exact_threshold).any() and table.shape == (2, 2):
                    _, p = fisher_exact(table)
                    test_used[var] = "Fisher's exact"
                else:
                    test_used[var] = "Chi-square"

                new_p_values[var] = round(p, 3)

            except:
                new_p_values[var] = None

        # ==================== CONTINUOUS ====================
        else:
            df[var] = pd.to_numeric(df[var], errors='coerce')

            groups = [df[df[groupby] == g][var].dropna() for g in group_levels]

            if len(groups) == 2:
                x, y = groups
                normal = is_normal(x) and is_normal(y)

                try:
                    if normal:
                        stat, p = ttest_ind(x, y, equal_var=False)
                        test_used[var] = "Welch's t-test"
                    else:
                        stat, p = mannwhitneyu(x, y)
                        test_used[var] = "Mann-Whitney"

                    new_p_values[var] = round(p, 3)

                except:
                    new_p_values[var] = None

            else:
                normal = all(is_normal(g) for g in groups)

                try:
                    if normal:
                        stat, p = f_oneway(*groups)
                        test_used[var] = "ANOVA"
                    else:
                        stat, p = kruskal(*groups)
                        test_used[var] = "Kruskal-Wallis"

                    new_p_values[var] = round(p, 3)

                except:
                    new_p_values[var] = None

    # -------------------- Build TableOne --------------------
    table = TableOne(
        df,
        columns=columns,
        categorical=categorical,
        groupby=groupby,
        pval=True,
        htest_name=True,
        **kwargs
    )

    table1_df = table.tableone
    table1_df.columns = table1_df.columns.get_level_values(1)
    table1_df = table1_df.reset_index()

    # -------------------- Replace p-values --------------------
    table1_df["P-Value"] = table1_df["level_0"].apply(
        lambda var: new_p_values.get(
            var.split(",")[0],
            table1_df.loc[table1_df["level_0"] == var, "P-Value"].iloc[0]
        )
    )

    # -------------------- Replace test names --------------------
    if "Test" in table1_df.columns:
        table1_df["Test"] = table1_df["level_0"].apply(
            lambda var: test_used.get(
                var.split(",")[0],
                table1_df.loc[table1_df["level_0"] == var, "Test"].iloc[0]
            )
        )


    # -------------------- Clean display --------------------
    table1_df.set_index(["level_0", "level_1"], inplace=True)

    first_idx = table1_df.groupby('level_0').head(1).index
    table1_df['P-Value'] = table1_df.apply(
        lambda row: row['P-Value'] if row.name in first_idx else '',
        axis=1
    )


    if "Test" in table1_df.columns:
        table1_df['Test'] = table1_df.apply(
        lambda row: row['Test'] if row.name in first_idx else '',
        axis=1
    )



    return table1_df



################################################
def paired_binary_summary(
    df,
    pairs,
    test1_label="Test 1",
    test2_label="Test 2",
    row_label="Variable",
    include_all_test1=True,
    decimals=1,
    pval_decimals=5,
    return_raw=False,
    column_order=None
):
    """
    General summary table for paired binary comparisons with McNemar test.

    Parameters
    ----------
    df : pandas.DataFrame
    pairs : list of tuples
        Each tuple: (row_name, test1_column, test2_column)
    test1_label : str
    test2_label : str
    row_label : str
        Name of the first column (e.g., "Variable", "Outcome")
    include_all_test1 : bool
    decimals : int
    pval_decimals : int
    return_raw : bool
    column_order : list or None
        Optional custom column order

    Returns
    -------
    pandas.DataFrame
    """

    rows = []

    for name, col1, col2 in pairs:
        sub = df.dropna(subset=[col1, col2]).copy()
        total = len(sub)

        if total == 0:
            continue

        test1_pos = sub[col1].sum()
        test2_pos = sub[col2].sum()

        concordant = (sub[col1] == sub[col2]).sum()
        discordant = (sub[col1] != sub[col2]).sum()

        b = ((sub[col1] == 1) & (sub[col2] == 0)).sum()
        c = ((sub[col1] == 0) & (sub[col2] == 1)).sum()

        table = [[0, b], [c, 0]]

        try:
            pval = mcnemar(table, exact=False).pvalue
        except Exception:
            pval = float("nan")

        def fmt(n, d):
            return f"{n} ({(n/d*100):.{decimals}f})" if d > 0 else "NA"

        row = {
            row_label: name,
            f"{test1_label} n (%)": fmt(test1_pos, total),
            f"{test2_label} n (%)": fmt(test2_pos, total),
            "Discordant n (%)": fmt(discordant, total),
            "p-value": f"{pval:.{pval_decimals}f}" if pd.notnull(pval) else "NA"
        }

        if include_all_test1:
            test1_all = df[col1].sum()
            total_all = df[col1].notna().sum()
            row[f"All {test1_label} n (%)"] = fmt(test1_all, total_all)

        if return_raw:
            row.update({
                "total": total,
                "b (1,0)": b,
                "c (0,1)": c,
                "concordant": concordant
            })

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Default adaptive column order
    if column_order is None:
        column_order = [
            row_label,
            f"All {test1_label} n (%)",
            f"{test1_label} n (%)",
            f"{test2_label} n (%)",
            "Discordant n (%)",
            "p-value"
        ]

    summary_df = summary_df[[c for c in column_order if c in summary_df.columns]]

    return summary_df



######################
import pandas as pd
import numpy as np
from scipy import stats

def paired_continuous_table(
    df,
    pairs,
    var_label="Variable",
    test1_label="Time 1",
    test2_label="Time 2",
    nonnormal=None,
    decimals=2,
    pval_decimals=4,
    test_type="auto"  # "auto", "parametric", "nonparametric"
):
    """
    TableOne-style summary for paired continuous variables with test name.

    Parameters
    ----------
    df : DataFrame
    pairs : list of tuples
        Each tuple: (name, col1, col2)
    var_label : str
    test1_label : str
    test2_label : str
    nonnormal : list or None
        variables to force nonparametric
    decimals : int
    pval_decimals : int
    test_type : str
        "auto", "parametric", or "nonparametric"

    Returns
    -------
    DataFrame
    """

    rows = []
    nonnormal = nonnormal or []

    for name, col1, col2 in pairs:
        sub = df[[col1, col2]].dropna()
        n = len(sub)

        if n == 0:
            continue

        x = sub[col1]
        y = sub[col2]
        diff = x - y

        # --- Determine test ---
        use_nonparam = False

        if test_type == "nonparametric":
            use_nonparam = True
        elif test_type == "parametric":
            use_nonparam = False
        else:  # auto
            if name in nonnormal:
                use_nonparam = True
            else:
                # calculate skew of differences
                skewness = diff.skew()
                # strongly skewed or very small sample -> use Wilcoxon
                if abs(skewness) > 0.5 or n < 15:
                    use_nonparam = True
                else:
                    try:
                        p_norm = stats.shapiro(diff).pvalue
                        use_nonparam = p_norm < 0.05
                    except Exception:
                        use_nonparam = True

        # --- Summary stats ---
        def mean_sd(a):
            return f"{a.mean():.{decimals}f} ({a.std():.{decimals}f})"

        def median_iqr(a):
            return f"{a.median():.{decimals}f} ({a.quantile(0.25):.{decimals}f}-{a.quantile(0.75):.{decimals}f})"

        if use_nonparam:
            stat1 = median_iqr(x)
            stat2 = median_iqr(y)
            try:
                pval = stats.wilcoxon(x, y).pvalue
            except Exception:
                pval = np.nan
            test_used = "Wilcoxon"
        else:
            stat1 = mean_sd(x)
            stat2 = mean_sd(y)
            try:
                pval = stats.ttest_rel(x, y).pvalue
            except Exception:
                pval = np.nan
            test_used = "Paired t-test"

        row = {
            var_label: name,
            f"{test1_label}": stat1,
            f"{test2_label}": stat2,
            "p-value": f"{pval:.{pval_decimals}f}" if pd.notnull(pval) else "NA",
            "Test": test_used,
            "n": n
        }

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # --- Column order (TableOne style) ---
    desired_order = [
        var_label,
        f"{test1_label}",
        f"{test2_label}",
        "p-value",
        "Test",
        "n"
    ]

    summary_df = summary_df[[c for c in desired_order if c in summary_df.columns]]

    return summary_df

##############################################################

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd
import numpy as np



def survival_analysis(df, time_to_event_col, event_col, patient_col=None, group_col=None,
                      high_risk_label=None, label_map=None, colors=None,
                      xlim=None, save_dir=None, figsize=(8, 5),dpi=100,):
    """
    Generic survival analysis function with consistent plot sizing.
    """
    df_clean = df.copy()
    df_clean[time_to_event_col] = pd.to_numeric(df_clean[time_to_event_col], errors='coerce')
    df_clean[event_col] = pd.to_numeric(df_clean[event_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[time_to_event_col, event_col])
    
    if len(df_clean) == 0:
        print("No data available after dropping missing time/event rows.")
        return

    # -------------------- 1. Swimmer Plot --------------------
    if patient_col and patient_col in df_clean.columns:
        df_sorted = df_clean.sort_values(by=time_to_event_col).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for i, (_, row) in enumerate(df_sorted.iterrows()):
            ax.hlines(
                y=i,
                xmin=0,
                xmax=row[time_to_event_col],
                linewidth=1 if row[event_col] == 1 else 0.5,
                color='red' if row[event_col] == 1 else 'grey',
                alpha=1 if row[event_col] == 1 else 0.7
            )
            if row[event_col] == 1:
                ax.plot(row[time_to_event_col], i, 'x', color='red', markersize=8)

        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted[patient_col], fontsize=8)
        ax.set_ylim(-1, len(df_sorted))  # standardize spacing

        legend_elements = [
            Line2D([0], [0], color='red', marker='x', lw=1, label='Event'),
            Line2D([0], [0], color='grey', lw=1, alpha=0.7, label='No Event')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        ax.set_xlabel("Time (Years)", fontsize=12)
        ax.set_ylabel("Patients", fontsize=12)
        ax.set_title("Swimmer Plot", fontsize=14)
        ax.grid(True, linewidth=0.2, color="#dcdde2")
        plt.margins(0)

        plt.tight_layout()  
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/swimmer_plot.png", dpi=300)
        plt.show()

    # -------------------- 2. Overall KM --------------------
    kmf = KaplanMeierFitter()
    kmf.fit(df_clean[time_to_event_col], event_observed=df_clean[event_col])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.suptitle('Kaplan-Meier Plot', fontsize=16)
    kmf.plot(ax=ax, color="#112171", label='Event Free Survival')
    ax.grid(True, linewidth=0.5, color='#c6c7cc', linestyle='-')
    ax.set_xlabel('Time (Years)', fontsize=12)
    ax.set_ylabel('Event Free Survival', fontsize=12)
    plt.legend(loc='lower left')
    plt.margins(0)

    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/km_plot.png", dpi=300)
    plt.show()

    # -------------------- 3. KM by Group --------------------
    if group_col and group_col in df_clean.columns:
        groups = df_clean[group_col].dropna().unique()
        if not label_map:
            label_map = {g: str(g) for g in groups}
        if not colors:
            colors = plt.cm.tab10.colors[:len(groups)]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.suptitle('Kaplan-Meier by Group', fontsize=14)

        for i, group in enumerate(groups):
            kmf_group = KaplanMeierFitter()
            mask = df_clean[group_col] == group
            kmf_group.fit(
                df_clean.loc[mask, time_to_event_col],
                event_observed=df_clean.loc[mask, event_col],
                label=label_map.get(group)
            )
            kmf_group.plot(ax=ax, color=colors[i])

        ax.set_xlabel('Time (Years)', fontsize=12)
        ax.set_ylabel('Event Free Survival', fontsize=12)
        ax.grid(True, linewidth=0.5, color='#c6c7cc')
        plt.legend(title='Group', loc='lower left')
        plt.margins(0)

        if xlim:
            plt.xlim(xlim)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/km_plot_by_group.png", dpi=300)
        plt.show()

# -------------------- 4. Log-rank Test --------------------
    if group_col and group_col in df_clean.columns and high_risk_label:
        ix = df_clean[group_col] == high_risk_label
        T_exp = df_clean.loc[ix, time_to_event_col]
        T_con = df_clean.loc[~ix, time_to_event_col]
        E_exp = df_clean.loc[ix, event_col]
        E_con = df_clean.loc[~ix, event_col]

        results = logrank_test(T_exp, T_con,
                            event_observed_A=E_exp,
                            event_observed_B=E_con)
        print("\nLog-rank Test Results:")
        results.print_summary()
        
