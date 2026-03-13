# hgb_analysis.py
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from tableone import TableOne
import scikit_posthocs as sp
import os
from scipy import stats
from IPython.display import display

#############################
def my_tableone(df, cols, cats, non_norm, group):
    new_p_values = {}

    # Ensure non-normal variables are numeric
    for variable in non_norm:
        df[variable] = pd.to_numeric(df[variable], errors='coerce')

    # Automatically detect the two unique levels of the grouping variable
    group_levels = df[group].dropna().unique()
    if len(group_levels) != 2:
        raise ValueError(f"Grouping variable '{group}' must have exactly 2 levels. Found: {group_levels}")

    # Perform Mann-Whitney U test for each non-normal variable
    for variable in non_norm:
        data1 = df[df[group] == group_levels[0]][variable].dropna()
        data2 = df[df[group] == group_levels[1]][variable].dropna()
        if len(data1) > 0 and len(data2) > 0:
            stat, p_value = mannwhitneyu(data1, data2)
            new_p_values[variable] = round(p_value, 3)  # round to 3 decimals
        else:
            new_p_values[variable] = None  # in case one group has no data

    # Create TableOne
    table = TableOne(
        df,
        columns=cols,
        categorical=cats,
        groupby=group,
        nonnormal=non_norm,
        htest_name=True,
        label_suffix=True,
        pval=True,
        missing=True,
        normal_test=True,
        display_all=True,
        include_null=False
    )

    table1_df = table.tableone
    table1_df.columns = table1_df.columns.get_level_values(1)
    table1_df = table1_df.reset_index()

    # Replace P-Value with our computed Mann-Whitney p-values
    table1_df["P-Value"] = table1_df["level_0"].apply(
        lambda var: new_p_values.get(var.split(",")[0], 
                                    table1_df.loc[table1_df["level_0"] == var, "P-Value"].iloc[0])
    )

    table1_df.set_index(["level_0", "level_1"], inplace=True)

    # Only keep P-Value on the first row of each variable
    first_idx = table1_df.groupby('level_0').head(1).index
    table1_df['P-Value'] = table1_df.apply(
        lambda row: row['P-Value'] if (row.name in first_idx) else '',
        axis=1
    )

    # Replace test names
    table1_df['Test'] = table1_df['Test'].replace({'Kruskal-Wallis': 'Mann-Whitney'})

    return table1_df

#############################
def styled_tableone(
    df,
    columns,
    categorical=None,
    nonnormal=None,
    groupby=None,
    alpha=0.05
):
    """
    Create and display a styled TableOne object with significant p-value highlighting.
    """

    if categorical is None:
        categorical = []

    if nonnormal is None:
        nonnormal = []

    if groupby is None:
        raise ValueError("groupby must be specified")

    # Create TableOne
    table = TableOne(
        df,
        groupby=groupby,
        columns=columns,
        categorical=categorical,
        nonnormal=nonnormal,
        htest_name=True,
        pval=True,
        normal_test=True,
    )

    t1_df = table.tableone.copy()

    

    # Locate p-value column (MultiIndex-safe)
    pval_cols = [
        col for col in t1_df.columns
        if any(
            'p' in str(level).lower() and 'val' in str(level).lower()
            for level in col
        )
    ]

    if not pval_cols:
        raise ValueError("No p-value column found in TableOne output.")

    pval_col = pval_cols[0]

    t1_df = t1_df.astype(object).fillna('')

    # Convert p-values to numeric
    t1_df[pval_col] = pd.to_numeric(t1_df[pval_col], errors='coerce')
    

    # Identify significant variables
    significant_rows = t1_df[t1_df[pval_col].astype(float) < alpha]


    var_names = (
        significant_rows.index
        .to_series()
        .apply(extract_var_name)
        .dropna()
        .unique()
        .tolist()
    )

    sig_vars = set(var_names)

    # Styled TableOne display
    styled_tableone = (
    t1_df.style
    .format({pval_col: "{:.3f}"})
    .apply(lambda row: highlight_tableone_significant(row, sig_vars), axis=1)
    )

    display(styled_tableone)

    return None



#############################
def analyze_hgb(df_blood_draws, pod_time="POD1", title=None):
    """
    Compare hemoglobin from first draw to a specified POD.
    
    Parameters
    ----------
    df_blood_draws : pd.DataFrame
        DataFrame with columns: ['StudyID','Hemoglobin','Time',
                                 'injury_to_lab_hrs','Draw_date']
    pod_time : str
        Post-operative day label to compare against the first draw (default="POD1").
    title : str
        Plot title (optional).
        
    Returns
    -------
    results_df : pd.DataFrame
        Summary with Wilcoxon test, median difference, and bootstrap CI.
    """
    
    # First draw
    first_draw = (
        df_blood_draws.dropna(subset=['injury_to_lab_hrs'])
        .sort_values(['StudyID','injury_to_lab_hrs'])
        .drop_duplicates(subset=['StudyID'], keep='first')
        [['StudyID','Hemoglobin']]
        .rename(columns={'Hemoglobin':'Hgb_first_draw'})
    )

    # POD draw
    pod_draw = (
        df_blood_draws[df_blood_draws['Time']==pod_time]
        .sort_values(['StudyID','Draw_date'])
        .drop_duplicates(subset=['StudyID'], keep='first')
        [['StudyID','Hemoglobin']]
        .rename(columns={'Hemoglobin':f'Hgb_{pod_time}'})
    )

    # Merge & clean
    df_hgb = pd.merge(first_draw, pod_draw, on='StudyID', how='outer')
    df_hgb = df_hgb.sort_values('StudyID').dropna().reset_index(drop=True)
    df_hgb['Hgb_first_draw'] = pd.to_numeric(df_hgb['Hgb_first_draw'], errors='coerce')
    df_hgb[f'Hgb_{pod_time}'] = pd.to_numeric(df_hgb[f'Hgb_{pod_time}'], errors='coerce')
    df_hgb['delta_Hgb'] = df_hgb[f'Hgb_{pod_time}'] - df_hgb['Hgb_first_draw']

    # Wilcoxon test & bootstrap
    stat, p_val = wilcoxon(df_hgb['Hgb_first_draw'], df_hgb[f'Hgb_{pod_time}'])
    median_diff = df_hgb['delta_Hgb'].median()
    np.random.seed(200)
    boot_medians = [
        np.median(np.random.choice(df_hgb['delta_Hgb'], size=len(df_hgb), replace=True))
        for _ in range(5000)
    ]
    ci_lower, ci_upper = np.percentile(boot_medians, [2.5, 97.5])

    results_df = pd.DataFrame({
        'Wilcoxon W':[stat],
        'p-value':[p_val],
        f'Median Difference ({pod_time} - First)':[median_diff],
        '95% CI Lower':[ci_lower],
        '95% CI Upper':[ci_upper]
    })

    # Plot
    df_long = df_hgb.melt(value_vars=['Hgb_first_draw', f'Hgb_{pod_time}'],
                          var_name='Timepoint', value_name='Hemoglobin')
    sns.set_theme(style="whitegrid", font_scale=1.2)
    custom_palette = ['#5DADE2', '#E4E73C']
    plt.figure(figsize=(7,5))
    sns.boxplot(data=df_long, x='Timepoint', y='Hemoglobin',
                palette=custom_palette, width=0.5, linewidth=1.5, fliersize=0)
    plt.title(title or f'Hemoglobin: First vs {pod_time}', fontsize=14)
    plt.ylabel('Hemoglobin (g/L)')
    plt.xlabel('')
    plt.tight_layout()
    plt.show()

    return results_df

#############################
def plot_variables_over_time(
    df, 
    custom_order=None, 
    variables=None, 
    hue=None, 
    style=None, 
    palette=None,
    xlabel="Timepoint",
    var_labels=None,  
    legend_title=None,
    dashes=None,
    out_dir="Results",
    with_reference_lines='Yes',
):
    """
    Plot TEG/other variables over ordered timepoints.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Time' column and variables to plot.
    custom_order : list, optional
        Ordering of time categories.
    variables : list, optional
        Variables to plot. Defaults to common TEG vars.
    hue : str, optional
        Column for color grouping.
    style : str, optional
        Column for line style grouping.
    palette : dict or list, optional
        Colors to use.
    xlabel : str, default="Timepoint"
        Label for x-axis.
    var_labels : dict, optional
        Mapping {colname: label}.
    legend_title : str, optional
        Title for legend.
    """

    rcParams.update({
    "font.family": "Arial",
    "font.weight": "normal",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold"
    })


    if variables is None:
        variables = ['R_time', 'K_time',  'MA', 'LY30', 'ACT','Alpha_Angle']
        # df[variables] = pd.to_numeric(df[variables], errors='coerce')

    if custom_order is None:
        custom_order = ['Pre_Op', 'POD1', 'POD3', 'POD5', 'POD7', 'Week2', 'Week4', 'Week6', 'Month3']

    if var_labels is None:
        var_labels = {var: var.replace('_',' ').title() for var in variables}
        
    df = df.copy()
    df = df[~df['Time'].isna()]
    df['Time'] = pd.Categorical(df['Time'], categories=custom_order, ordered=True)

    if palette is None and hue is not None:
        palette = sns.color_palette("Set2", n_colors=df[hue].nunique())

    if dashes is None:
        dashes={"No": (4, 2), "Yes": (None, None)}



    if with_reference_lines=='No':
        output_dir = os.path.join(out_dir, "without_reference_lines")
        os.makedirs(output_dir, exist_ok=True)

        for var in variables:
            fig, ax = plt.subplots(figsize=(10, 4))

            time_labels = {
                'Pre_Op': 'PreOp',
                'Post_Op': 'PO1H',
                'POD1': 'POD1',
                'POD3': 'POD3',
                'POD5': 'PD5',
                'POD7': 'PD7',
                'Week2': 'Week2',
                'Week4': 'Week4',
                'Week6': 'Week6',
                'Month3': 'Month3'
            }



            line_kwargs = dict(
                x='Time',
                y=var,
                data=df,
                hue=hue,
                style=style,
                estimator='mean',
                errorbar='se',
                markers=True,
                linewidth=2,
                palette=palette
            )

            # Only add dashes if style is used
            if style is not None:
                if dashes is None:
                    dashes = {"No": (4, 2), "Yes": (None, None)}
                line_kwargs["dashes"] = dashes

            sns.lineplot(**line_kwargs)
            ax.set_xticklabels([time_labels.get(t.get_text(), t.get_text()) 
                        for t in ax.get_xticklabels()])
            


            plt.setp(ax.get_xticklabels(), rotation=0)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(var_labels.get(var, var), fontsize=12)  
            if legend_title:
                ax.legend(title=legend_title, loc=2)
            ax.set_title('') #var_labels.get(var, var), fontsize=14
            ax.grid(True, alpha=0.3)
            # ax = plt.gca()
            # print(ax.xaxis.label.get_fontname())

            plt.margins(0)
            out_path = os.path.join(output_dir, f"{var}_over_time.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.show()

    elif with_reference_lines=='Yes':
        os.makedirs(out_dir, exist_ok=True)

    

        for var in variables:
            fig, ax = plt.subplots(figsize=(10, 4))

            time_labels = {
                'Pre_Op': 'PreOp',
                'Post_Op': 'PO1H',
                'POD1': 'POD1',
                'POD3': 'POD3',
                'POD5': 'PD5',
                'POD7': 'PD7',
                'Week2': 'Week2',
                'Week4': 'Week4',
                'Week6': 'Week6',
                'Month3': 'Month3'
            }



            line_kwargs = dict(
                x='Time',
                y=var,
                data=df,
                hue=hue,
                style=style,
                estimator='mean',
                errorbar='se',
                markers=True,
                linewidth=2,
                palette=palette
            )

            # Only add dashes if style is used
            if style is not None:
                if dashes is None:
                    dashes = {"No": (4, 2), "Yes": (None, None)}
                line_kwargs["dashes"] = dashes

            sns.lineplot(**line_kwargs)
            ax.set_xticklabels([time_labels.get(t.get_text(), t.get_text()) 
                        for t in ax.get_xticklabels()])
            
            # Add horizontal reference lines for specific variables
            if var == "MA":
                ax.axhline(y=65, linestyle="--", linewidth=1.5, color='black')
            elif var == "AA-inh":
                ax.axhline(y=20, linestyle="--", linewidth=1.5, color='black')
            elif var == "AA-ma":
                ax.axhline(y=50, linestyle="--", linewidth=1.5, color='black')

            plt.setp(ax.get_xticklabels(), rotation=0)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(var_labels.get(var, var), fontsize=12)  
            if legend_title:
                ax.legend(title=legend_title, loc=2)
            ax.set_title('') #var_labels.get(var, var), fontsize=14
            ax.grid(True, alpha=0.3)
            # ax = plt.gca()
            # print(ax.xaxis.label.get_fontname())

            plt.margins(0)
            out_path = os.path.join(out_dir, f"{var}_over_time.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.show()





        

#############################
def display_summary_tables(df, filter_col, filter_value='Yes', extra_col=None, drop_study=True):
    from IPython.display import display_html
    """
    Display side-by-side tables per Study (Hip, Femur, Pelvis, Pathway)
    
    Parameters:
    - df: pandas DataFrame
    - filter_col: str, column to filter by (e.g., 'VTE', 'Withdraw')
    - filter_value: value to keep (default 'Yes')
    - extra_col: str or None, additional column to include (e.g., 'VTE_type')
    - drop_study: bool, whether to remove 'Study' column from display
    """
    
    # 1) Filter rows
    cols = ['Study', 'StudyID']
    if extra_col:
        cols.append(extra_col)
    
    df_filtered = df[df[filter_col] == filter_value][cols].drop_duplicates().reset_index(drop=True)
    
    # 2) Create tables per Study
    studies = ['Hip', 'Femur', 'Pelvis', 'Pathway','Arthoplasty']
    tables = {s: df_filtered[df_filtered['Study'] == s].copy() for s in studies}
    
    # 3) Add numbering and drop Study column if requested
    for s in tables:
        tables[s] = tables[s].reset_index(drop=True)
        tables[s].insert(0, 'No', tables[s].index + 1)
        if drop_study and 'Study' in tables[s].columns:
            tables[s] = tables[s].drop(columns=['Study'])
    
    # 4) Build HTML for side-by-side display
    html = """
    <style>
    .table-container {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 30px;
    }
    .table-block {
        text-align: center;
    }
    .table-block table {
        margin-top: 5px;
        border-collapse: collapse;
    }
    .table-block th, .table-block td {
        border: 1px solid #999;
        padding: 4px 8px;
    }
    </style>
    
    <div class="table-container">
    """
    
    for t in tables.values():
        html += f"<div class='table-block'>{t.to_html(index=False)}</div>"
    
    html += "</div>"
    
    display_html(html, raw=True)



#############################
def display_value_counts_per_study(df, col):
    from IPython.display import display_html
    """
    Display value_counts of a column per Study (not per StudyID), side-by-side
    with Study names as headers above each table.

    Parameters:
    - df: pandas DataFrame
    - col: str, column name to count values for
    """
    
    studies = ['Hip', 'Femur', 'Pelvis', 'Pathway']
    tables = {}

    for s in studies:
        # Filter by study
        df_s = df[df['Study'] == s].copy()
        if df_s.empty:
            tables[s] = pd.DataFrame(columns=[col, 'count'])
            continue
        
        # Compute value_counts for the whole Study
        vc = df_s[col].value_counts(dropna=False).reset_index()
        vc.columns = [col, 'count']
        vc.insert(0, 'No', range(1, len(vc)+1))
        
        tables[s] = vc

    # Display side-by-side in HTML
    html = """
    <style>
    .table-container {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 30px;
    }
    .table-block {
        text-align: center;
    }
    .table-block table {
        margin-top: 5px;
        border-collapse: collapse;
    }
    .table-block th, .table-block td {
        border: 1px solid #999;
        padding: 4px 8px;
    }
    </style>
    
    <div class="table-container">
    """
    
    for study_name, t in tables.items():
        html += f"<div class='table-block'><h3>{study_name}</h3>{t.to_html(index=False)}</div>"
    
    html += "</div>"
    
    display_html(html, raw=True)


########################################## Dec 23rd
def hemoglobin_prior_to_first_rbc(
    df,
    study_id_col="StudyID",
    lab_time_col="time_injury_lab_hours",
    rbc_time_col="time_injury_rbc_hours",
    hb_col="Hemoglobin",
    rbc_flag_col="blood_rbc_yn",
    rbc_date="blood_date",
    blood_draw_date="Draw_date_lab",
    blood_rbc='blood_rbc'
):
    """
    Returns the hemoglobin value measured closest in time BEFORE
    the first RBC transfusion for each StudyID.

    - Keeps all StudyIDs where blood_rbc_yn=='Yes'
    - If no Hb exists prior to first RBC, Hemoglobin and lab_time_col are NaN
    """

    required_cols = {study_id_col, lab_time_col, rbc_time_col, hb_col, rbc_flag_col, rbc_date, blood_draw_date, blood_rbc}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Only StudyIDs with RBC transfusion
    rbc_patients = df[df[rbc_flag_col] == 'Yes'][[study_id_col, rbc_time_col]].dropna()

    # First RBC transfusion time per StudyID
    first_rbc = (
        rbc_patients
        .groupby(study_id_col, as_index=False)
        .min()
        .rename(columns={rbc_time_col: "first_rbc_time"})
    )

    # Include all relevant columns for labs
    labs = df[[study_id_col, lab_time_col, hb_col, rbc_date, blood_draw_date, blood_rbc]].dropna(subset=[hb_col])

    # Merge labs with first RBC times to keep all RBC patients
    labs_pre_rbc = labs.merge(first_rbc, on=study_id_col, how="right")

    # Keep only labs before first RBC
    labs_pre_rbc = labs_pre_rbc[labs_pre_rbc[lab_time_col] < labs_pre_rbc["first_rbc_time"]]

    # Pick the latest lab before RBC
    hb_prior = (
        labs_pre_rbc
        .sort_values([study_id_col, lab_time_col])
        .groupby(study_id_col, as_index=False)
        .last()
    )

    # Merge back with all RBC patients to keep StudyIDs with no prior Hb as NaN
    result = first_rbc.merge(
        hb_prior[[study_id_col, lab_time_col, hb_col, rbc_date, blood_draw_date, blood_rbc]],
        on=study_id_col,
        how='left'
    )

    # Rename Hb column
    result = result.rename(columns={hb_col: 'Hemoglobin_prior_to_first_transfusion'})

    return result




#############################
def dunn_test(df, value_col, group_col, p_adjust='bonferroni'):
    """
    Perform Dunn's post-hoc test.

    Parameters
    ----------
    df : pandas.DataFrame
    value_col : str
        Numeric variable (e.g. lab value)
    group_col : str
        Grouping variable (e.g. treatment, timepoint)
    p_adjust : str
        Multiple testing correction (bonferroni, holm, fdr_bh, etc.)

    Returns
    -------
    pandas.DataFrame
        Pairwise adjusted p-values
    """
    data = df[[value_col, group_col]].dropna()

    result = sp.posthoc_dunn(
        data,
        val_col=value_col,
        group_col=group_col,
        p_adjust=p_adjust
    )

    return result



#############################
def detect_non_normal(df, cols, alpha=0.05, min_n=5):
    non_normal = []

    for c in cols:
        x = df[c].dropna()

        # skip if too few observations
        if len(x) < min_n:
            non_normal.append(c)
            continue

        try:
            _, p = shapiro(x)
            if p < alpha:
                non_normal.append(c)
        except Exception:
            # if test fails, be conservative
            non_normal.append(c)

    return non_normal






#############################
def extract_var_name(idx):
    """Extract base variable name from TableOne index (handles MultiIndex)."""
    if isinstance(idx, tuple):
        idx = idx[0]
    return idx.split(',')[0].strip()

RED = '#ffcccc'
#############################
def highlight_tableone_significant(row, sig_vars):
    """Highlight entire TableOne rows for significant variables."""
    var = extract_var_name(row.name)
    if var in sig_vars:
        return [f'background-color: {RED}'] * len(row)
    return [''] * len(row)
#############################
def highlight_pvals(val):
    """Highlight significant p-values in post-hoc tables."""
    if pd.notna(val) and val < 0.05:
        return f'background-color: {RED}'
    return ''

#############################
def compare_to_reference(
    df,
    time_col='Time',
    value_col='MA',
    reference_value=65,
    alpha=0.05,
    time_order=None,
    return_styled=True
):
    """
    For each timepoint:
        - Test normality (Shapiro-Wilk)
        - If normal -> One-sided t-test (mean > reference)
        - If not normal -> One-sided Wilcoxon
        - Flag only if statistically greater than reference AND mean >= reference
    """

    results = []

    for timepoint, group in df.groupby(time_col):
        values = group[value_col].dropna()
        n = len(values)

        if n < 3:
            continue

        # ---- Normality ----
        shapiro_stat, shapiro_p = stats.shapiro(values)
        is_normal = shapiro_p > alpha

        # ---- One-sided test (> reference) ----
        if is_normal:
            test_name = "One-sample t-test (greater)"
            stat, p_two_sided = stats.ttest_1samp(values, reference_value)
            if stat > 0:
                p_val = p_two_sided / 2
            else:
                p_val = 1.0
        else:
            test_name = "Wilcoxon (greater)"
            stat, p_two_sided = stats.wilcoxon(values - reference_value)
            median_diff = np.median(values - reference_value)
            if median_diff > 0:
                p_val = p_two_sided / 2
            else:
                p_val = 1.0

        mean_val = values.mean()

        hyper_sig = (p_val < alpha) and (mean_val >= reference_value)

        results.append({
            "Timepoint": timepoint,
            "N": n,
            "Mean": mean_val,
            "SD": values.std(),
            # "Normality_p": shapiro_p,
            "Test Used": test_name,
            # "Test Statistic": stat,
            "p-value": p_val,
            "Significant?": "Yes" if hyper_sig else "No"
        })

    results_df = pd.DataFrame(results)

    # ---- Optional ordering ----
    if time_order is not None:
        results_df["Timepoint"] = pd.Categorical(
            results_df["Timepoint"],
            categories=time_order,
            ordered=True
        )
        results_df = results_df.sort_values("Timepoint")

    if not return_styled:
        return results_df

    # =========================
    # Styling
    # =========================

    def highlight_hyper(row):
        if row["Significant?"] == "Yes":
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)

    styled = (
        results_df
        .style
        .apply(highlight_hyper, axis=1)
        .format({
            "Mean": "{:.2f}",
            "SD": "{:.2f}",
            # "Normality_p": "{:.4f}",
            # "Test Statistic": "{:.3f}",
            "p-value": "{:.4f}"
        })
    )

    return styled




#############################
def proportion_summary(
    df,
    time_col="Time",
    state_col="hypercoagulable_state",
    threshold=0.50,
    alpha=0.05
):
    """
    Compute hypercoagulable proportion %, perform one-sample proportion test,
    and return styled dataframe.

    Parameters
    ----------
    counts : pd.DataFrame
        Contingency table (Time × hypercoagulable_state) with columns ['Yes','No'].

    threshold : float
        Null hypothesis proportion.

    alpha : float
        Significance level.

    Returns
    -------
    Styled pandas dataframe
    """
    counts = (df.groupby(time_col)[state_col].value_counts().unstack(fill_value=0))
    percent_df = counts.div(counts.sum(axis=1), axis=0) * 100

    percent_table = counts.astype(str) + " (" + percent_df.round(1).astype(str) + "%)"

    # -------------------------
    # Statistical testing
    # -------------------------
    p_values = []

    for time in counts.index:
        count_yes = counts.loc[time, "Yes"]
        nobs = counts.loc[time].sum()

        _, p = proportions_ztest(
            count_yes,
            nobs,
            value=threshold,
            alternative='larger'
        )

        p_values.append(p)

    percent_table["p-value (> {:.0%})".format(threshold)] = np.round(p_values, 4)

    percent_table["Significant"] = np.where(
        percent_table.iloc[:, -1] < alpha,
        "Yes",
        "No"
    )

    # -------------------------
    # Styling
    # -------------------------
    def highlight_significant(row):
        if row["Significant"] == "Yes":
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)

    styled = (
        percent_table
        .style
        .apply(highlight_significant, axis=1)
        .format({
            percent_table.columns[-2]: "{:.4f}"
        })
    )

    return styled




#### THIS IS FOR DANIYYA'S PROJECT - DO NOT USE FOR OTHER PROJECTS WITHOUT REVIEWING THE CODE AND ENSURING IT FITS THE NEW DATASET AND ANALYTIC NEEDS! ####
def table1_for_Daniyya(df, time_order, groupby, demographic_yn='No'):


    RED = "\033[1;31m"
    BLUE = "\033[1;34m"
    RESET = "\033[0m"

    # ===============================
    # DEMOGRAPHIC ANALYSIS PIPELINE
    # ===============================
    if demographic_yn == 'Yes':

        analyses = [
            {
                "title": "All Patients",
                "columns": ['Age','Sex','BMI',
                            'Diabetes History','Cancer History',
                            'Cardiovascular Disease History','Pulmonary Disease History',
                            'Stroke History','Comorbidity (yes/no)','WOMAC Score','UCLA Score', 
                            'Ethnicity', 'Smoking History', 'Alcohol (yes/no)', 'Pulmonary Complication',
                            'Cardiovascular Complication', 'Infection Complication','Surgical Complication','ASA Classification','Anesthesia Type','TXA Type','Arthroplasty Type',
                            'Intraoperative Fluids','Intraoperative Blood Loss','Intraoperative Fluids Amount','Postoperative Fluids Amount','LOS','Surgical Approach'],
                 "categorical": ['Sex','Diabetes History','Cancer History',
                                'Cardiovascular Disease History','Pulmonary Disease History',
                                'Stroke History','Comorbidity (yes/no)','Ethnicity', 'Smoking History', 'Alcohol (yes/no)','UCLA Score', 'Pulmonary Complication',
                                'Cardiovascular Complication', 'Infection Complication','Surgical Complication','ASA Classification','Anesthesia Type', 'TXA Type','Arthroplasty Type',
                                'Intraoperative Fluids','Surgical Approach'],
                "nonnormal": ['Intraoperative Blood Loss', 'LOS', 'Postoperative Fluids Amount'],
                "filter": None
            }
        ]

        for analysis in analyses:

            df_base = df.copy()

            if analysis["filter"]:
                df_base = analysis["filter"](df_base)

            datasets = {
                "VTE Included": df_base,
                "VTE Excluded": df_base.query("VTE == 'No'")
            }

            for label, df_ in datasets.items():

                print(f"{RED}{analysis['title']} | {label} - grouped by {groupby}{RESET}")

                if df_.empty:
                    print("No data\n")
                    continue

                group_counts = df_[groupby].value_counts()

                if df_[groupby].nunique() < 2 or (group_counts == 1).any():
                    print("Insufficient groups for comparison\n")
                    continue
                # if df_[groupby].nunique() < 2:
                #     print("Insufficient groups for comparison\n")
                #     continue

                table = TableOne(
                    df_,
                    columns=analysis["columns"],
                    categorical=analysis["categorical"],
                    nonnormal=analysis["nonnormal"],
                    groupby=groupby,
                    pval=True,
                    normal_test=True,
                    htest_name=True,
                    display_all=True,
                    include_null=False
                ).tableone

                display(table.replace("Kruskal-Wallis", "Mann-Whitney U"))


        return


    # ===============================
    # TEG ANALYSIS PIPELINE
    # ===============================

    if time_order is None:
        time_iter = sorted(df["Time"].dropna().unique())
    else:
        time_iter = time_order

    # Define analyses by grouping variable
    if groupby == "hypercoagulable_state":

        analyses = [
            {
                "title": "All Patients",
                "columns": ["LY30"],
                "categorical": [],
                "nonnormal": [],
                "filter": None
            }
        ]

    elif groupby == "Pre_op_med":

        analyses = [
            {
                "title": "All Patients",
                "columns": ['MA','AA-inh','AA-ma','LY30',
                            'hypercoagulable_state',
                       
                            'platelet_hyperactive_state_ref50',
                            'platelet_hyperactive_state_ref55',
                            'platelet_inhibition_state_ref20',
                            'platelet_inhibition_state_ref50',
                            'platelet_inhibition_state_ref70'],
                "categorical": ['hypercoagulable_state',
                                'platelet_hyperactive_state_ref50',
                                'platelet_hyperactive_state_ref55',
                                'platelet_inhibition_state_ref20',
                                'platelet_inhibition_state_ref50',
                                'platelet_inhibition_state_ref70'],
                "nonnormal": [],
                "filter": None
            }
        ]

    else:

        analyses = [
            {
                "title": "All Patients",
                "columns": ["MA", "hypercoagulable_state"],
                "categorical": ["hypercoagulable_state"],
                "nonnormal": [],
                "filter": None
            },
            {
                "title": "ASA Excluded",
                "columns": ['AA-inh','AA-ma',
                            'platelet_hyperactive_state_ref50',
                                'platelet_hyperactive_state_ref55',
                                'platelet_inhibition_state_ref20',
                                'platelet_inhibition_state_ref50',
                                'platelet_inhibition_state_ref70'],
                "categorical": ['platelet_hyperactive_state_ref50',
                                'platelet_hyperactive_state_ref55',
                                'platelet_inhibition_state_ref20',
                                'platelet_inhibition_state_ref50',
                                'platelet_inhibition_state_ref70'],
                "nonnormal": [],
                "filter": lambda d: d.query("Pre_op_med != 'ASA'")
            }
        ]

    # ===============================
    # MAIN TEG ANALYSIS LOOP
    # ===============================

    for analysis in analyses:

        df_base = df.copy()

        if analysis["filter"]:
            df_base = analysis["filter"](df_base)

        datasets = {
            "VTE Included": df_base,
            "VTE Excluded": df_base.query("VTE == 'No'")
        }

        for label, df_ in datasets.items():

            if label == "VTE Excluded" and groupby == 'VTE':
                continue

            print(f"{RED}{analysis['title']} | {label} - grouped by {groupby}{RESET}")

            for Time in time_iter:
                df_time = df_[df_["Time"] == Time]

                print(f"{BLUE}{Time}{RESET}")
                print(f"{BLUE}{'*'*67}{RESET}")

                

                if df_time.empty:
                    print("No data\n")
                    continue
                
                group_counts = df_time[groupby].value_counts()

                if df_time[groupby].nunique() < 2 or (group_counts == 1).any():
                    print("Insufficient groups for comparison\n")
                    continue
                                # if df_time[groupby].nunique() < 2:
                #     print("Insufficient groups for comparison\n")
                #     continue

                display(TableOne(
                    df_time,
                    columns=analysis["columns"],
                    categorical=analysis["categorical"],
                    nonnormal=analysis["nonnormal"],
                    groupby=groupby,
                    pval=True,
                    normal_test=True,
                    htest_name=True,
                    display_all=True,
                    include_null=False
                ).tableone)

