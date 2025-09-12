# hgb_analysis.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from tableone import TableOne

def my_tableone (df, cols, cats, non_norm, group):
    
    new_p_values={}
    for variable in non_norm:
        # Convert the column to numeric, coerce errors to NaN
        df[variable] = pd.to_numeric(df[variable], errors='coerce')


    # Perform Mann-Whitney U test for each variable and print the p-values
    if group=='Surgery':
        for variable in non_norm:
            statistic, p_value = mannwhitneyu(df[df['Surgery']=='Hip'][variable].dropna(), 
                                            df[df['Surgery']=='Hip_pathway'][variable].dropna())
            
    elif group=='Pre_op_doac':
        for variable in non_norm:
            statistic, p_value = mannwhitneyu(df[df['Pre_op_doac']=='Yes'][variable].dropna(), 
                                            df[df['Pre_op_doac']=='No'][variable].dropna())
    else:
        pass
      

    table = TableOne(df, columns=cols, categorical=cats, groupby=group, nonnormal=non_norm, htest_name=True,
                    label_suffix=True, pval=True, missing=True, normal_test=True, display_all=True, include_null=False)
    
    table1_df = table.tableone

    table1_df.columns = table1_df.columns.get_level_values(1)
    table1_df = table1_df.reset_index()


    table1_df["P-Value"] = table1_df["level_0"].apply(
        lambda var: new_p_values.get(var.split(",")[0], table1_df["P-Value"].loc[table1_df["level_0"] == var].iloc[0]))

    table1_df.set_index(["level_0", "level_1"], inplace=True)

    # Get the first index for each group in 'level_0'
    first_idx = table1_df.groupby('level_0').head(1).index

    # Set P-Value to NaN for all rows except the first row for each group
    table1_df['P-Value'] = table1_df.apply(lambda row: row['P-Value'] if (row.name in first_idx) else '', axis=1)
    table1_df['Test'] = table1_df['Test'].replace({'Kruskal-Wallis': 'Mann-Whitney'})

    return table1_df

def analyze_hgb(df_blood_draws, pod_time="POD1", title=None):
    """
    Compare hemoglobin from first draw to a specified POD.
    
    Parameters
    ----------
    df_blood_draws : pd.DataFrame
        DataFrame with columns: ['StudyID','Hemoglobin','Time',
                                 'time_from_injury_to_draw_hours','Draw_date']
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
        df_blood_draws.dropna(subset=['time_from_injury_to_draw_hours'])
        .sort_values(['StudyID','time_from_injury_to_draw_hours'])
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

    if variables is None:
        variables = ['R_time', 'K_time',  'MA', 'LY30', 'ACT','Alpha_Angle']
        # df[variables] = pd.to_numeric(df[variables], errors='coerce')

    if custom_order is None:
        custom_order = ['Pre_op', 'POD1', 'POD3', 'POD5', 'POD7', 'Week2', 'Week4', 'Week6', 'Month3']

    if var_labels is None:
        var_labels = {var: var.replace('_',' ').title() for var in variables}
        
    df = df.copy()
    df = df[~df['Time'].isna()]
    df['Time'] = pd.Categorical(df['Time'], categories=custom_order, ordered=True)

    if palette is None and hue is not None:
        palette = sns.color_palette("Set2", n_colors=df[hue].nunique())

    for var in variables:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(
            x='Time', y=var, data=df,
            hue=hue, style=style,
            estimator='mean', errorbar='se',
            markers=True, linewidth=2,
            palette=palette,
            dashes={"No": (4, 2), "Yes": (None, None)}
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(var_labels.get(var, var), fontsize=12)  
        if legend_title:
            ax.legend(title=legend_title, loc=2)
        ax.set_title(var_labels.get(var, var), fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.margins(0)
        plt.show()


