import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

# Set visualization styles
#sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (10, 6)

def get_column_types(df: pd.DataFrame, categorical_threshold: int = 50) -> tuple:
    """
    Identifies numerical and categorical columns. PostalCode and RegistrationYear 
    are treated as categorical if their unique count is below a threshold.
    """
    numerical_cols = []
    categorical_cols = []

    for col in df.columns:
        # Exclude IDs from being plotted
        if 'ID' in col or 'Code' in col or 'mmcode' in col or 'Year' in col:
            # Treat high-cardinality codes as categorical for grouping later, 
            # but don't try to plot bar charts for PostalCode (too many unique values)
            if df[col].nunique() < categorical_threshold:
                 categorical_cols.append(col)
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check for binary features (which should be treated as categorical)
            if df[col].nunique() <= 2:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            # Use a threshold to avoid plotting bar charts for columns like PostalCode or Model with millions of unique values
            if df[col].nunique() < categorical_threshold:
                categorical_cols.append(col)
            else:
                # Flag high-cardinality object columns (like Model or Make) 
                # to be handled separately or ignored in a full bar chart plot
                pass
                
    return numerical_cols, categorical_cols

# Assuming df_clean is your final processed DataFrame
# numerical_features, categorical_features = get_column_types(df_clean)

def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: list):
    """
    Plots histograms and Box plots for numerical variables.
    """
    print("\n## 📈 Histograms (Numerical Variables)")
    
    # Define a subset of key columns for focused analysis
    key_numerical = ['TotalClaims', 'TotalPremium', 'SumInsured', 'ExcessSelected', 'VehicleAge']
    cols_to_plot = [col for col in key_numerical if col in numerical_cols]

    for col in cols_to_plot:
        # Create a figure with two subplots: histogram and box plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram (Density Plot)
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f'Distribution of {col}', fontsize=14)
        axes[0].set_xlabel(col)
        axes[0].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
        axes[0].legend()

        # Box Plot (for identifying outliers) 

        #[Image of Box Plot]

        sns.boxplot(x=df[col].dropna(), ax=axes[1])
        axes[1].set_title(f'Box Plot of {col} (Outliers)', fontsize=14)

        plt.tight_layout()
        plt.show()

# Example usage:
# plot_numerical_distributions(df_clean, numerical_features)

def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: list):
    """
    Plots bar charts (Count Plots) for categorical variables.
    Focuses on variables with fewer unique categories for clear visualization.
    """
    print("\n## 📊 Bar Charts (Categorical Variables)")
    
    # Define a subset of key categorical columns for hypothesis testing
    key_categorical = ['Province', 'Gender', 'MaritalStatus', 'AlarmImmobiliser', 'TrackingDevice', 'LegalType']
    cols_to_plot = [col for col in key_categorical if col in categorical_cols and df[col].nunique() < 20] # Limit to 20 unique values for readability

    for col in cols_to_plot:
        plt.figure(figsize=(10, 6))
        
        # Calculate value counts
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'Count']
        
        # Plot count plot
        sns.barplot(x='Count', y=col, data=value_counts, palette="viridis")
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

# Example usage:
# plot_categorical_distributions(df_clean, categorical_features)

def analyze_categorical_geography(df: pd.DataFrame, comparison_cols: list):
    """
    Analyzes the distribution of categorical columns across different Provinces.
    """
    print("\n--- Analyzing Categorical Trends Across Provinces ---")
    
    # Ensure the comparison columns exist
    comparison_cols = [col for col in comparison_cols if col in df.columns]

    for col in comparison_cols:
        # Calculate the count of each category within each Province
        province_counts = df.groupby('Province')[col].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()

        # Filter the top N categories to keep the visualization clear (e.g., top 5)
        top_categories = df[col].value_counts().nlargest(5).index
        province_counts_filtered = province_counts[province_counts[col].isin(top_categories)]

        plt.figure(figsize=(12, 6))
        
        # Use a grouped bar chart (or stacked bar chart if preferred)
        sns.barplot(
            data=province_counts_filtered,
            x='Province',
            y='Percentage',
            hue=col,
            palette='tab20'
        )
        plt.title(f'Distribution of {col} by Province (Top 5 Categories)', fontsize=14)
        plt.xlabel('Province')
        plt.ylabel(f'Percentage of Policies (%)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

def analyze_numerical_geography(df: pd.DataFrame, numerical_cols: list):
    """
    Analyzes the mean of numerical columns across different Provinces.
    """
    print("\n--- Analyzing Numerical Trends Across Provinces ---")
    
    # Ensure the comparison columns exist
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    for col in numerical_cols:
        # Calculate the mean of the numerical feature by Province
        province_means = df.groupby('Province')[col].mean().sort_values(ascending=False).reset_index(name=f'Mean_{col}')

        plt.figure(figsize=(10, 6))
        
        # Use a bar chart to compare means 
        sns.barplot(
            data=province_means,
            x='Province',
            y=f'Mean_{col}',
            palette='coolwarm'
        )
        plt.title(f'Average {col} by Province', fontsize=14)
        plt.xlabel('Province')
        plt.ylabel(f'Mean {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
def plot_loss_ratio_by_province(df: pd.DataFrame):
    """Calculates and plots the Loss Ratio by Province."""
    
    # Aggregate Claims and Premium by Province
    province_summary = df.groupby('Province').agg(
        TotalClaims=('TotalClaims', 'sum'),
        TotalPremium=('TotalPremium', 'sum')
    ).reset_index()

    # Calculate Loss Ratio
    province_summary['LossRatio'] = province_summary['TotalClaims'] / province_summary['TotalPremium']
    
    # Sort for visualization
    province_summary = province_summary.sort_values('LossRatio', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=province_summary,
        x='Province',
        y='LossRatio',
        palette='Spectral' # Use a color palette to distinguish high/low risk
    )
    plt.title('Loss Ratio (Claims / Premium) by Province', fontsize=14)
    plt.xlabel('Province')
    plt.ylabel('Loss Ratio')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_box_plots_for_outliers_separate(df: pd.DataFrame, numerical_cols: list):
    """
    Generates a new, individual box plot figure for each key numerical column.
    Uses context manager for localized style and explicitly clears the figure 
    buffer to prevent overlap.
    """
    print("\n## 🔎 Outlier Detection: Individual Box Plots (Minimized & Gridless)")
    
    key_numerical_cols = ['TotalClaims', 'TotalPremium', 'SumInsured', 
                          'ExcessSelected', 'kilowatts', 'VehicleAge',
                          'CapitalOutstanding', 'cubiccapacity']
    
    cols_to_plot = [col for col in key_numerical_cols if col in numerical_cols]
    
    if not cols_to_plot:
        print("No numerical columns found to plot.")
        return

    # Use the context manager to apply 'white' style only within this block
    with sns.axes_style("white"):
        for col in cols_to_plot:
            # Create a new figure for each column, using a smaller size (e.g., 5x4)
            plt.figure(figsize=(5, 4), dpi=100) 
            
            # Plotting the box plot 
            sns.boxplot(y=df[col].dropna())
            
            # Use suptitle to ensure the title is on the figure itself, not just the axes
            plt.suptitle(f'Box Plot for {col}', fontsize=12) 
            plt.ylabel(col)
            
            # Ensure no grid lines are present on the axes just in case
            plt.gca().yaxis.grid(False) 
            
            plt.tight_layout()
            plt.show() 
            
            # CRITICAL ADDITION: Clear and close the figure buffer after displaying
            plt.clf() 
            plt.close()

def filter_gender_for_testing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame to include only policies with specified Gender 
    ('Male' or 'Female') for the A/B hypothesis test.
    
    Returns:
        A new DataFrame containing only the specified gender policies.
    """
    
    # Define the values to keep
    specified_genders = ['Male', 'Female']
    
    # Filter the DataFrame
    df_gender_test = df[df['Gender'].isin(specified_genders)].copy()
    
    original_count = len(df)
    filtered_count = len(df_gender_test)
    
    print("--- Gender Data Filtering ---")
    print(f"Original Policies Count: {original_count:,.0f}")
    print(f"Filtered Policies Count (Male/Female only): {filtered_count:,.0f}")
    print(f"Data retained: {filtered_count / original_count * 100:.2f}%")
    
    return df_gender_test

def run_frequency_test(group_A, group_B, feature_name):
    """Performs Z-test for Claim Frequency (proportions)."""
    n_A = len(group_A)
    n_B = len(group_B)
    # Check for empty groups before proceeding
    if n_A < 30 or n_B < 30: # Use 30 as a common minimum threshold
        print(f"Skipping test for {feature_name}: Sample size too small (A={n_A}, B={n_B}).")
        return None, None, np.nan, np.nan # Return NaN instead of running the unstable test
    
    successes_A = group_A['HasClaim'].sum()
    successes_B = group_B['HasClaim'].sum()

    stat, p_value = proportions_ztest(
        count=[successes_A, successes_B], 
        nobs=[n_A, n_B]
    )
    
    freq_A = successes_A / n_A
    freq_B = successes_B / n_B
    
    return stat, p_value, freq_A, freq_B

def run_mean_test(group_A, group_B, metric_col, feature_name):
    """Performs Two-Sample T-test for numerical metrics (Severity/Margin)."""
    # Use Welch's t-test (equal_var=False) as variances are often unequal in real data
    # Drop NaNs just in case, though they should be clean here
    stat, p_value = stats.ttest_ind(
        group_A[metric_col].dropna(), 
        group_B[metric_col].dropna(), 
        equal_var=False 
    )
    
    mean_A = group_A[metric_col].mean()
    mean_B = group_B[metric_col].mean()
    
    return stat, p_value, mean_A, mean_B