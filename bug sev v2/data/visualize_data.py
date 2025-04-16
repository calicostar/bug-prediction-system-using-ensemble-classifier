import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE

DATA_DIR = "D:/bug sev v2 p3/bug sev v2/data"
OUTPUT_DIR = "D:/bug sev v2 p3/bug sev v2/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class Bug:
    def __init__(self, project_name, project_version, severity, code, code_comment, code_no_comment, lc, pi, ma, nbd, ml, d, mi, fo, r, e):
        self.project_name = project_name
        self.project_version = project_version
        self.label = severity
        self.code = code
        self.code_comment = code_comment
        self.code_no_comment = code_no_comment
        self.lc = lc
        self.pi = pi
        self.ma = ma
        self.nbd = nbd
        self.ml = ml
        self.d = d
        self.mi = mi
        self.fo = fo
        self.r = r
        self.e = e

def clean_data(df):
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)  # Fill missing categorical data with mode
        else:
            df[col].fillna(df[col].mean(), inplace=True)  # Fill missing numerical data with mean

    # Separate boolean columns and handle them separately
    bool_cols = df.select_dtypes(include=['bool']).columns
    non_bool_df = df.drop(columns=bool_cols)
    
    # Remove outliers using IQR
    Q1 = non_bool_df.quantile(0.25)
    Q3 = non_bool_df.quantile(0.75)
    IQR = Q3 - Q1
    
    # Align the DataFrame and Series
    non_bool_df, Q1 = non_bool_df.align(Q1, axis=1, copy=False)
    non_bool_df, Q3 = non_bool_df.align(Q3, axis=1, copy=False)
    
    condition = ~((non_bool_df < (Q1 - 1.5 * IQR)) | (non_bool_df > (Q3 + 1.5 * IQR))).any(axis=1)
    non_bool_df = non_bool_df[condition]
    
    # Reattach boolean columns
    df = pd.concat([non_bool_df, df[bool_cols]], axis=1)
    
    return df

def split_dataset():
    d4j = pd.read_csv(os.path.join(DATA_DIR, "d4j_methods_sc_metrics_comments.csv"))
    bugs_jar = pd.read_csv(os.path.join(DATA_DIR, "bugsjar_methods_sc_metrics_comments.csv"))
    
    # Plot boxplots before outlier removal
    plot_boxplots(d4j, ['LC', 'PI', 'MA', 'NBD', 'ML', 'D', 'MI', 'FO', 'R', 'E'], "d4j Data - Before Outlier Removal", "boxplots_before_outlier_removal_d4j.png")
    plot_boxplots(bugs_jar, ['LC', 'PI', 'MA', 'NBD', 'ML', 'D', 'MI', 'FO', 'R', 'E'], "BugsJar Data - Before Outlier Removal", "boxplots_before_outlier_removal_bugsjar.png")
    
    # Clean data
    d4j = clean_data(d4j)
    bugs_jar = clean_data(bugs_jar)
    
    # Convert categorical severity to numerical
    d4j['Severity'] = categorical_to_number(d4j['Severity'], 'd4j')
    bugs_jar['Severity'] = categorical_to_number(bugs_jar['Severity'], 'bugsjar')
    
    bugs = []
    bugs.extend(create_bugs(d4j))
    bugs.extend(create_bugs(bugs_jar))

    df_bugs = pd.DataFrame(bugs)
    df_bugs.drop_duplicates(keep='first', inplace=True)

    # Ensure no NaN values in 'label' column
    df_bugs = df_bugs.dropna(subset=['label'])

    # Stratified split to ensure equal class distribution
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)
    for train_index, test_index in stratified_split.split(df_bugs, df_bugs['label']):
        train_val, test = df_bugs.iloc[train_index], df_bugs.iloc[test_index]
        train, val = train_test_split(train_val, test_size=0.1765, random_state=666, stratify=train_val['label'])
        break

    # Ensure scaling
    cols = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']
    scaler = RobustScaler()
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train.loc[:, cols] = scaler.fit_transform(train[cols])
    val.loc[:, cols] = scaler.transform(val[cols])
    test.loc[:, cols] = scaler.transform(test[cols])

    # Apply SMOTE to handle class imbalance in the training set
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(train[cols], train['label'])
    smote_df = pd.DataFrame(x_train, columns=cols)
    smote_df['label'] = y_train

    # Visualize data
    visualize_data(train, val, test, smote_df)

def create_bugs(df):
    bugs = []
    for index, row in df.iterrows():
        if row["IsBuggy"]:
            bug = Bug(project_name=row["ProjectName"], project_version=row["ProjectVersion"], severity=row["Severity"],
                      code=row["SourceCode"], code_comment=row["CodeComment"],
                      code_no_comment=row["CodeNoComment"], lc=row["LC"], pi=row["PI"], ma=row["MA"],
                      nbd=row["NBD"], ml=row["ML"], d=row["D"], mi=row["MI"], fo=row["FO"], r=row["R"], e=row["E"])
            bugs.append(bug.__dict__)
    return bugs

def categorical_to_number(severity, dataset_type):
    if dataset_type == 'd4j':
        return severity.replace({"Critical": 0, "High": 1, "Medium": 2, "Low": 3})
    elif dataset_type == 'bugsjar':
        return severity.replace({"Blocker": 0, "Critical": 0, "Major": 1, "Minor": 3, "Trivial": 3})

def number_to_categorical(value, dataset_type):
    if dataset_type == 'd4j':
        mapping = {0: "Critical", 1: "High", 2: "Medium", 3: "Low"}
    elif dataset_type == 'bugsjar':
        mapping = {0: "Blocker/Critical", 1: "Major", 3: "Minor/Trivial"}
    return mapping.get(value, value)

def plot_boxplots(df, cols, title, filename):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(cols):
        plt.subplot(len(cols) // 3 + 1, 3, i + 1)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_scatter(df, title, filename, label_col='label'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='lc', y='pi', hue=label_col, palette='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel('lc')
    plt.ylabel('pi')
    plt.legend(title=label_col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_bar(df_before, df_after, title, filename, label_col='label', dataset_type='d4j'):
    plt.figure(figsize=(10, 6))
    before_counts = df_before[label_col].value_counts().sort_index()
    after_counts = df_after[label_col].value_counts().sort_index()
    df_counts = pd.DataFrame({'Before SMOTE': before_counts, 'After SMOTE': after_counts})
    df_counts.index = df_counts.index.map(lambda x: number_to_categorical(x, dataset_type))
    ax = df_counts.plot(kind='bar', ax=plt.gca())
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def visualize_data(train, val, test, smote_df):
    # Columns to visualize
    columns = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']

    # Print class distribution before SMOTE
    pre_smote_distribution = train['label'].value_counts()
    print("Class distribution before SMOTE:")
    print(pre_smote_distribution)
    
    # Save class distribution before SMOTE to a file
    pre_smote_distribution.to_csv(os.path.join(OUTPUT_DIR, "class_distribution_before_smote.csv"))

    # Plotting vertical boxplots for training data before and after outlier removal
    plot_boxplots(train, columns, "Train Data - Before Outlier Removal", "boxplots_before_outlier_removal.png")
    plot_boxplots(train, columns, "Train Data - After Outlier Removal", "boxplots_after_outlier_removal.png")

    # Visualizing scatter plot before SMOTE
    plot_scatter(train, "Before SMOTE", "scatter_before_smote.png", label_col='label')

    # Print class distribution after SMOTE
    post_smote_distribution = smote_df['label'].value_counts()
    print("Class distribution after SMOTE:")
    print(post_smote_distribution)
    
    # Save class distribution after SMOTE to a file
    post_smote_distribution.to_csv(os.path.join(OUTPUT_DIR, "class_distribution_after_smote.csv"))

    # Visualizing scatter plot after SMOTE
    plot_scatter(smote_df, "After SMOTE", "scatter_after_smote.png", label_col='label')

    # Visualizing SMOTE results in a bar graph
    plot_bar(train, smote_df, "Class Distribution Before and After SMOTE", "bar_smote_comparison.png", label_col='label')

if __name__ == '__main__':
    split_dataset()
