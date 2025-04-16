import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE

DATA_DIR = "D:/bug sev v2 p4/bug sev v2/data"

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

def split_dataset(output_dir):
    d4j = pd.read_csv(os.path.join(DATA_DIR, "d4j_methods_sc_metrics_comments.csv"))
    bugs_jar = pd.read_csv(os.path.join(DATA_DIR, "bugsjar_methods_sc_metrics_comments.csv"))
    
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
    train = pd.DataFrame(x_train, columns=cols)
    train['label'] = y_train

    write_bugs(train, "train_scaled", output_dir)
    write_bugs(val, "valid_scaled", output_dir)
    write_bugs(test, "test_scaled", output_dir)

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

def write_bugs(bugs, name, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    jsonl_path = os.path.join(data_dir, f"{name}.jsonl")
    csv_path = os.path.join(data_dir, f"{name}.csv")

    with open(jsonl_path, 'w') as f:
        for bug in bugs:
            f.write(json.dumps(bug) + "\n")

    df = pd.DataFrame(bugs)
    df.to_csv(csv_path, index=False)
    print(f"Files written: {jsonl_path}, {csv_path}")

def categorical_to_number(severity, dataset_type):
    if dataset_type == 'd4j':
        return severity.replace({"Critical": 0, "High": 1, "Medium": 2, "Low": 3})
    elif dataset_type == 'bugsjar':
        return severity.replace({"Blocker": 0, "Critical": 0, "Major": 1, "Minor": 3, "Trivial": 3})

if __name__ == '__main__':
    output_directory = "D:/bug sev v2 p4/bug sev v2/data"
    split_dataset(output_directory)









