import pandas as pd
import logging 
 

DROP_COLS = ["customerID"]

INTERNET_SERVICE_COLS = [
    "OnlineBackup",
    "OnlineSecurity",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

YES_NO_COLS = [
    "OnlineBackup",
    "OnlineSecurity",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]


def read_df(file_path):
    return pd.read_csv(file_path)


def save_df(df, file_path):
    df.to_csv(file_path, index=False)


def drop_columns(df, columns):
    return df.drop(columns=columns)


def remove_duplicates(df):
    return df.drop_duplicates().copy()


def convert_to_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fill_totalcharges_missing_values(df):
    nulls = df["TotalCharges"].isna()

    df.loc[nulls, "TotalCharges"] = (
        df.loc[nulls, "tenure"]
        * df.loc[nulls, "MonthlyCharges"]
    )

    return df


def replace_category_value(df, old_value, new_value, columns):
    for col in columns:
        df[col] = df[col].replace(old_value, new_value)

    return df


def yes_no_to_bool(df, columns):
    mapping = {
        "No": False,
        "Yes": True,
    }

    for col in columns:
        df[col] = df[col].map(mapping)

    return df


def convert_binary_to_bool(df, columns):
    for col in columns:
        df[col] = df[col].astype(bool)

    return df


def handle_totalcharges_column(df):
    df = convert_to_numeric(df, ["TotalCharges"])
    df = fill_totalcharges_missing_values(df)

    return df


def handle_internet_service_columns(df):
    df = replace_category_value(
        df,
        "No internet service",
        "No",
        INTERNET_SERVICE_COLS,
    )

    return df


def handle_nominal_columns(df):
    df = handle_internet_service_columns(df)

    df = yes_no_to_bool(
        df,
        YES_NO_COLS,
    )

    df = convert_binary_to_bool(
        df,
        ["SeniorCitizen"],
    )

    return df


def preprocess_data(df):
    logging.basicConfig(level=logging.INFO)
    logging.info("Dropping unsed columns")
    df = drop_columns(df, DROP_COLS)

    logging.info("Removing duplicates")
    df = remove_duplicates(df)

    logging.info("Handing TotalCharges column")
    df = handle_totalcharges_column(df)

    logging.info("Handing nominal columns")
    df = handle_nominal_columns(df)

    return df


def main():
    input_path = (
        r"Data\Raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    output_path = (
        r"data/cleaned/"
        "customer_churn_cleaned.csv"
    )

    df = read_df(input_path)

    df = preprocess_data(df)

    save_df(df, output_path)


if __name__ == "__main__":
    main()