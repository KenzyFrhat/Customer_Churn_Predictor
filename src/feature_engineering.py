import pandas as pd
# from preprocessing import df
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split 
import joblib
import logging


df = pd.read_csv(r"Data\cleaned\customer_churn_cleaned.csv")

ONE_HOT_ENCODING = ["gender",
                    "MultipleLines", 
                    "InternetService", 
                    "PaymentMethod"]

ORDINAL_COLS = ["Contract"]

NUMERICAL_COLS = [
    "tenure", "TotalCharges", "MonthlyCharges"
]

DEUPLICATED_COLS = ["PhoneService"] #this columns became duplicated columns after one hot ecoding operation



def split_data(df, test_size = 0.2, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns = ["Churn"]), 
        df["Churn"], 
        test_size = test_size, 
        random_state = random_state
    )
    return  X_train, X_test, y_train, y_test


def numerical_col_transformer():
    return Pipeline([
    ("normalizer", StandardScaler())
])


def categorical_col_transformer():
    return Pipeline([
    (
                "hot encoding", 
                OneHotEncoder(handle_unknown = "ignore"), 
                ONE_HOT_ENCODING
    ), 
                (

                "Oridnal encoding",
                OrdinalEncoder(
                    categories = [
                        ["Month-to-month", "One year", "Two year"]
                    ]
                        )
                )

])



def Pipline():
    preprocessor = ColumnTransformer(
        transformers = [
            (
                "numerical_cols", numerical_col_transformer(), NUMERICAL_COLS
            ), 
            (
                "hot encoding", 
                OneHotEncoder(handle_unknown = "ignore"), 
                ONE_HOT_ENCODING
            ), 
            (
                "Oridnal encoding",
                OrdinalEncoder(
                    categories = [
                        ["Month-to-month", "One year", "Two year"]
                    ]
                        ), ORDINAL_COLS
            ),
            (
                "drop_deplicated_cols", 
                "drop", 
                DEUPLICATED_COLS
            )
        ], 
        remainder = "passthrough"
    )
    return preprocessor 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Splitting Data...")
    X_train, X_test, y_train, y_test  = split_data(df)

    logging.info("Preprocessing....")
    preprocessor = Pipline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logging.info("Saving splited and preprocessed data...")
    joblib.dump(X_train, r"Data\ML_ready\X_train.pkl")
    joblib.dump(X_test, r"Data\ML_ready\X_test.pkl")
    joblib.dump(y_train, r"Data\ML_ready\y_train.pkl")
    joblib.dump(y_test, r"Data\ML_ready\y_test.pkl")

