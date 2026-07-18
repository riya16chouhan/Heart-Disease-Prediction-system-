import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib


def main():
    df = pd.read_csv('heart.csv')

    # Map dataset categories to the labels used in the Streamlit app
    df['Sex'] = df['Sex'].map({'M': 'Male', 'F': 'Female'})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 'Yes', 'N': 'No'})

    # Rename Age to lowercase `age` to match app's input key
    df = df.rename(columns={'Age': 'age'})

    # Features and target
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    # One-hot encode categorical columns that app expects
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ST_Slope', 'ExerciseAngina']
    X = pd.get_dummies(X, columns=categorical_cols)

    # Keep track of expected columns (app will fill missing with zeros)
    expected_columns = list(X.columns)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train a simple KNN classifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_scaled, y)

    # Save artifacts with filenames expected by the app
    joblib.dump(model, 'KNN_heart.pkl')
    joblib.dump(scaler, 'scaler (5).pkl')
    joblib.dump(expected_columns, 'columns.pkl')

    print('Saved: KNN_heart.pkl, scaler (5).pkl, columns.pkl')


if __name__ == '__main__':
    main()
