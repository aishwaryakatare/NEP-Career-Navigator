import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
import joblib  # For saving and loading the model

warnings.filterwarnings('ignore')

# Load data from Excel file
df = pd.read_excel('datasets\\science1.xlsx')  # Adjust the path if necessary

# Label encoding for categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target
X = df.drop(columns=['Field of Interest'])  # Features (all categorical variables)
y = df['Field of Interest']  # Target (categorical)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the initial model to determine feature importance
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importances and select top n important features
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_n_features = 19  # Choose the top n features
important_features = feature_importances.head(top_n_features).index
X_train_top = X_train[important_features]
X_test_top = X_test[important_features]

# Retrain the model with the top features
model_top_features = RandomForestClassifier(random_state=42)
model_top_features.fit(X_train_top, y_train)

# Predict and evaluate with top features
y_pred_top = model_top_features.predict(X_test_top)
print("Accuracy with Top Features:", accuracy_score(y_test, y_pred_top))
print("Classification Report with Top Features:\n", classification_report(y_test, y_pred_top))

# Save the trained model
model_path = 'Science_Model_Categorical.pkl'
joblib.dump(model_top_features, model_path)
print(f"Model successfully saved to {model_path}.")

# Save the label encoders
encoder_path = 'label_encoders_categorical.pkl'
joblib.dump(label_encoders, encoder_path)
print(f"Label encoders successfully saved to {encoder_path}.")
