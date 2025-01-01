import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from imblearn.over_sampling import SMOTE
import warnings
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_excel('datasets\\ArtsOccupationskills.xlsx',sheet_name='Occupations')

# Encode categorical columns using LabelEncoder
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate the dataset into features (X) and target (y)
X = df.drop(columns=['Occupation'])  # Features
y = df['Occupation']  # Target

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the initial RandomForest model on all features
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Extract feature importances and select top N features
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_n_features = 18  # Number of top features to select
important_features = feature_importances.head(top_n_features).index

# Create new training and testing datasets with only the top features
X_train_top = X_train[important_features]
X_test_top = X_test[important_features]

# Retrain the RandomForest model using only the top features
model_top_features = RandomForestClassifier(random_state=42)
model_top_features.fit(X_train_top, y_train)

# Evaluate the model with top features
y_pred_top = model_top_features.predict(X_test_top)
print("Accuracy with Top Features:", accuracy_score(y_test, y_pred_top))
print("Classification Report with Top Features:\n", classification_report(y_test, y_pred_top))

# Perform cross-validation using the F1 score
cv = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')
f1_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring=f1_scorer)
print("Cross-Validated F1 Scores:", f1_scores)
print("Mean F1 Score:", f1_scores.mean())

# Evaluate training and testing accuracy with the original model
y_train_pred = model.predict(X_train)
train_accuracy_overall = accuracy_score(y_train, y_train_pred)
print("Overall Training Accuracy:", train_accuracy_overall)

y_test_pred = model.predict(X_test)
test_accuracy_overall = accuracy_score(y_test, y_test_pred)
print("Overall Testing Accuracy:", test_accuracy_overall)

# Save the model trained on top features using joblib
model_filename = "arts_model.pkl"
joblib.dump(model_top_features, model_filename)
print(f"Model saved to {model_filename}")

# Save label encoders
label_encoder_filename = "arts_label_encoders.pkl"
joblib.dump(label_encoders, label_encoder_filename)
print(f"Label encoders saved to {label_encoder_filename}")

# Load the saved model
loaded_model = joblib.load(model_filename)
print("Model loaded successfully.")

# Evaluate the loaded model
y_pred_loaded = loaded_model.predict(X_test_top)
print("Accuracy with Loaded Model:", accuracy_score(y_test, y_pred_loaded))
print("Classification Report with Loaded Model:\n", classification_report(y_test, y_pred_loaded))
