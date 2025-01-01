import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import warnings
import pickle
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer

# Load your data
df = pd.read_excel('datasets\\CommerceOccupationsnew.xlsx',sheet_name='Occupations')

# Label Encoding for categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split the data into features (X) and target (y)
X = df.drop(columns=['Occupation'])  # Features
y = df['Occupation']  # Target

# Handle missing values (optional step)
# imputer = SimpleImputer(strategy='mean')
# X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)  # Reassign column names

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Feature importance extraction (optional: to select top features)
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_n_features = 20  # Adjust this to choose the top n features
important_features = feature_importances.head(top_n_features).index
X_train_top = X_train[important_features]
X_test_top = X_test[important_features]

# Retrain the model using the top features
model_top_features = RandomForestClassifier(random_state=42)
model_top_features.fit(X_train_top, y_train)

# Make predictions and evaluate the model with top features
y_pred_top = model_top_features.predict(X_test_top)
print("Accuracy with Top Features:", accuracy_score(y_test, y_pred_top))
print("Classification Report with Top Features:\n", classification_report(y_test, y_pred_top))

# Cross-validation with F1 Score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')
f1_scores = cross_val_score(model_top_features, X_resampled, y_resampled, cv=cv, scoring=f1_scorer)
print("Cross-Validated F1 Scores:", f1_scores)
print("Mean F1 Score:", f1_scores.mean())

# Save the trained model and label encoders using pickle
with open('career_prediction_model.pkl', 'wb') as model_file:
    pickle.dump(model_top_features, model_file)

with open('label_encoders.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoders, encoder_file)

# Optionally, print overall model accuracy on train and test data
y_train_pred = model_top_features.predict(X_train_top)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Overall Training Accuracy:", train_accuracy)

y_test_pred = model_top_features.predict(X_test_top)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Overall Testing Accuracy:", test_accuracy)
