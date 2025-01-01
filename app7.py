from flask import Flask, render_template, request, redirect, url_for
import joblib
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)



# Load models and label encoders for each field
model_commerce = pickle.load(open('career_prediction_model.pkl', 'rb'))
label_encoders_commerce = pickle.load(open('label_encoders.pkl', 'rb'))

model_science = joblib.load('Science_Model_Categorical.pkl')
label_encoders_science = joblib.load('label_encoders_categorical.pkl')

model_arts = joblib.load('arts_model.pkl')
label_encoders_arts = joblib.load('arts_label_encoders.pkl')

# Load datasets
skills_data_commerce = pd.read_excel('datasets\\CommerceOccupationsnew.xlsx', sheet_name='Sheet4')
skills_data_science = pd.read_excel('datasets\\scienceskillsnew.xlsx')
skills_data_arts = pd.read_excel('datasets\\ArtsOccupationskills.xlsx', sheet_name='Skills')

# Survey questions
questions = [
    {"question": "Favorite type of work?", "options": ["Creative projects", "Research", "Business management", "Social work"]},
    {"question": "Environment do you prefer to work in?", "options": ["Studio", "Laboratory", "Office", "Remote work"]},
    {"question": "What drives you to succeed?", "options": ["Self-expression", "Discovery", "Financial growth", "Helping others"]},
    {"question": "How do you prefer to learn new things?", "options": ["Hands-on experience", "Experimentation", "Case studies", "Online courses"]},
    {"question": "Ideal job role?", "options": ["Artist", "Researcher", "Entrepreneur", "Social worker"]},
    {"question": "Motivates you the most?", "options": ["Creativity", "Solving problems", "Achieving goals", "Making a difference"]},
    {"question": "What kind of challenges do you enjoy?", "options": ["Expressing unique ideas", "Solving complex problems", "Overcoming market competition", "Social challenges"]},
    {"question": "What type of career suits your skills best?", "options": ["Visual arts", "Engineering", "Finance", "Technology development"]},
]

# Initialize scores
scores = {'Science': 0, 'Commerce': 0, 'Arts': 0, 'Other': 0}

def calculate_scores(answer):
    """Update scores based on the selected answer."""
    global scores
    print(f"Answer selected: {answer}")  # Debug print statement
    # Arts-related answers
    if answer in ["Creative projects", "Studio", "Self-expression", "Hands-on experience", "Artist", "Creativity", "Expressing unique ideas", "Visual arts"]:
        scores["Arts"] += 1
    # Science-related answers
    elif answer in ["Research", "Laboratory", "Discovery", "Experimentation", "Researcher", "Solving problems", "Solving complex problems", "Engineering"]:
        scores["Science"] += 1
    # Commerce-related answers
    elif answer in ["Business management", "Office", "Financial growth", "Case studies", "Entrepreneur", "Achieving goals", "Overcoming market competition", "Finance"]:
        scores["Commerce"] += 1
    # Other-related answers
    elif answer in ["Social work", "Remote work", "Helping others", "Online courses", "Social worker", "Making a difference", "Social challenges", "Technology development"]:
        scores["Other"] += 1
    print(f"Scores after answer: {scores}")  # Debug print statement





@app.route('/vision')
def vision():
    """Render the Vision page."""
    return render_template('vision.html')

@app.route('/', methods=['GET'])
def landing():
    """Render the landing page."""
    return render_template('landing.html')

@app.route('/contact')
def contact():
    return render_template('contactus.html')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    """Render the survey and handle responses."""
    
    if request.method == 'POST':
        # Loop through the questions and update scores based on answers
        for idx, question in enumerate(questions):
            answer = request.form.get(f"answer_{idx}")
            if answer:
                # Calculate scores based on the selected answer
                calculate_scores(answer)
                print(f"Answer for question {idx}: {answer}")
                print(f"Current Scores: {scores}")

        # After all answers are processed, determine the recommended field
        recommended_field = max(scores, key=scores.get)

        print(f"Recommended Field: {recommended_field}")  # This will print the correct field

        # Redirect to the prediction page for the recommended field
        return redirect(url_for('redirect_field', field=recommended_field))

    # Pass both questions and their index to the template
    return render_template('survey.html', questions=questions)


@app.route('/redirect/<field>')
def redirect_field(field):
    """Display prediction and allow user to proceed to the next field."""
    if field == "Arts":
        return render_template('prediction_result.html', field=field, next_page=url_for('arts_predict'))
    elif field == "Commerce":
        return render_template('prediction_result.html', field=field, next_page=url_for('commerce_predict'))
    elif field == "Science":
        return render_template('prediction_result.html', field=field, next_page=url_for('science_predict'))
    elif field == "Other":
        return render_template('prediction_result.html', field=field, next_page=url_for('other_predict'))
    else:
        return "Invalid field"

# ========================== Commerce Prediction ==========================
@app.route('/commerce_predict', methods=['GET', 'POST'])
def commerce_predict():
    """Render the commerce-specific prediction page."""
    if request.method == 'GET':
        return render_template('commerce.html')
    try:
        numerical_features = [
            'Business Communication Skills', 'Decision-Making', 'Marketing Knowledge',
            'Risk Management', 'Taxation Knowledge'
        ]
        numerical_values = [int(request.form[feature]) for feature in numerical_features]
        categorical_features = [
            'Financial Analysis Skills', 'Accounting Knowledge', 'Negotiation Skills',
            'Team Management', 'Financial Regulations', 'Customer Service Skills',
            'Sales Acumen', 'Technological Adaptability', 'Market Research Skills',
            'Strategic Planning', 'Budgeting & Forecasting', 'Data Analysis',
            'Investment Knowledge', 'Product Development Insight', 'Supply Chain Knowledge'
        ]
        categorical_values = [request.form[feature] for feature in categorical_features]
        encoded_categorical_values = [
            label_encoders_commerce[feature].transform([value])[0]
            for feature, value in zip(categorical_features, categorical_values)
        ]
        input_features = np.array(numerical_values + encoded_categorical_values).reshape(1, -1)
        prediction = model_commerce.predict(input_features)[0]
        occupation = label_encoders_commerce['Occupation'].inverse_transform([prediction])[0]
        career_details = skills_data_commerce[skills_data_commerce['Field of Interest'] == occupation]
        if career_details.empty:
            return render_template(
                'error.html',
                message=f"No career details found for the predicted occupation: {occupation}."
            )
        career_details = career_details.iloc[0]
        return render_template(
            'result.html',
            prediction=occupation,
            foundational_skills=career_details['Foundational Skills'],
            intermediate_skills=career_details['Intermediate-Level Skills'],
            professional_skills=career_details['Professional-Level Skills']
        )
    except Exception as e:
        return render_template('error.html', message=f"An error occurred: {str(e)}")

# ========================== Science Prediction ==========================
@app.route('/science_predict', methods=['GET', 'POST'])
def science_predict():
    """Render the science-specific prediction page."""
    if request.method == 'GET':
        return render_template('Science.html')
    try:
        categorical_features = [
            'Experiment Comfort', 'Problem Solving', 'Math Comfort', 'Tech Interest',
            'Field vs. Lab', 'Long-Term Projects', 'Attention to Detail', 'Real-World Applications',
            'Work Style', 'Adaptability', 'Interest in Reading', 'Creativity',
            'Patient Interaction', 'Design Interest', 'Technical Comfort', 'Bio/Chem Interest',
            'Earth/Space Interest', 'Environmental Interest', 'Human Behavior'
        ]
        categorical_values = [request.form[feature] for feature in categorical_features]
        encoded_values = [label_encoders_science[feature].transform([value])[0] 
                          for feature, value in zip(categorical_features, categorical_values)]
        input_features = np.array(encoded_values).reshape(1, -1)
        prediction = model_science.predict(input_features)[0]
        field_of_interest = label_encoders_science['Field of Interest'].inverse_transform([prediction])[0]
        career_details = skills_data_science[skills_data_science['Field of Interest'] == field_of_interest]
        if career_details.empty:
            return render_template('error.html', 
                                   message=f"No career details found for: {field_of_interest}.")
        career_details = career_details.iloc[0]
        return render_template(
            'result.html',
            prediction=field_of_interest,
            foundational_skills=career_details['Foundational Skills'],
            intermediate_skills=career_details['Intermediate-Level Skills'],
            professional_skills=career_details['Professional-Level Skills']
        )
    except Exception as e:
        return render_template('error.html', message=f"An error occurred: {str(e)}")

# ========================== Arts Prediction ==========================
@app.route('/arts_predict', methods=['GET', 'POST'])
def arts_predict():
    """Render the arts-specific prediction page."""
    if request.method == 'GET':
        return render_template('Arts.html')
    try:
        numerical_features = [
            'Social Awareness', 'Communication Skills', 'Empathy and Counseling Skills',
            'Critical Thinking', 'Cultural Literacy', 'Research Skills'
        ]
        numerical_values = [int(request.form[feature]) for feature in numerical_features]
        categorical_features = [
            'Public Speaking', 'Writing and Editing', 'Interpersonal Skills',
            'Ethical Judgment', 'Problem-Solving', 'Legal Knowledge',
            'Analytical Skills', 'Negotiation Skills', 'Advocacy',
            'Strategic Thinking', 'Language Proficiency', 'Emotional Intelligence'
        ]
        categorical_values = [request.form[feature] for feature in categorical_features]
        encoded_categorical_values = [
            label_encoders_arts[feature].transform([value])[0]
            for feature, value in zip(categorical_features, categorical_values)
        ]
        input_features = np.array(numerical_values + encoded_categorical_values).reshape(1, -1)
        prediction = model_arts.predict(input_features)[0]
        occupation = label_encoders_arts['Field of Interest'].inverse_transform([prediction])[0]
        career_details = skills_data_arts[skills_data_arts['Field of Interest'] == occupation]
        if career_details.empty:
            return render_template(
                'error.html',
                message=f"No career details found for the predicted occupation: {occupation}."
            )
        career_details = career_details.iloc[0]
        return render_template(
            'result.html',
            prediction=occupation,
            foundational_skills=career_details['Foundational Skills'],
            intermediate_skills=career_details['Intermediate-Level Skills'],
            professional_skills=career_details['Professional-Level Skills']
        )
    except Exception as e:
        return render_template('error.html', message=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
