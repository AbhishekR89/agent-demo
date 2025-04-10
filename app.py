from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

app = Flask(__name__)

# Check if model exists, if not, train it dynamically
if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
    import pandas as pd

    data = {
        'issue': [
            'internet not working',
            'slow internet speed',
            'billing issue',
            'payment not processed',
            'router not connecting',
            'technical support needed',
            'account suspended',
            'refund request',
            'unable to login',
            'change password request'
        ],
        'category': [
            'Technical',
            'Technical',
            'Billing',
            'Billing',
            'Technical',
            'Technical',
            'Account',
            'Billing',
            'Account',
            'Account'
        ]
    }

    df = pd.DataFrame(data)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['issue'])
    model = MultinomialNB()
    model.fit(X, df['category'])

    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
else:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

department_mapping = {
    'Technical': 'Technical Support Team',
    'Billing': 'Billing Department',
    'Account': 'Account Management Team'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    user_message = request.form['message']
    transformed_message = vectorizer.transform([user_message])
    prediction = model.predict(transformed_message)[0]
    assigned_team = department_mapping.get(prediction, 'General Support')

    # Prepare result
    result = {
        'issue': user_message,
        'category': prediction,
        'assigned_team': assigned_team,
        'status': 'Service request has been raised successfully!'
    }

    return render_template('index.html', response=result)

if __name__ == '__main__':
    app.run(debug=True)
