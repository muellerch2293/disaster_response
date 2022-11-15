# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Dependencies: 
(venv) at00202766@atca-050132 disaster_response % pip3 list
Package            Version
------------------ ----------
click              8.1.3
contourpy          1.0.6
cycler             0.11.0
Flask              2.2.2
fonttools          4.38.0
greenlet           2.0.0
importlib-metadata 5.0.0
itsdangerous       2.1.2
Jinja2             3.1.2
joblib             1.2.0
kiwisolver         1.4.4
MarkupSafe         2.1.1
matplotlib         3.6.2
nltk               3.7
numpy              1.23.4
packaging          21.3
pandas             1.5.1
Pillow             9.3.0
pip                22.3.1
plotly             5.11.0
pyparsing          3.0.9
python-dateutil    2.8.2
pytz               2022.6
regex              2022.10.31
scikit-learn       1.1.3
scipy              1.9.3
setuptools         60.10.0
six                1.16.0
sklearn            0.0.post1
SQLAlchemy         1.4.42
tenacity           8.1.0
threadpoolctl      3.1.0
tqdm               4.64.1
Werkzeug           2.2.2
wheel              0.38.4
zipp               3.10.0