import os
import json
import logging
import pandas as pd
import joblib
import re
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from flask import current_app as app, render_template, request, jsonify, send_from_directory, send_file
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from db_connection import create_connection, insert_prediction  # Import the necessary functions from db_connection.py

# Define directories
MODEL_SAVE_DIR = "D:/bug sev v2 p4/bug sev v2/models/saved new"
GRAPH_SAVE_DIR = "D:/bug sev v2 p4/bug sev v2/app/static/graphs"

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.makedirs("log")
file_handler = logging.FileHandler(os.path.join("log", "log-predict.txt"))
logger.addHandler(file_handler)

# Create graph save directory if it doesn't exist
if not os.path.exists(GRAPH_SAVE_DIR):
    os.makedirs(GRAPH_SAVE_DIR)

# Load the saved models
VOTING_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "Advanced_Ensemble_Model.pkl")
STACKING_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "Best_Stacking_Model.pkl")
voting_model = joblib.load(VOTING_MODEL_PATH)
stacking_model = joblib.load(STACKING_MODEL_PATH)

# Function to extract features from pure code
def extract_code_features(code):
    code_length = len(code)
    num_comments = len(re.findall(r'//.*|/\*[\s\S]*?\*/|#.*', code))  # Regex to count comments
    num_functions = len(re.findall(r'\bdef\b|\bfunction\b|\bfunc\b|\bvoid\b|\bpublic\b', code))  # Regex to count functions
    return {
        'code_length': code_length,
        'num_comments': num_comments,
        'num_functions': num_functions,
    }

def plot_probabilities(probs, project_name, model_type):
    labels = ['Critical', 'High', 'Medium', 'Low']
    plt.figure(figsize=(10, 6))
    plt.bar(labels, probs, color=['red', 'orange', 'yellow', 'green'])
    plt.xlabel('Severity')
    plt.ylabel('Probability')
    plt.title(f'Probabilities for Project: {project_name} ({model_type})')
    plt.ylim(0, 1)
    project_name_safe = project_name.replace(" ", "_").replace("/", "_")
    graph_path = f'{GRAPH_SAVE_DIR}/probabilities_{project_name_safe}_{model_type}.png'
    plt.savefig(graph_path)
    plt.close()
    return f'/static/graphs/probabilities_{project_name_safe}_{model_type}.png'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/introduction')
def introduction():
    return render_template('intro.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/how-to-use')
def how_to_use():
    return render_template('how_to_use.html')

@app.route('/dataset')
def datasets():
    return render_template('datasets.html')

@app.route('/classifier')
def ensemble_classifier():
    return render_template('ensemble_classifier.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_type = request.args.get('model')

    if isinstance(data, dict):  # Convert single dictionary to list of dictionaries
        data = [data]

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Extract features from the pure code
    df_code_features = df['code'].apply(lambda x: extract_code_features(x))

    # Convert the extracted code features to a DataFrame and combine with code metrics
    df_code_features = pd.json_normalize(df_code_features)
    combined_df = pd.concat([df.drop(columns=['code', 'code_comment', 'code_no_comment'], errors='ignore'), df_code_features], axis=1)

    # Define the columns used for prediction (matching the trained model)
    cols = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']

    # Ensure the combined DataFrame has the necessary columns
    for col in cols:
        if col not in combined_df:
            combined_df[col] = 0.0  # Fill missing columns with a default value

    # Ensure the combined DataFrame has the necessary columns
    combined_df = combined_df[cols]
    combined_df = combined_df.astype(float)  # Convert all columns to float

        # Select the appropriate model
    if model_type == 'voting':
        model = voting_model
    else:
        model = stacking_model

    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(combined_df)
    probabilities = model.predict_proba(combined_df)

    # Define the severity mapping
    severity_mapping = {
        3: "Low",
        2: "Medium",
        1: "High",
        0: "Critical"
    }

    threshold = 0.45

    # Process the predictions
    is_buggy = ["Yes" if severity_mapping[pred] != "Low" and max(probabilities[i]) >= threshold else "No" for i, pred in enumerate(predictions)]  # Assuming 3 represents non-buggy cases
    severity = [severity_mapping[pred] for pred in predictions]

    # Prepare the results
    results = []
    for i in range(len(predictions)):
        probs = probabilities[i]
        print(f"Probabilities for project {df.iloc[i]['project_name']}: {probs}")  # Debugging print
        project_name = df.iloc[i]['project_name']
        graph_path = plot_probabilities(probs, project_name, model_type)
        result = {
            "Project Name": project_name,
            "isBuggy": is_buggy[i],
            "Severity": severity[i],
            "Model Type": model_type,
            "Probabilities": {
                "Low": probs[3],
                "Medium": probs[2],
                "High": probs[1],
                "Critical": probs[0]
            },
            "Graph": graph_path
        }
        results.append(result)


        # Save prediction to database
        connection = create_connection()
        insert_prediction(connection, project_name, is_buggy[i], severity[i], probs[3], probs[2], probs[1], probs[0], graph_path, model_type)

    return jsonify(results)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file and (file.filename.endswith('.jsonl') or file.filename.endswith('.csv') or file.filename.endswith('.json')):
        content = file.read().decode('utf-8')
        if file.filename.endswith('.jsonl'):
            try:
                data = [json.loads(line) for line in content.splitlines() if line.strip()]
            except json.JSONDecodeError as e:
                return jsonify({"error": f"JSONDecodeError: {e}"})
        elif file.filename.endswith('.csv'):
            import csv
            import io
            reader = csv.DictReader(io.StringIO(content))
            data = [row for row in reader]
        elif file.filename.endswith('.json'):
            try:
                data = json.loads(content)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError as e:
                return jsonify({"error": f"JSONDecodeError: {e}"})
        # Process the data here
        return jsonify(data)
    else:
        return jsonify({"error": "Invalid file format"})

@app.route('/graphs/<filename>')
def graphs(filename):
    return send_from_directory(os.path.join(app.root_path, GRAPH_SAVE_DIR), filename)

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    data = request.get_json()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    styleN = styles['Normal']

    for item in data:
        project_name = item['Project Name']
        severity = item['Severity']
        is_buggy = item['isBuggy']
        model = item['Model Type']
        probabilities = item['Probabilities']
        graph_path = item['Graph']

        elements.append(Paragraph(f"Project Name: {project_name}", styleN))
        elements.append(Paragraph(f"Severity: {severity}", styleN))
        elements.append(Paragraph(f"Is Buggy: {is_buggy}", styleN))
        elements.append(Paragraph(f"Model Type: {model}", styleN))
        elements.append(Paragraph("Probabilities:", styleN))
        elements.append(Paragraph(f"Critical: {probabilities['Critical']:.4f}", styleN))
        elements.append(Paragraph(f"High: {probabilities['High']:.4f}", styleN))
        elements.append(Paragraph(f"Medium: {probabilities['Medium']:.4f}", styleN))
        elements.append(Paragraph(f"Low: {probabilities['Low']:.4f}", styleN))
        elements.append(Spacer(1, 12))

        graph_full_path = os.path.join(app.root_path, graph_path[1:])
        elements.append(Image(graph_full_path, width=400, height=200))
        elements.append(Spacer(1, 24))

    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='prediction_results.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    from db_connection import create_connection

    app.run(debug=True)
  
