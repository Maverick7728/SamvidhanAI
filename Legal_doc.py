import streamlit as st
import pandas as pd
import numpy as np
import spacy
import random
from datetime import datetime, timedelta
import jinja2
from jinja2 import Template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict

# Page configuration
st.set_page_config(page_title="Legal Document Generator", layout="wide")

# App title
st.title("Legal Document Generator with NER and Best Matching Holdings")

# Try to load the spaCy model
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_nlp_model():
    try:
        return spacy.load("en_core_web_lg")
    except:
        st.warning("Installing spaCy model, please wait...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
        return spacy.load("en_core_web_lg")

# Load the spaCy model
with st.spinner("Loading NLP model..."):
    nlp = load_nlp_model()
    st.success("NLP model loaded!")

# Function to generate a random date string
def generate_random_date():
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2025, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_date = start_date + timedelta(days=random_days)
    return random_date.strftime("%d %B %Y")

# Extract entities using spaCy
def extract_entities(text):
    doc = nlp(text)
    
    legal_entities = {
        "PERSON": [],
        "ORG": [],
        "DATE": [generate_random_date()],
        "MONEY": [],
        "LAW": [],
        "GPE": [],
        "NORP": [],
        "CARDINAL": [],
        "ORDINAL": []
    }
    
    for ent in doc.ents:
        if ent.label_ != "DATE" and ent.label_ in legal_entities:
            legal_entities[ent.label_].append(ent.text)
    
    return doc, legal_entities

# Define a legal document template using Jinja2 (same as in the notebook)
legal_document_template = """

<!DOCTYPE html>

<html>

<head>

<style>

body {

    font-family: 'Times New Roman', Times, serif;

    margin: 2.5cm;

    line-height: 1.5;

}

.header {

    text-align: center;

    margin-bottom: 20px;

}

.title {

    font-size: 18pt;

    font-weight: bold;

    text-align: center;

    margin: 20px 0;

}

.section {

    margin: 15px 0;

}

.section-title {

    font-weight: bold;

    text-decoration: underline;

}

.signature {

    margin-top: 50px;

}

.date {

    margin-top: 20px;

}

.indent {

    margin-left: 20px;

}

.best-holding {

    background-color: #f0f8ff;

    padding: 15px;

    border-left: 5px solid #4682b4;

    margin: 10px 0;

}

</style>

</head>

<body>

<div class="header">

    <h2>LEGAL DOCUMENT</h2>

    <p>Case No. {{ case_number }}</p>

</div>



<div class="section">

    <p class="section-title">DATE:</p>

    <p>{{ date }}</p>

</div>



<div class="section">

    <p class="section-title">PARTIES INVOLVED:</p>

    <ul>

    {% for person in persons %}

        <li>{{ person }}</li>

    {% endfor %}

    </ul>

</div>



<div class="section">

    <p class="section-title">ORGANIZATIONS MENTIONED:</p>

    <ul>

    {% for org in organizations %}

        <li>{{ org }}</li>

    {% endfor %}

    </ul>

</div>



<div class="section">

    <p class="section-title">JURISDICTION:</p>

    <ul>

    {% for place in places %}

        <li>{{ place }}</li>

    {% endfor %}

    </ul>

</div>



<div class="section">

    <p class="section-title">LEGAL REFERENCES:</p>

    <ul>

    {% for law in laws %}

        <li>{{ law }}</li>

    {% endfor %}

    </ul>

</div>



<div class="section">

    <p class="section-title">MONETARY VALUES INVOLVED:</p>

    <ul>

    {% for money in monetary_values %}

        <li>{{ money }}</li>

    {% endfor %}

    </ul>

</div>



<div class="section">

    <p class="section-title">CASE DISCRIPTION:</p>

    <p class="indent">{{ summary }}</p>

</div>



<div class="section">

    <p class="section-title">BEST MATCHING HOLDING (TF-IDF):</p>

    <div class="best-holding">

        <p><strong>{{ tfidf_holding_number }}</strong>: {{ tfidf_holding }} <span style="color: #4682b4;">(Similarity: {{ tfidf_score }})</span></p>

    </div>

</div>



<div class="section">

    <p class="section-title">BEST MATCHING HOLDING (GloVe):</p>

    <div class="best-holding">

        <p><strong>{{ glove_holding_number }}</strong>: {{ glove_holding }} <span style="color: #4682b4;">(Similarity: {{ glove_score }})</span></p>

    </div>

</div>



<div class="date">

    <p>DATED this {{ date }}.</p>

</div>



{% if judge %}

<div class="signature">

    <p>____________________________</p>

    <p>JUDGE {{ judge }}</p>

</div>

{% endif %}



</body>

</html>

"""

# Create Jinja2 template
template = Template(legal_document_template)

# Load example dataset from CaseHOLD (or mock data if not available)
@st.cache(ttl=3600, allow_output_mutation=True)
def load_case_data():
    try:
        from datasets import load_dataset
        dataset = load_dataset("casehold/casehold", trust_remote_code=True)
        df = pd.DataFrame(dataset['train'])
        return df
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        # Create mock data
        mock_data = {
            'example_id': list(range(10)),
            'citing_prompt': [
                "The court found that the defendant violated Section 1983 when..."
            ] * 10,
            'holding_0': ["First holding option..."] * 10,
            'holding_1': ["Second holding option..."] * 10,
            'holding_2': ["Third holding option..."] * 10,
            'holding_3': ["Fourth holding option..."] * 10,
            'holding_4': ["Fifth holding option..."] * 10,
            'label': [random.randint(0, 4) for _ in range(10)]
        }
        return pd.DataFrame(mock_data)

# TF-IDF for finding best holding
def find_best_holding_tfidf(example_id, df):
    row = df[df['example_id'] == example_id]
    
    if row.empty:
        return {
            'best_holding': "No data found for this example", 
            'similarity_score': 0,
            'holding_text': "N/A",
            'holding_number': "N/A"
        }
        
    citing_prompt = row['citing_prompt'].values[0]
    holdings = [row[f'holding_{i}'].values[0] for i in range(5)]
    
    corpus = [citing_prompt] + holdings
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    best_index = similarities.argmax()
    best_score = similarities[best_index]
    best_holding_col = f"holding_{best_index}"
    
    return {
        'best_holding': best_holding_col,
        'similarity_score': round(float(best_score), 4),
        'holding_text': holdings[best_index],
        'holding_number': f"Holding_{best_index}"
    }

# GloVe for finding best holding
def find_best_holding_glove(example_id, df, nlp_model):
    row = df[df['example_id'] == example_id]
    
    if row.empty:
        return {
            'best_holding': "No data found for this example", 
            'similarity_score': 0,
            'holding_text': "N/A",
            'holding_number': "N/A"
        }
        
    citing_prompt = row['citing_prompt'].values[0]
    holdings = [row[f'holding_{i}'].values[0] for i in range(5)]
    
    # Use spaCy's built-in word vectors (which are GloVe-like)
    prompt_doc = nlp_model(citing_prompt)
    holdings_docs = [nlp_model(holding) for holding in holdings]
    
    # Calculate similarities
    similarities = [prompt_doc.similarity(holding_doc) for holding_doc in holdings_docs]
    
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    best_holding_col = f"holding_{best_index}"
    
    return {
        'best_holding': best_holding_col,
        'similarity_score': round(float(best_score), 4),
        'holding_text': holdings[best_index],
        'holding_number': f"Holding_{best_index}"
    }

# Load data
with st.spinner("Loading case data..."):
    df = load_case_data()

# Main app interface
st.subheader("Enter a case ID to analyze")

# Simplified case selection - direct input only
available_ids = sorted(df['example_id'].unique())
max_id = max(available_ids) if available_ids else 0

# Allow user to directly input a case ID
selected_id = st.number_input("Enter Case ID (number)", min_value=0, max_value=max_id, value=None)

# Proceed only if the user has entered a case ID
if selected_id is not None:
    # Get the case text
    row = df[df['example_id'] == selected_id]
    if not row.empty:
        citing_prompt = row['citing_prompt'].values[0]
        
        # Display the citing prompt
        st.subheader("Citing Prompt")
        st.write(citing_prompt)
        
        # Get holdings
        holdings = [row[f'holding_{i}'].values[0] for i in range(5)]
        
        # Display all holdings
        st.subheader("Available Holdings")
        for i, holding in enumerate(holdings):
            st.write(f"**Holding {i}:** {holding[:200]}...")
        
        # Find best holdings using both methods
        with st.spinner("Finding best holdings..."):
            tfidf_result = find_best_holding_tfidf(selected_id, df)
            glove_result = find_best_holding_glove(selected_id, df, nlp)
        
        # Extract NER entities
        with st.spinner("Extracting named entities..."):
            doc, entities = extract_entities(citing_prompt)
        
        # Extract a potential judge
        judge = None
        for token in doc:
            if token.text.lower() == "judge" and token.i < len(doc)-1:
                next_tokens = [doc[token.i+i] for i in range(1, 4) if token.i+i < len(doc)]
                for next_token in next_tokens:
                    if next_token.ent_type_ == "PERSON":
                        judge = next_token.text
                        break
                if judge:
                    break
        
        # Use unique entities to avoid repetition
        unique_persons = list(set(entities["PERSON"]))[:8]
        unique_orgs = list(set(entities["ORG"]))[:8]
        unique_places = list(set(entities["GPE"]))[:8]
        unique_laws = list(set(entities["LAW"]))[:8]
        unique_money = list(set(entities["MONEY"]))[:8]
        
        # Generate the legal document
        template_data = {
            "case_number": f"CASE-{selected_id:05d}",
            "date": entities["DATE"][0] if entities["DATE"] else "N/A",
            "persons": unique_persons if unique_persons else ["No individuals identified"],
            "organizations": unique_orgs if unique_orgs else ["No organizations identified"],
            "places": unique_places if unique_places else ["No locations identified"],
            "laws": unique_laws if unique_laws else ["No legal references identified"],
            "monetary_values": unique_money if unique_money else ["No monetary values identified"],
            "summary": citing_prompt[:500] + "..." if len(citing_prompt) > 500 else citing_prompt,
            "judge": judge,
            "tfidf_holding": tfidf_result['holding_text'],
            "tfidf_score": tfidf_result['similarity_score'],
            "tfidf_holding_number": tfidf_result['holding_number'],
            "glove_holding": glove_result['holding_text'],
            "glove_score": glove_result['similarity_score'],
            "glove_holding_number": glove_result['holding_number']
        }
        
        # Render the document
        rendered_document = template.render(**template_data)
        
        # Display the document
        st.subheader("Generated Legal Document")
        st.components.v1.html(rendered_document, height=600, scrolling=True)
        
        # Download button
        st.download_button(
            label="Download Document as HTML",
            data=rendered_document,
            file_name=f"legal_document_case_{selected_id}.html",
            mime="text/html"
        )
    else:
        st.error("No data found for the selected case ID.")
else:
    st.info("Please enter a Case ID above to generate the legal document.")
