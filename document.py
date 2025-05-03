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
# --- Added potentially missing imports for standalone run ---
from datasets import load_dataset
import os
import subprocess # For installing model if needed
# --- End added imports ---

# Page configuration
st.set_page_config(page_title="Legal Document Generator", layout="wide")

# App title
st.title("Legal Document Generator with NER and Best Matching Holdings")

# Try to load the spaCy model (using Streamlit caching)
# Use @st.cache_resource for models
@st.cache_resource(show_spinner=False)
def load_nlp_model(model_name="en_core_web_lg"):
    try:
        nlp_model = spacy.load(model_name)
        return nlp_model
    except OSError:
        st.warning(f"SpaCy model '{model_name}' not found. Attempting to download...")
        try:
            # Use subprocess to run the download command
            st.info(f"Running: python -m spacy download {model_name}")
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True, capture_output=True)
            st.success(f"Model '{model_name}' downloaded.")
            # Need to reload after download
            nlp_model = spacy.load(model_name)
            st.success("NLP model loaded!")
            return nlp_model
        except subprocess.CalledProcessError as cpe:
             st.error(f"Failed to download model '{model_name}'. Error: {cpe.stderr.decode() if cpe.stderr else cpe}")
             st.error("Please try installing the model manually: python -m spacy download en_core_web_lg")
             return None # Indicate failure
        except Exception as e:
             st.error(f"Failed to download or load model '{model_name}': {e}")
             # Try loading small as fallback
             try:
                  st.warning("Trying to load 'en_core_web_sm'...")
                  nlp_model = spacy.load('en_core_web_sm')
                  st.success("Fallback model 'en_core_web_sm' loaded.")
                  return nlp_model
             except OSError:
                  st.error("Could not load 'en_core_web_sm' either. NER features disabled.")
                  return None

# Load the spaCy model instance
nlp = load_nlp_model()

# Function to generate a random date string (as defined in original notebook code)
def generate_random_date():
    start_date = datetime(2000, 1, 1); end_date = datetime(2025, 12, 31)
    delta = end_date - start_date; random_days = random.randint(0, delta.days)
    random_date = start_date + timedelta(days=random_days)
    return random_date.strftime("%d %B %Y")

# Extract entities using spaCy (using the loaded global 'nlp' model)
def extract_entities(text):
    # Original entity extraction logic used in Streamlit section
    if not nlp:
         return None, {"PERSON": [], "ORG": [], "DATE": [generate_random_date()], "MONEY": [], "LAW": [], "GPE": [], "NORP": [], "CARDINAL": [], "ORDINAL": []}

    doc = nlp(str(text)) # Process the text
    legal_entities = { # Initialize with expected keys and placeholder date
        "PERSON": [], "ORG": [], "DATE": [generate_random_date()], "MONEY": [],
        "LAW": [], "GPE": [], "NORP": [], "CARDINAL": [], "ORDINAL": []
    }
    for ent in doc.ents:
        # Only add if label is in our predefined keys, and *not* DATE
        if ent.label_ != "DATE" and ent.label_ in legal_entities:
            legal_entities[ent.label_].append(ent.text)
    return doc, legal_entities

# Define a legal document template using Jinja2 (same as in the notebook)
# Use raw string literal (r"""...""") to avoid escaping issues
legal_document_template = r"""
<!DOCTYPE html>
<html>
<head>
<style>
body { font-family: 'Times New Roman', Times, serif; margin: 2.5cm; line-height: 1.5; } .header { text-align: center; margin-bottom: 20px; } .title { font-size: 18pt; font-weight: bold; text-align: center; margin: 20px 0; } .section { margin: 15px 0; } .section-title { font-weight: bold; text-decoration: underline; } .signature { margin-top: 50px; } .date { margin-top: 20px; } .indent { margin-left: 20px; } .best-holding { background-color: #f0f8ff; padding: 15px; border-left: 5px solid #4682b4; margin: 10px 0; }
</style>
</head>
<body>
<div class="header"> <h2>LEGAL MEMORANDUM</h2> <p>Case No. {{ case_number }}</p> </div>
<div class="section"> <p class="section-title">DATE:</p> <p>{{ date }}</p> </div>
<div class="section"> <p class="section-title">PARTIES INVOLVED:</p> <ul> {% for person in persons %} <li>{{ person }}</li> {% else %} <li>No individuals identified</li> {% endfor %} </ul> </div>
<div class="section"> <p class="section-title">ORGANIZATIONS MENTIONED:</p> <ul> {% for org in organizations %} <li>{{ org }}</li> {% else %} <li>No organizations identified</li> {% endfor %} </ul> </div>
<div class="section"> <p class="section-title">JURISDICTION:</p> <ul> {% for place in places %} <li>{{ place }}</li> {% else %} <li>No locations identified</li> {% endfor %} </ul> </div>
<div class="section"> <p class="section-title">LEGAL REFERENCES:</p> <ul> {% for law in laws %} <li>{{ law }}</li> {% else %} <li>No legal references identified</li> {% endfor %} </ul> </div>
<div class="section"> <p class="section-title">MONETARY VALUES INVOLVED:</p> <ul> {% for money in monetary_values %} <li>{{ money }}</li> {% else %} <li>No monetary values identified</li> {% endfor %} </ul> </div>
<div class="section"> <p class="section-title">CASE SUMMARY:</p> <p class="indent">{{ summary }}</p> </div>
<div class="section"> <p class="section-title">BEST MATCHING HOLDING:</p> <div class="best-holding"> <p><strong>Holding {{ best_holding_index }}:</strong> {{ best_holding }}</p> </div> </div>
<div class="date"> <p>DATED this {{ date }}.</p> </div>
{% if judge %} <div class="signature"> <p>____________________________</p> <p>JUDGE {{ judge }}</p> </div> {% endif %}
</body>
</html>
"""
# Create Jinja2 template object
try:
    template = Template(legal_document_template)
except Exception as template_err:
    st.error(f"Failed to create Jinja2 template: {template_err}")
    template = None # Ensure it's None if failed

# Load example dataset from CaseHOLD (using Streamlit caching)
# Use @st.cache_data for data loading
@st.cache_data(show_spinner="Loading CaseHOLD dataset...")
def load_case_data():
    try:
        # Using casehold/casehold as in original script's GloVe section
        dataset_st = load_dataset("casehold/casehold", trust_remote_code=True)
        # Use train split for the selection dropdown
        df_st = pd.DataFrame(dataset_st['train'])
        # Ensure columns are lowercase
        df_st.columns = df_st.columns.str.strip().str.lower()
        return df_st
    except Exception as e:
        st.error(f"Could not load dataset 'casehold/casehold': {e}. Using mock data.")
        # Create mock data if loading fails
        mock_data = { # Using sample data from original script block
            'example_id': list(range(10)),
            'citing_prompt': ["The court found that the defendant violated Section 1983 when..."] * 10,
            'holding_0': ["First holding option..."] * 10, 'holding_1': ["Second holding option..."] * 10,
            'holding_2': ["Third holding option..."] * 10, 'holding_3': ["Fourth holding option..."] * 10,
            'holding_4': ["Fifth holding option..."] * 10, 'label': [random.randint(0, 4) for _ in range(10)]
        }
        # Ensure mock data columns are lowercase too
        df_mock = pd.DataFrame(mock_data)
        df_mock.columns = df_mock.columns.str.strip().str.lower()
        return df_mock

# Define TF-IDF function specific to the structure needed by Streamlit app
def find_best_holding_tfidf_st(example_id_st, df_st_data):
    if df_st_data is None: return {'error': True, 'holding_text': 'Error: DataFrame not available'}
    row_st = df_st_data[df_st_data['example_id'] == example_id_st]
    if row_st.empty:
        return {'best_holding_index': -1, 'similarity_score': 0, 'holding_text': 'Error: ID not found', 'error': True}
    try:
        prompt = str(row_st['citing_prompt'].iloc[0])
        holdings = [str(row_st[f'holding_{i}'].iloc[0]) for i in range(5)]
        corpus = [prompt] + holdings
        vectorizer = TfidfVectorizer(stop_words='english') # Consistent with other uses
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        return {
            'best_holding_index': best_idx,       # 0-4 index
            'similarity_score': round(best_score, 4),
            'holding_text': holdings[best_idx]
        }
    except Exception as e:
         return {'best_holding_index': -1, 'similarity_score': 0, 'holding_text': f'TF-IDF Error: {e}', 'error': True}

# Define GloVe function for Streamlit app
# This uses spaCy's .similarity method (which relies on the loaded model's vectors)
def find_best_holding_glove_st(example_id_st, df_st_data, nlp_model):
    # Using spaCy's built-in similarity (depends on loaded model)
    if not nlp_model or not hasattr(nlp_model, 'vocab'):
         return {'best_holding_index': -1, 'similarity_score': 0, 'holding_text': 'Error: NLP model unavailable for spaCy Sim', 'error': True}
    if df_st_data is None: return {'error': True, 'holding_text': 'Error: DataFrame not available'}

    row_st = df_st_data[df_st_data['example_id'] == example_id_st]
    if row_st.empty:
        return {'best_holding_index': -1, 'similarity_score': 0, 'holding_text': 'Error: ID not found', 'error': True}
    try:
        prompt = str(row_st['citing_prompt'].iloc[0])
        holdings = [str(row_st[f'holding_{i}'].iloc[0]) for i in range(5)]

        # Process prompt and holdings with spaCy
        prompt_doc = nlp_model(prompt)
        # Check if prompt has a usable vector
        prompt_has_usable_vector = prompt_doc.has_vector and np.any(prompt_doc.vector != 0)

        if not prompt_has_usable_vector:
            st.warning(f"Prompt for ID {example_id_st} has no usable vector. spaCy similarities will be 0.")
            similarities = [0.0] * len(holdings)
        else:
            holdings_docs = [nlp_model(h) for h in holdings]
            similarities = []
            for h_doc in holdings_docs:
                 holding_has_usable_vector = h_doc.has_vector and np.any(h_doc.vector != 0)
                 if holding_has_usable_vector:
                      sim = prompt_doc.similarity(h_doc)
                      if np.isnan(sim):
                          sim = 0.0
                 else:
                      sim = 0.0
                 similarities.append(sim)

        if not similarities:
             return {'best_holding_index': -1, 'similarity_score': 0, 'holding_text': 'Error: No similarities calculated', 'error': True}

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        return {
            'best_holding_index': best_index,
            'similarity_score': round(best_score, 4),
            'holding_text': holdings[best_index]
        }
    except Exception as e:
         st.error(f"Error in find_best_holding_glove_st for ID {example_id_st}: {e}")
         return {'best_holding_index': -1, 'similarity_score': 0, 'holding_text': f'spaCy Sim Error: {e}', 'error': True}

# --- Main App Interface ---

# Load the case data
df = load_case_data()

st.subheader("Select a case to analyze")

# Case ID selection logic (handle if df is empty or None)
selected_id = None
if df is not None and not df.empty and 'example_id' in df.columns:
    available_ids = sorted(df['example_id'].unique())
    # Changed from selectbox to text_input with validation
    input_id = st.text_input("Enter a Case Example ID:", placeholder="Type a case ID here")
    
    # Validate input
    if input_id:
        try:
            input_id_int = int(input_id)
            if input_id_int in available_ids:
                selected_id = input_id_int
            else:
                st.error(f"Case ID {input_id_int} not found in the dataset. Available IDs range from {min(available_ids)} to {max(available_ids)}.")
        except ValueError:
            st.error("Please enter a valid numeric Case ID.")
else:
    st.error("Failed to load or process case data. Cannot display analysis.")

if selected_id is not None and df is not None:
    row = df[df['example_id'] == selected_id]
    if not row.empty:
        citing_prompt = row['citing_prompt'].iloc[0]
        holdings_display = [row[f'holding_{i}'].iloc[0] for i in range(5)]

        st.markdown("---")
        st.subheader("Citing Prompt")
        st.write(citing_prompt)

        st.subheader("Available Holdings")
        for i, holding in enumerate(holdings_display):
            st.write(f"**Holding {i}:** {holding[:200]}...")

        st.markdown("---")
        tfidf_result = None
        glove_result = None
        doc = None
        entities_dict = {}
        judge = None

        # Calculate results in the background without showing comparison
        with st.spinner("Analyzing holdings..."):
            tfidf_result = find_best_holding_tfidf_st(selected_id, df)
            glove_result = find_best_holding_glove_st(selected_id, df, nlp)
            
            # Determine the best overall holding based on accuracy (TF-IDF is preferred)
            best_result = None
            if tfidf_result and 'error' not in tfidf_result:
                best_result = {
                    'best_holding_index': tfidf_result['best_holding_index'],
                    'best_score': tfidf_result['similarity_score'],
                    'best_holding': tfidf_result['holding_text'],
                    'method': 'TF-IDF'
                }
            elif glove_result and 'error' not in glove_result:
                best_result = {
                    'best_holding_index': glove_result['best_holding_index'],
                    'best_score': glove_result['similarity_score'],
                    'best_holding': glove_result['holding_text'],
                    'method': 'GloVe/spaCy'
                }
            else:
                st.error("Both analysis methods failed to find a best matching holding.")

        st.markdown("---")
        st.subheader("Named Entity Recognition")
        if nlp:
            with st.spinner("Extracting named entities..."):
                doc, entities_dict = extract_entities(citing_prompt)

            if doc:
                for token in doc:
                    if token.text.lower() == "judge" and token.i < len(doc)-1:
                        next_tokens = [doc[token.i+i] for i in range(1, 4) if token.i+i < len(doc)]
                        for nt in next_tokens:
                            if nt.ent_type_ == "PERSON": judge = nt.text; break
                        if judge: break

                st.write("Entities extracted from the citing prompt:")
                ner_col1, ner_col2, ner_col3 = st.columns(3)
                ner_cols = [ner_col1, ner_col2, ner_col3]
                entity_types_order = ["PERSON", "ORG", "GPE", "LAW", "MONEY", "DATE", "NORP", "CARDINAL", "ORDINAL"]
                col_idx = 0
                found_ner = False
                for etype in entity_types_order:
                    if etype in entities_dict and entities_dict[etype]:
                        found_ner = True
                        unique_ents = list(set(entities_dict[etype]))[:5]
                        with ner_cols[col_idx % 3]:
                            st.markdown(f"**{etype}:**")
                            for ent_text in unique_ents: st.write(f"- {ent_text}")
                            if len(set(entities_dict[etype])) > 5: st.caption("...")
                        col_idx += 1
                if judge:
                    found_ner = True
                    with ner_cols[col_idx % 3]:
                        st.info(f"**Judge (Heuristic):** {judge}")
                if not found_ner:
                    st.info("No relevant entities found by NER in the citing prompt.")
            else:
                st.warning("Could not process text for NER (spaCy doc object is None).")
        else:
            st.warning("NLP model not loaded, skipping NER.")

        st.markdown("---")
        st.subheader("Generated Legal Document")

        if template is not None and best_result is not None:
            unique_persons_st = list(set(entities_dict.get("PERSON",[])))[:8]
            unique_orgs_st = list(set(entities_dict.get("ORG", [])))[:8]
            unique_places_st = list(set(entities_dict.get("GPE", [])))[:8]
            unique_laws_st = list(set(entities_dict.get("LAW", [])))[:8]
            unique_money_st = list(set(entities_dict.get("MONEY", [])))[:8]
            date_st = entities_dict.get("DATE", ["N/A"])[0]

            template_data_st = {
                "case_number": f"CASE-{selected_id:05d}",
                "date": date_st,
                "persons": unique_persons_st if unique_persons_st else ["No individuals identified"],
                "organizations": unique_orgs_st if unique_orgs_st else ["No organizations identified"],
                "places": unique_places_st if unique_places_st else ["No locations identified"],
                "laws": unique_laws_st if unique_laws_st else ["No legal references identified"],
                "monetary_values": unique_money_st if unique_money_st else ["No monetary values identified"],
                "summary": citing_prompt[:500] + "..." if len(citing_prompt) > 500 else citing_prompt,
                "judge": judge,
                "best_holding_index": best_result['best_holding_index'],
                "best_holding": best_result['best_holding'],
                "best_score": best_result['best_score']
            }

            try:
                 rendered_document_st = template.render(**template_data_st)
                 
                 # Revert to components.v1.html but with a much larger height
                 st.components.v1.html(
                     rendered_document_st, 
                     height=1200,  # Increased height to show more content
                     scrolling=False  # Try to disable scrolling
                 )

                 st.download_button(
                     label="Download Document as HTML",
                     data=rendered_document_st,
                     file_name=f"legal_document_case_{selected_id}.html",
                     mime="text/html"
                 )
            except Exception as render_err:
                 st.error(f"Error rendering document template: {render_err}")
        else:
             st.warning("Cannot generate document. Template or analysis results might be missing.")

    else:
        st.error(f"No data found for the selected Case ID: {selected_id}")
