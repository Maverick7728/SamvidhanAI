import legal as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from wordcloud import WordCloud
from collections import Counter
import numpy as np
import jinja2
import base64
from io import BytesIO

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

# Page configuration
st.set_page_config(page_title="Legal Text Analyzer", layout="wide")

# Sidebar
st.sidebar.title("Legal Text Analysis")
page = st.sidebar.selectbox(
    "Choose a feature",
    ["Text Analysis", "Named Entity Recognition", "Contract Generator"]
)

# Contract template
contract_template = """
# LEGAL SERVICES AGREEMENT

THIS LEGAL SERVICES AGREEMENT (the "Agreement") is made and entered into as of {{ date }}, by and between {{ client_name }} ("Client") and {{ law_firm }} ("Law Firm").

## 1. SCOPE OF SERVICES

Law Firm agrees to provide the following legal services to Client (the "Services"):
{% for service in services %}
- {{ service }}
{% endfor %}

## 2. FEES AND PAYMENT

Client agrees to pay Law Firm for the Services at the rate of ${{ hourly_rate }} per hour. Law Firm will bill Client on a monthly basis, and payment is due within {{ payment_term }} days of receipt of each invoice.

## 3. TERM AND TERMINATION

This Agreement shall commence on {{ start_date }} and shall continue until {{ end_date }} or until terminated by either party with {{ termination_notice }} days written notice.

## 4. CONFIDENTIALITY

Law Firm acknowledges that during the engagement, Law Firm will have access to confidential information of Client, which shall be kept strictly confidential.

## 5. GOVERNING LAW

This Agreement shall be governed by and construed in accordance with the laws of the State of {{ governing_state }}.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first above written.

CLIENT:                             LAW FIRM:
{{ client_name }}                   {{ law_firm }}

By: ________________________        By: ________________________
Name: {{ client_rep }}              Name: {{ firm_rep }}
Title: {{ client_title }}           Title: {{ firm_title }}
"""

# Define functions
def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    processed_sentences = []
    all_tokens = []
    
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        
        processed_sentences.append(' '.join(filtered_tokens))
        all_tokens.extend(filtered_tokens)
    
    return {
        'original_text': text,
        'sentences': sentences,
        'processed_sentences': processed_sentences,
        'all_tokens': all_tokens,
        'num_sentences': len(sentences),
        'num_tokens': len(all_tokens)
    }

def analyze_word_frequency(tokens, top_n=30):
    word_freq = Counter(tokens)
    most_common = word_freq.most_common(top_n)
    df = pd.DataFrame(most_common, columns=['Word', 'Frequency'])
    return {'word_freq': word_freq, 'dataframe': df}

def generate_wordcloud(word_freq):
    wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate_from_frequencies(word_freq)
    return wc

def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    entity_counts = Counter([ent[1] for ent in entities])
    return {'entities': entities, 'entity_counts': entity_counts}

def get_wordcloud_img(wordcloud):
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    return img

def generate_contract(template_text, context):
    template = jinja2.Template(template_text)
    rendered_contract = template.render(**context)
    return rendered_contract

def create_download_link(content, filename, text):
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Main area - Text Analysis Page
if page == "Text Analysis":
    st.title("Legal Text Analysis")
    
    # Text input
    text_input = st.text_area("Enter legal text for analysis", height=200)
    
    if text_input:
        # Preprocess text
        preprocessing_results = preprocess_text(text_input)
        
        # Display basic stats
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of sentences: {preprocessing_results['num_sentences']}")
        with col2:
            st.write(f"Number of tokens (after preprocessing): {preprocessing_results['num_tokens']}")
        
        # Word frequency analysis
        freq_analysis = analyze_word_frequency(preprocessing_results['all_tokens'])
        
        # Word frequency table
        st.subheader("Top Words")
        st.dataframe(freq_analysis['dataframe'].head(20), use_container_width=True)
        
        # Word frequency chart
        st.subheader("Word Frequency Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Frequency', y='Word', data=freq_analysis['dataframe'].head(15), palette='viridis', ax=ax)
        st.pyplot(fig)
        
        # Word cloud
        st.subheader("Word Cloud")
        wordcloud = generate_wordcloud(freq_analysis['word_freq'])
        st.image(wordcloud.to_array(), use_column_width=True)

# Named Entity Recognition Page
elif page == "Named Entity Recognition":
    st.title("Named Entity Recognition")
    
    # Text input
    text_input = st.text_area("Enter legal text for entity extraction", height=200)
    
    if text_input:
        # Process with spaCy for NER
        with st.spinner("Performing Named Entity Recognition..."):
            ner_results = perform_ner(text_input)
        
        # Display entities table
        st.subheader("Entities Found")
        entities_df = pd.DataFrame(ner_results['entities'], columns=['Entity', 'Type'])
        st.dataframe(entities_df, use_container_width=True)
        
        # Entity type distribution
        st.subheader("Entity Type Distribution")
        entity_counts_df = pd.DataFrame(list(ner_results['entity_counts'].items()), 
                                        columns=['Entity Type', 'Count'])
        entity_counts_df = entity_counts_df.sort_values('Count', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Count', y='Entity Type', data=entity_counts_df, palette='viridis', ax=ax)
        st.pyplot(fig)
        
        # Entity extraction by type
        st.subheader("Extracted Entities by Type")
        selected_type = st.selectbox("Select entity type", options=sorted(entity_counts_df['Entity Type'].tolist()))
        
        if selected_type:
            filtered_entities = [entity for entity, entity_type in ner_results['entities'] if entity_type == selected_type]
            unique_entities = sorted(set(filtered_entities))
            
            if unique_entities:
                st.write(f"{len(unique_entities)} unique {selected_type} entities found:")
                st.write(", ".join(unique_entities))
            else:
                st.write(f"No entities of type {selected_type} found.")

# Contract Generator Page
elif page == "Contract Generator":
    st.title("Legal Contract Generator")
    
    # Contract form
    st.subheader("Contract Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        date = st.date_input("Agreement Date")
        client_name = st.text_input("Client Name", "ABC Corporation")
        law_firm = st.text_input("Law Firm", "Legal Eagles LLP")
        hourly_rate = st.number_input("Hourly Rate ($)", min_value=0, value=350)
        payment_term = st.number_input("Payment Term (days)", min_value=1, value=30)
    
    with col2:
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        termination_notice = st.number_input("Termination Notice (days)", min_value=1, value=15)
        governing_state = st.text_input("Governing State", "California")
    
    st.subheader("Representatives")
    col1, col2 = st.columns(2)
    
    with col1:
        client_rep = st.text_input("Client Representative", "John Smith")
        client_title = st.text_input("Client Title", "CEO")
    
    with col2:
        firm_rep = st.text_input("Firm Representative", "Jane Doe")
        firm_title = st.text_input("Firm Title", "Managing Partner")
    
    st.subheader("Services")
    services = []
    for i in range(4):
        service = st.text_input(f"Service {i+1}", value="" if i > 0 else "Legal representation")
        if service:
            services.append(service)
    
    # Generate contract
    if st.button("Generate Contract"):
        contract_context = {
            'date': date.strftime("%Y-%m-%d"),
            'client_name': client_name,
            'law_firm': law_firm,
            'services': services,
            'hourly_rate': str(hourly_rate),
            'payment_term': str(payment_term),
            'start_date': start_date.strftime("%Y-%m-%d"),
            'end_date': end_date.strftime("%Y-%m-%d"),
            'termination_notice': str(termination_notice),
            'governing_state': governing_state,
            'client_rep': client_rep,
            'firm_rep': firm_rep,
            'client_title': client_title,
            'firm_title': firm_title
        }
        
        generated_contract = generate_contract(contract_template, contract_context)
        
        st.subheader("Generated Contract")
        st.markdown(generated_contract)
        
        # Download link
        st.markdown(
            create_download_link(
                generated_contract, 
                f"legal_contract_{date.strftime('%Y-%m-%d')}.md", 
                "Download Contract as Markdown"
            ),
            unsafe_allow_html=True
        )


# # Display the Streamlit code with a message on how to run it
# print("To use the Streamlit interface, save the following code to a file named 'legal_text_analyzer.py' and run with: streamlit run legal_text_analyzer.py")
# print("\n" + "-" * 80 + "\n")
# print(streamlit_code)

# # Create a function to save the Streamlit code to a file
# def save_streamlit_code(code, filename="legal_text_analyzer.py"):
#     with open(filename, 'w') as f:
#         f.write(code)
#     print(f"\nStreamlit code saved to {filename}")
#     print(f"To run the app, use the command: streamlit run {filename}")

# # Uncomment to save the Streamlit code
# # save_streamlit_code(streamlit_code)