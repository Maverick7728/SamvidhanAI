# SamvidhanAI - Legal Document Generator

SamvidhanAI is an AI-powered legal document generator that leverages natural language processing techniques to extract relevant information from legal text and produce structured legal memoranda.

## Overview

This application uses SpaCy's Named Entity Recognition (NER), text similarity metrics, and template-based document generation to assist legal professionals in creating standardized legal documents without requiring large language models (LLMs).

## Key Features

- **Named Entity Recognition (NER)**: Automatically extracts key entities from legal text including:
  - People (PERSON)
  - Organizations (ORG)
  - Locations (GPE)
  - Legal references (LAW)
  - Monetary values (MONEY)
  - Dates (DATE)

- **Best Matching Holdings**: Uses two similarity algorithms to find the most relevant legal holdings:
  - TF-IDF vectorization with cosine similarity
  - SpaCy's word embeddings for semantic similarity analysis
  - Automatically selects the highest accuracy method for each case

- **Interactive Document Generation**: Provides a clean, user-friendly interface for:
  - Manual case ID selection
  - Entity recognition visualization
  - Single-page formatted legal memorandum display
  - Document export functionality

## Technologies Used

- **SpaCy**: For entity extraction and word similarity
- **Streamlit**: For the interactive web interface
- **Pandas/NumPy**: For data manipulation
- **Scikit-learn**: For TF-IDF vectorization and cosine similarity
- **Jinja2**: For HTML template rendering
- **CaseHOLD Dataset**: For legal case examples and holdings

## Getting Started

1. Install the required dependencies:
   ```
   pip install streamlit spacy pandas numpy scikit-learn jinja2 datasets
   ```

2. Download the SpaCy model:
   ```
   python -m spacy download en_core_web_lg
   ```

3. Run the application:
   ```
   streamlit run legal_doc_generator_app_notebook.py
   ```

4. Enter a case ID from the available range and click "Analyze Case" to generate a legal memorandum

## Example Output

The generated legal memorandum includes:
- Case number and date
- Parties involved and organizations mentioned
- Jurisdiction and legal references
- Monetary values identified
- Case summary
- Best matching legal holding
- Judge signature section

## Future Development

- Support for custom document templates
- Additional entity types specific to legal domains
- Integration with case law databases
- Batch processing capability for multiple documents
