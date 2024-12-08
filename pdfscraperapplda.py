import streamlit as st
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Set up the stopwords and stemmer
en_stopwords = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Function to process the PDF and extract relevant sentences
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to find sentences containing custom keywords
def find_sustainability_sentences_with_stems(text, stems):
    sentences = re.split(r'(?<=[.!?]) +', text)
    relevant_sentences = []
    for sentence in sentences:
        sentence_words = re.findall(r'\b\w+\b', sentence)
        stemmed_sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        if any(stem in stemmed_sentence_words for stem in stems):
            relevant_sentences.append(sentence)
    return relevant_sentences

# Streamlit UI
st.title("Sustainability Keyword Extraction from PDF with Topic Modeling")

# File uploader
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Text input for custom sustainability keywords
custom_keywords_input = st.text_input("Enter your own set of sustainability-related keywords (comma separated):",
                                      "sustainability, renewable, climate, energy, environment, recycling, carbon")

# Convert the user input into a list of keywords
if custom_keywords_input:
    custom_keywords = [keyword.strip().lower() for keyword in custom_keywords_input.split(',')]

    # Stem each custom keyword
    stems = [stemmer.stem(keyword) for keyword in custom_keywords]

    if pdf_file is not None:
        # Step 1: Extract text from the uploaded PDF
        text = extract_text_from_pdf(pdf_file)
        text = re.sub(r'[^a-zA-Z0-9.,]', ' ', text) 
        text = re.sub('[0-9]+', '', text)

        # Step 2: Find sentences related to sustainability
        sustainability_sentences = find_sustainability_sentences_with_stems(text, stems)

        # Display only the first 5 sustainability sentences
        if sustainability_sentences:
            st.subheader("Sustainability-related Sentences:")
            for sentence in sustainability_sentences[:5]:  # Only display the first 5 sentences
                st.write(sentence)
        else:
            st.write("No sustainability-related sentences found.")

        # Step 3: Tokenize and filter words
        all_sustainability_words = []
        for sentence in sustainability_sentences:
            sustain_words = word_tokenize(sentence)
            sustain_words_filter = [w for w in sustain_words if w.lower() not in en_stopwords and w not in string.punctuation]
            sustain_words_filter = [w for w in sustain_words_filter if len(w) > 2]
            all_sustainability_words.extend(sustain_words_filter)

        # Step 4: Frequency Distribution of words
        sustain_freq = FreqDist(all_sustainability_words)

        # Display the 50 most common words as a table
        st.subheader("Top 50 Most Common Words and Their Frequencies")
        most_common_words = sustain_freq.most_common(50)  # Get the 50 most common words
        freq_table = [(word, freq) for word, freq in most_common_words]
        st.write(freq_table)

        # Step 5: Generate and display word cloud
        wordcloud = WordCloud(width=1000, height=500, stopwords=en_stopwords,
                              colormap="plasma", collocations=False, max_words=700).generate(' '.join(all_sustainability_words))

        st.subheader("Word Cloud of Sustainability-related Words")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot()

        # Step 6: Topic Modeling with LDA
        # Convert the sentences to a single text corpus for topic modeling
        corpus = [" ".join(word_tokenize(sentence)) for sentence in sustainability_sentences]

        # Vectorize the text using TF-IDF (passing stopwords as a list)
        vectorizer = TfidfVectorizer(stop_words=list(en_stopwords), max_df=0.95, min_df=2)
        X = vectorizer.fit_transform(corpus)

        # Apply LDA (Latent Dirichlet Allocation)
        lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
        lda_model.fit(X)

        # Display the topics with top 10 words
        st.subheader("Topic Modeling (LDA) Results")

        for topic_idx, topic in enumerate(lda_model.components_):
            st.write(f"Topic #{topic_idx + 1}:")
            top_words_idx = topic.argsort()[-10:][::-1]  # Get top 10 words for each topic
            top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
            st.write(", ".join(top_words))
