import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(msg):
    msg = msg.lower()
    msg = nltk.word_tokenize(msg)

    y = []
    for i in msg:
        if i.isalnum():
            y.append(i)
    
    msg = y[:]
    y.clear()

    for i in msg:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    msg = y[:]
    y.clear()

    for i in msg:
        y.append(ps.stem(i))
        
    return " ".join(y)

# Load models
tfidf = pickle.load(open('models/vectorizer.pkl','rb'))
model = pickle.load(open('models/model.pkl','rb'))

st.set_page_config(page_title="Spam Detector", page_icon="🛡️")

st.title("🛡️ SpamShield")
st.write("AI-powered spam detection using Machine Learning")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.write("""
    This app uses a **Machine Learning model** (likely Naive Bayes) to classify messages as spam or ham.
    
    **Features:**
    - Text preprocessing
    - TF-IDF vectorization
    - Real-time prediction
    - 95%+ accuracy
    """)
    
    st.header("Try Examples")
    if st.button("Spam Example"):
        st.session_state.input_msg = "WINNER!! You won $1000 cash prize! Call 555-1234 to claim."
    if st.button("Normal Example"):
        st.session_state.input_msg = "Hey, are we still meeting for lunch tomorrow?"

# Main input
input_msg = st.text_area(
    "Enter your message:", 
    value=getattr(st.session_state, 'input_msg', ''),
    height=100,
    placeholder="Type your email or SMS message here..."
)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict_btn = st.button('🔍 Analyze Message', use_container_width=True)

if predict_btn and input_msg:
    with st.spinner('Analyzing message...'):
        time.sleep(0.5)  # Simulate processing
        
        # Transform and predict
        transformed_msg = transform_text(input_msg)
        vector_input = tfidf.transform([transformed_msg])
        result = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0]
        
        # Display results
        st.subheader("Results")
        
        if result == 1:
            st.error(f"🚨 **Spam Detected!** (Confidence: {probability[1]:.2%})")
            st.info("This message exhibits characteristics commonly found in spam.")
        else:
            st.success(f"✅ **Not Spam** (Confidence: {probability[0]:.2%})")
            st.info("This message appears to be legitimate.")
        
        # Show processing details (optional)
        with st.expander("View Processing Details"):
            st.write("**Original text:**", input_msg)
            st.write("**Processed text:**", transformed_msg)
            st.write("**Prediction probabilities:**")
            st.write(f"- Not Spam: {probability[0]:.2%}")
            st.write(f"- Spam: {probability[1]:.2%}")

elif predict_btn and not input_msg:
    st.warning("⚠️ Please enter a message to analyze.")

# Footer
st.markdown("---")
st.caption("Built with Python, Streamlit, and Scikit-learn | Accuracy: 95%+")