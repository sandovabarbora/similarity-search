import streamlit as st
import requests
from PIL import Image
import io
import os
from pathlib import Path
import time

# API Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Similarity Search",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stImage > img {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .tweet-card {
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        background-color: white;
    }
    .tweet-stats {
        color: #666;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def get_api_stats():
    """Get statistics about the loaded datasets"""
    try:
        response = requests.get(f"{API_URL}/stats")
        return response.json()
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def search_similar_images(image_bytes):
    """Send image to API and get similar images"""
    try:
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(f"{API_URL}/search/image", files=files)
        return response.json()
    except Exception as e:
        st.error(f"Error during image search: {str(e)}")
        return None

def search_similar_texts(text):
    """Send text to API and get similar texts"""
    try:
        response = requests.post(
            f"{API_URL}/search/text",
            data={'text': text}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error during text search: {str(e)}")
        return None

def display_tweet_card(tweet_data):
    """Display a tweet card with styling"""
    st.markdown(
        f"""
        <div class="tweet-card">
            <div>{tweet_data['text']}</div>
            <div class="tweet-stats">
                🔄 {tweet_data['retweets']} Retweets | 
                ❤️ {tweet_data['likes']} Likes | 
                📅 {tweet_data['timestamp']} |
                Similarity: {tweet_data['similarity_score']:.2%}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    # Title and description
    st.title("🔍 Similarity Search")
    
    # Get API stats
    stats = get_api_stats()
    
    # Create tabs for different search types
    tab1, tab2 = st.tabs(["Image Search", "Text Search"])
    
    # Image Search Tab
    with tab1:
        if stats:
            st.info(f"Image database contains {stats['images']['total']} images")
        
        uploaded_file = st.file_uploader(
            "Upload an image to find similar ones",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                if st.button("🔍 Find Similar Images", type="primary", key="image_search"):
                    with st.spinner("Searching for similar images..."):
                        results = search_similar_images(uploaded_file.getvalue())
                        
                        if results and results.get('status') == 'success':
                            with col2:
                                st.subheader("Similar Images")
                                
                                similar_images = results['results']
                                cols = st.columns(3)
                                for idx, img_data in enumerate(similar_images):
                                    with cols[idx % 3]:
                                        try:
                                            img_path = img_data['path']
                                            if os.path.exists(img_path):
                                                img = Image.open(img_path)
                                                st.image(
                                                    img,
                                                    caption=f"Similarity: {img_data['similarity_score']:.2%}",
                                                    use_container_width=True
                                                )
                                        except Exception as e:
                                            st.error(f"Error loading image: {img_path}")
                        else:
                            st.error("No results found or error in search")
    
    # Text Search Tab
    with tab2:
        if stats:
            st.info(f"Text database contains {stats['texts']['total']} texts")
        
        search_text = st.text_area(
            "Enter text to find similar content",
            height=100,
            placeholder="Enter your text here..."
        )
        
        if st.button("🔍 Find Similar Texts", type="primary", key="text_search"):
            if not search_text.strip():
                st.warning("Please enter some text to search")
            else:
                with st.spinner("Searching for similar texts..."):
                    results = search_similar_texts(search_text)
                    
                    if results and results.get('status') == 'success':
                        st.subheader("Similar Texts")
                        
                        for text_data in results['results']:
                            display_tweet_card(text_data)
                    else:
                        st.error("No results found or error in search")
    
    # Sidebar information
    with st.sidebar:
        st.subheader("About")
        st.write("""
        This application uses deep learning to find similar content in our database.
        You can search for similar images or texts using our advanced similarity algorithms.
        """)
        
        if stats:
            st.subheader("System Stats")
            st.write("Image Database:")
            st.write(f"- Total images: {stats['images']['total']}")
            st.write(f"- Feature dimension: {stats['images']['feature_dimension']}")
            
            st.write("Text Database:")
            st.write(f"- Total texts: {stats['texts']['total']}")
            st.write(f"- Feature dimension: {stats['texts']['feature_dimension']}")

if __name__ == "__main__":
    main()