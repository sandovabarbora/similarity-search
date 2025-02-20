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
    page_title="Image Similarity Search",
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
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def get_api_stats():
    """Get statistics about the loaded dataset"""
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
        response = requests.post(f"{API_URL}/search", files=files)
        return response.json()
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return None

def main():
    # Title and description
    st.title("🔍 Image Similarity Search")
    
    # Get API stats
    stats = get_api_stats()
    if stats:
        st.info(f"Database contains {stats['total_images']} images")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image to find similar ones",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        # Create columns for layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Uploaded Image")
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            # Search button
            if st.button("🔍 Find Similar Images", type="primary"):
                with st.spinner("Searching for similar images..."):
                    # Get search results
                    results = search_similar_images(uploaded_file.getvalue())
                    
                    if results and results.get('status') == 'success':
                        with col2:
                            st.subheader("Similar Images")
                            
                            # Create grid for similar images
                            similar_images = results['results']
                            cols = st.columns(3)
                            for idx, img_data in enumerate(similar_images):
                                with cols[idx % 3]:
                                    try:
                                        # Load and display image
                                        img_path = img_data['path']
                                        if os.path.exists(img_path):
                                            img = Image.open(img_path)
                                            st.image(
                                                img,
                                                caption=f"Similarity: {img_data['similarity_score']:.2%}",
                                                use_column_width=True
                                            )
                                    except Exception as e:
                                        st.error(f"Error loading image: {img_path}")
                    else:
                        st.error("No results found or error in search")

    # Add information about the system
    with st.sidebar:
        st.subheader("About")
        st.write("""
        This application uses deep learning to find similar images in the Flickr30k dataset.
        Upload any image to find visually similar ones in our database.
        """)
        
        if stats:
            st.subheader("System Stats")
            st.write(f"- Total images: {stats['total_images']}")
            st.write(f"- Feature dimension: {stats['feature_dimension']}")

if __name__ == "__main__":
    main()