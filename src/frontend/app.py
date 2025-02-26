import base64
import streamlit as st
import datetime
import hashlib
import io
import os
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Configuration and Constants
API_URL = "http://localhost:8000"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# Custom CSS for a more modern, Pinterest/Instagram-like design
def load_modern_css():
    return """
    <style>
    :root {
        --primary-color: #ff5757;
        --background-color: #f5f5f5;
        --card-background: white;
        --text-color: #333;
        --accent-color: #4a4a4a;
    }

    body, .stApp {
        background-color: var(--background-color) !important;
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }

    /* Header Styling */
    .app-header {
        background: linear-gradient(135deg, var(--primary-color), #ff8a5b);
        color: white;
        padding: 1.5rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Card Design */
    .image-card {
        background: var(--card-background);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        overflow: hidden;
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
    }

    .image-card:hover {
        transform: scale(1.03);
    }

    .image-card img {
        width: 100%;
        height: 250px;
        object-fit: cover;
    }

    .image-card-actions {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem;
        background: rgba(255,255,255,0.9);
    }

    /* Filter Buttons */
    .filter-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .filter-btn {
        background-color: var(--background-color);
        border: 1px solid var(--accent-color);
        color: var(--accent-color);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        transition: all 0.3s ease;
    }

    .filter-btn:hover, .filter-btn.active {
        background-color: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
    }

    /* Search Results Grid */
    .results-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1rem;
    }

    /* Sidebar */
    .sidebar .stSidebar {
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """

# Image Processing Functions
def apply_advanced_filter(image, filter_name):
    """Enhanced image filtering with more sophisticated transformations."""
    if image is None:
        return None

    img = image.copy()

    # Advanced filter implementations
    if filter_name == "Original":
        return img
    elif filter_name == "Grayscale":
        return ImageOps.grayscale(img)
    elif filter_name == "Vintage":
        # More nuanced vintage effect
        img = ImageEnhance.Color(img).enhance(0.7)
        img = ImageEnhance.Brightness(img).enhance(1.1)
        img = ImageEnhance.Contrast(img).enhance(1.3)
        return img.filter(ImageFilter.SMOOTH)
    elif filter_name == "Cinematic":
        # Moody, cinematic look
        img = ImageEnhance.Color(img).enhance(0.8)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        return img.filter(ImageFilter.SMOOTH_MORE)
    elif filter_name == "Soft Blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1))
    elif filter_name == "High Contrast":
        return ImageOps.autocontrast(img, cutoff=1)
    return img

# Authentication Utilities
def hash_password(password):
    """Secure password hashing."""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    """Enhanced authentication mechanism."""
    # In a real-world scenario, this would check against a secure database
    valid_users = {
        "demo_user": hash_password("demo_password"),
        "admin": hash_password("admin_password")
    }
    
    hashed_input = hash_password(password)
    if username in valid_users and hashed_input == valid_users[username]:
        st.session_state.user = {
            "username": username,
            "logged_in": True,
            "login_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return True
    return False

# API Interaction Functions
def get_api_stats():
    """Fetch and cache API statistics."""
    try:
        response = requests.get(f"{API_URL}/stats")
        return response.json()
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return None

def search_similar_content(content_bytes, similarity_threshold=0.5):
    """
    Enhanced content search with method selection.
    Supports different embedding techniques for similarity search.
    """
    try:
        if len(content_bytes) > MAX_IMAGE_SIZE:
            st.error("Content too large. Maximum size is 10 MB")
            return None
        
        files = {"file": ("content.jpg", content_bytes, "image/jpeg")}
        
        response = requests.post(
            f"{API_URL}/search/image", 
            files=files, 
            timeout=30
        )
        
        if response.status_code != 200:
            st.error(f"Search Error: {response.status_code}")
            return None
        
        result = response.json()
        
        # Optional: Filter results by similarity threshold
        if result.get('results'):
            result['results'] = [
                r for r in result['results'] 
                if r.get('similarity_score', 0) >= similarity_threshold
            ]
        
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"Search Request Failed: {e}")
        return None

# Path Resolution Utility
def resolve_image_path(relative_path):
    """
    Resolve image path with multiple fallback strategies
    
    Args:
        relative_path (str): Relative path to the image
    
    Returns:
        str: Absolute path to the image, or original path if not found
    """
    # Base project paths to search
    base_paths = [
        os.getcwd(),  # Current working directory
        os.path.dirname(os.path.abspath(__file__)),  # Script directory
        os.path.join(os.getcwd(), 'data', 'raw', 'images'),
        os.path.join(os.getcwd(), 'models'),
    ]
    
    # Try to resolve the path
    for base_path in base_paths:
        full_path = os.path.join(base_path, relative_path)
        if os.path.exists(full_path):
            return full_path
    
    # If no path found, return original
    return relative_path

# Main Streamlit Application
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Visual Discovery", 
        page_icon="🔍", 
        layout="wide"
    )
    
    # Load custom CSS
    st.markdown(load_modern_css(), unsafe_allow_html=True)
    
    # Authentication Flow
    if not st.session_state.get("user", {}).get("logged_in", False):
        st.markdown("""
        <div class="app-header">
            <h1>🔍 Visual Discovery</h1>
            <p>Explore Content Through Visual Similarity</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Login Form
        st.sidebar.title("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            if authenticate(username, password):
                st.rerun()
            else:
                st.error("Invalid credentials")
        
        # Demo Login
        if st.sidebar.button("Demo Login"):
            authenticate("demo_user", "demo_password")
            st.rerun()
        return

    # Main Application Interface
    st.markdown("""
    <div class="app-header">
        <h1>🔍 Visual Discovery</h1>
        <p>Find Similar Content Effortlessly</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with User Info and Search Configuration
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.user['username']}")
        
        # Additional Configuration Options
        st.subheader("Search Configuration")
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5,  # Lowered to 0.5 
            step=0.05
        )
        
        # Logout
        if st.button("Logout"):
            del st.session_state.user
            st.rerun()

    # Main Content Area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Content Upload
        uploaded_file = st.file_uploader(
            "Upload an Image", 
            type=["jpg", "jpeg", "png", "webp"]
        )

        if uploaded_file:
            # Image Processing
            image = Image.open(uploaded_file)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Filter Options
            st.markdown('<div class="filter-container">', unsafe_allow_html=True)
            filters = ["Original", "Grayscale", "Vintage", "Cinematic", "Soft Blur", "High Contrast"]
            selected_filter = st.radio(
                "Apply Filter", 
                filters, 
                horizontal=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Apply Selected Filter
            filtered_image = apply_advanced_filter(image, selected_filter)
            st.image(filtered_image, caption=f"{selected_filter} Filter", use_container_width=True)

            # Search Button
            if st.button("🔍 Find Similar Content"):
                with st.spinner("Searching for similar content..."):
                    # Convert image to bytes
                    img_byte_arr = io.BytesIO()
                    filtered_image.save(img_byte_arr, format="JPEG")
                    img_bytes = img_byte_arr.getvalue()

                    # Perform Search with configurable threshold
                    results = search_similar_content(
                        img_bytes, 
                        similarity_threshold
                    )

                    if results and results.get("status") == "success":
                        st.session_state.search_results = results["results"]
                        st.rerun()
                    else:
                        st.error("No similar content found.")

    with col2:
    # Search Results Display
        st.subheader("Similar Content")
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            results = st.session_state.search_results

            st.markdown('<div class="results-grid">', unsafe_allow_html=True)
            for result in results:
                # Resolve image path
                image_path = resolve_image_path(result.get('path', ''))
                
                # Validate and display image using base64 encoding
                try:
                    with open(image_path, "rb") as img_file:
                        img_bytes = img_file.read()
                        encoded_img = base64.b64encode(img_bytes).decode("utf-8")
                    
                    st.markdown(f'''
                    <div class="image-card">
                        <img src="data:image/jpeg;base64,{encoded_img}" alt="Similar Content">
                        <div class="image-card-actions">
                            <span>Similarity: {result.get('similarity_score', 0):.2%}</span>
                            <div>
                                <button>❤️</button>
                                <button>💾</button>
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not display image: {image_path}. Error: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Upload an image to start exploring similar content!")


if __name__ == "__main__":
    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = {}
    main()