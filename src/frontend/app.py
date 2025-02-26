import datetime
import hashlib
import io
import os
import random

import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# API Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="STRV Vision", page_icon="👁️", layout="wide", initial_sidebar_state="expanded"
)

# ------------ STYLING ------------


# Custom CSS - balanced version
def load_css():
    return """
    <style>
    /* STRV colors */
    :root {
        --strv-red: #ff0043;
        --strv-black: #101010;
        --strv-dark-gray: #2a2a2a;
        --strv-gray: #4a4a4a;
        --strv-light-gray: #e0e0e0;
    }
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* Header */
    .main-header {
        background-color: var(--strv-black);
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .main-header span.red {
        color: var(--strv-red);
    }
    
    /* Story layout */
    .stories-container {
        display: flex;
        gap: 15px;
        padding: 10px 0;
        overflow-x: auto;
    }
    
    .story-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        border: 2px solid var(--strv-red);
    }
    
    /* Action buttons */
    .action-button {
        display: inline-block;
        padding: 5px 10px;
        background-color: var(--strv-light-gray);
        color: var(--strv-black);
        border-radius: 4px;
        margin: 5px;
        font-size: 14px;
        cursor: pointer;
    }
    
    .action-button:hover {
        background-color: var(--strv-red);
        color: white;
    }
    
    .action-button.active {
        background-color: var(--strv-red);
        color: white;
    }
    </style>
    """


# ------------ FILTERS ------------


# Apply filter function
def apply_filter(image, filter_name):
    """Apply filter to image"""
    if image is None:
        return None

    img = image.copy()

    if filter_name == "Original":
        return img
    elif filter_name == "Grayscale":
        return ImageOps.grayscale(img)
    elif filter_name == "Warm":
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.5)
        return img
    elif filter_name == "Cool":
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(0.7)
    elif filter_name == "Vintage":
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.8)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.2)
    elif filter_name == "Blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2))

    return img


# Path helper
def get_image_path(path):
    """Ensure the image path exists"""
    if path is None:
        return None

    # Check original path
    if os.path.exists(path):
        return path

    # Try with models folder
    if "models" not in path:
        alt_path = os.path.join("models", os.path.basename(path))
        if os.path.exists(alt_path):
            return alt_path

    # Try with src/models folder
    alt_path = os.path.join("src", "models", os.path.basename(path))
    if os.path.exists(alt_path):
        return alt_path

    return None


# API functions
@st.cache_data
def get_api_stats():
    """Get statistics about the loaded datasets"""
    try:
        response = requests.get(f"{API_URL}/stats")
        return response.json()
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def get_sample_paths():
    """Get sample image paths from the API"""
    try:
        response = requests.get(f"{API_URL}/stats")
        stats = response.json()

        # Check if sample paths are available
        if "images" in stats and "sample_paths" in stats["images"]:
            return stats["images"]["sample_paths"]
        return []
    except Exception as e:
        st.error(f"Error getting sample paths: {str(e)}")
        return []


def search_similar_images(image_bytes):
    """Send image to API and get similar images"""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(f"{API_URL}/search/image", files=files)
        return response.json()
    except Exception as e:
        st.error(f"Error during image search: {str(e)}")
        return None


def get_recommendations(item_id=None, user_likes=None, user_saves=None):
    """
    Simple recommendation function (normally would use a more sophisticated algorithm)

    For this demo, it either:
    1. Returns similar images if an item_id is provided
    2. Or gets a random set of images from the database
    """
    try:
        # Get sample paths
        sample_paths = get_sample_paths()
        valid_paths = [path for path in sample_paths if get_image_path(path)]

        # If we don't have enough sample paths, use a mock search
        if len(valid_paths) < 6:
            # Do a mock search to get more images
            mock_results = {"status": "success", "results": []}

            # Create some dummy results
            for i in range(10):
                mock_results["results"].append(
                    {
                        "path": f"https://picsum.photos/seed/recommendation{i}/300",
                        "similarity_score": random.uniform(0.7, 0.95),
                    }
                )

            return mock_results

        # Otherwise use actual images
        mock_results = {"status": "success", "results": []}

        # Use the valid paths
        for i, path in enumerate(valid_paths[:10]):
            mock_results["results"].append(
                {"path": path, "similarity_score": random.uniform(0.7, 0.95)}
            )

        return mock_results
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        # Return a fallback
        return {
            "status": "success",
            "results": [
                {"path": f"https://picsum.photos/seed/fallback{i}/300", "similarity_score": 0.8}
                for i in range(6)
            ],
        }


# ------------ AUTHENTICATION ------------


# Simple auth functions
def hash_password(password):
    """Create a simple hash for a password"""
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate(username, password):
    """Simple authentication - in a real app, this would check a database"""
    # For demo, just accept any non-empty username and password
    if username and password:
        # Hash the password (in a real app, you'd compare with a stored hash)
        hash_password(password)

        # Set user session
        st.session_state.user = {
            "username": username,
            "logged_in": True,
            "login_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return True
    return False


def logout():
    """Log out the user"""
    if "user" in st.session_state:
        del st.session_state.user
    st.session_state.show_login = True


# ------------ SESSION STATE MANAGEMENT ------------

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.likes = set()
    st.session_state.saved = set()
    st.session_state.applied_filter = "Original"
    st.session_state.show_results = False
    st.session_state.active_category = "For You"
    st.session_state.show_login = True
    st.session_state.categories = ["For You", "Design", "Technology", "Photography", "Architecture"]
    st.session_state.explore_data = None


# User interaction functions
def toggle_like(item_id):
    """Toggle like status for an item"""
    if item_id in st.session_state.likes:
        st.session_state.likes.remove(item_id)
    else:
        st.session_state.likes.add(item_id)


def toggle_save(item_id):
    """Toggle save status for an item"""
    if item_id in st.session_state.saved:
        st.session_state.saved.remove(item_id)
    else:
        st.session_state.saved.add(item_id)


def set_active_category(category):
    """Set the active category"""
    st.session_state.active_category = category


# ------------ LOGIN COMPONENT ------------


def login_form():
    """Display a login form"""
    st.markdown(
        """
        <div style="text-align:center;padding:30px;">
            <h2>Welcome to STRV Vision</h2>
            <p>Please log in to continue</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Log In")

        if submit:
            if authenticate(username, password):
                st.session_state.show_login = False
                st.rerun()
            else:
                st.error("Invalid username or password")

    # Add a demo login option
    if st.button("Demo Login"):
        authenticate("demo_user", "demo_password")
        st.session_state.show_login = False
        st.rerun()


# ------------ LOAD EXPLORE DATA ------------


def load_explore_data():
    """Load data for the explore page (uses recommendation engine)"""
    # Only load once or when requested
    if st.session_state.explore_data is None:
        # Get recommendations based on likes and saves
        recommendations = get_recommendations(
            user_likes=st.session_state.likes, user_saves=st.session_state.saved
        )

        # Process recommendations results
        if recommendations and recommendations.get("status") == "success":
            explore_data = []
            for i, item in enumerate(recommendations["results"]):
                # Get valid image path
                path = item["path"]
                valid_path = get_image_path(path)

                # If valid path exists, add to explore data
                if valid_path and os.path.exists(valid_path):
                    explore_data.append(
                        {
                            "id": f"explore_{i}",
                            "path": valid_path,
                            "username": f"creator_{i % 5}",
                            "likes": random.randint(10, 500),
                            "description": f"Similar to content you've liked",
                            "similarity": item["similarity_score"],
                        }
                    )
                else:
                    # Use a fallback image if path doesn't exist
                    explore_data.append(
                        {
                            "id": f"explore_{i}",
                            "path": f"https://picsum.photos/seed/explore{i}/300",
                            "username": f"creator_{i % 5}",
                            "likes": random.randint(10, 500),
                            "description": f"Recommended for you",
                            "similarity": random.uniform(0.7, 0.95),
                        }
                    )

            # Store in session state
            st.session_state.explore_data = explore_data


# ------------ MAIN APP ------------


def main():
    # Apply custom CSS
    st.markdown(load_css(), unsafe_allow_html=True)

    # Show login screen if not logged in
    if st.session_state.get("show_login", True) and not st.session_state.get("user"):
        login_form()
        return

    # Display app header
    st.markdown(
        """
        <div class="main-header">
            <h1>👁️ STRV <span class="red">Vision</span></h1>
            <p>Visual Intelligence Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get API stats
    stats = get_api_stats()

    # Create sidebar
    with st.sidebar:
        # User section
        if "user" in st.session_state:
            st.subheader(f"Welcome, {st.session_state.user['username']}")
            st.write(f"Logged in since: {st.session_state.user['login_time']}")

            # Logout button
            if st.button("Logout"):
                logout()
                st.rerun()

        st.title("About STRV Vision")
        st.write(
            """
        STRV Vision uses AI to find visually similar images. 
        Upload an image to discover content that matches your style.
        """
        )

        if stats:
            st.subheader("Database Stats")
            st.write(f"• {stats['images']['total']} images indexed")
            st.write(f"• Using {stats['search_method']} search algorithm")
            if "using_faiss" in stats:
                st.write(f"• FAISS acceleration: {stats.get('using_faiss', False)}")

        st.markdown("---")

        # Saved items section
        st.subheader("Saved Items")

        if st.session_state.saved:
            # Show saved items
            saved_items = list(st.session_state.saved)

            for i, item_id in enumerate(saved_items):
                if i >= 4:  # Limit to 4 items
                    break

                # Display item
                st.write(f"Item {i+1}")

                # If item_id starts with 'explore_', it's a path
                if isinstance(item_id, str) and os.path.exists(item_id):
                    # It's a file path
                    try:
                        st.image(Image.open(item_id), use_container_width=True)
                    except:
                        st.image(
                            f"https://picsum.photos/seed/saved{i}/300", use_container_width=True
                        )
                else:
                    # It's a generated ID
                    st.image(f"https://picsum.photos/seed/saved{i}/300", use_container_width=True)

                # Add remove button
                if st.button("Remove", key=f"remove_{item_id}"):
                    toggle_save(item_id)
                    st.rerun()

            # Show count if more than 4
            if len(saved_items) > 4:
                st.write(f"+ {len(saved_items) - 4} more saved items")
        else:
            st.write("No saved items yet. Save images by clicking the bookmark icon.")

        # Refresh explore page
        if st.button("Refresh Recommendations"):
            st.session_state.explore_data = None
            st.rerun()

    # Main content area with tabs
    tab1, tab2 = st.tabs(["Explore", "Search"])

    with tab1:
        # Load explore data
        load_explore_data()

        # Category selection - interactive
        st.subheader("Discover")

        # Create category buttons
        cat_cols = st.columns(len(st.session_state.categories))
        for i, category in enumerate(st.session_state.categories):
            with cat_cols[i]:
                if st.button(
                    category,
                    key=f"cat_{category}",
                    type="primary" if category == st.session_state.active_category else "secondary",
                ):
                    set_active_category(category)
                    st.rerun()

        # Featured section - using real images if available
        st.subheader("Recommended for You")

        # Use sample images from API if available
        sample_paths = get_sample_paths()
        valid_samples = [path for path in sample_paths if get_image_path(path)]

        if valid_samples and len(valid_samples) >= 2:
            # Use real images
            sample1 = get_image_path(valid_samples[0])
            sample2 = get_image_path(valid_samples[1])

            feat_col1, feat_col2 = st.columns(2)

            with feat_col1:
                st.image(sample1, use_container_width=True)
                st.markdown("**Design Inspiration**")
                st.write("Similar to your saved content")

                # Like button
                if st.button(
                    "❤️" if sample1 in st.session_state.likes else "🤍", key=f"like_feat_1"
                ):
                    toggle_like(sample1)
                    st.rerun()

            with feat_col2:
                st.image(sample2, use_container_width=True)
                st.markdown("**Visual Style**")
                st.write("Based on your preferences")

                # Like button
                if st.button(
                    "❤️" if sample2 in st.session_state.likes else "🤍", key=f"like_feat_2"
                ):
                    toggle_like(sample2)
                    st.rerun()
        else:
            # Use placeholder images
            feat_col1, feat_col2 = st.columns(2)

            with feat_col1:
                st.image(
                    "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80"
                )
                st.markdown("**Design Excellence**")
                st.write("Upload your design to find similar inspirations")

            with feat_col2:
                st.image(
                    "https://images.unsplash.com/photo-1531403009284-440f080d1e12?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80"
                )
                st.markdown("**Tech Workspace**")
                st.write("Discover workspace design themes")

        # Explore grid - using real recommendations
        st.subheader("For You")

        if st.session_state.explore_data:
            # Show recommendation explanation
            if st.session_state.likes or st.session_state.saved:
                st.info("These recommendations are based on content you've liked and saved")
            else:
                st.info("Start liking or saving items to get personalized recommendations")

            # Display recommended items in a grid
            items_per_row = 3

            for i in range(0, len(st.session_state.explore_data), items_per_row):
                # Create a row of columns
                row_cols = st.columns(items_per_row)

                # Fill each column with an item
                for j in range(items_per_row):
                    idx = i + j
                    if idx < len(st.session_state.explore_data):
                        item = st.session_state.explore_data[idx]
                        with row_cols[j]:
                            # Display image
                            try:
                                if os.path.exists(item["path"]):
                                    st.image(Image.open(item["path"]), use_container_width=True)
                                else:
                                    st.image(
                                        f"https://picsum.photos/seed/explore{idx}/300",
                                        use_container_width=True,
                                    )
                            except:
                                st.image(
                                    f"https://picsum.photos/seed/explore{idx}/300",
                                    use_container_width=True,
                                )

                            # Item details
                            st.markdown(f"**@{item['username']}** • {item['likes']} likes")
                            st.caption(item["description"])

                            # Like button
                            if st.button(
                                "❤️" if item["path"] in st.session_state.likes else "🤍",
                                key=f"like_explore_{idx}",
                            ):
                                toggle_like(item["path"])
                                st.rerun()

                            # Save button - separate line to avoid nested columns error
                            if st.button(
                                "📌" if item["path"] in st.session_state.saved else "📍",
                                key=f"save_explore_{idx}",
                            ):
                                toggle_save(item["path"])
                                st.rerun()
        else:
            st.write("Loading recommendations...")

        # Tips section
        with st.expander("Tips for Better Results"):
            st.write("• Upload clear, high-quality images to get the best matches")
            st.write("• Try different filters to see how they affect your search results")
            st.write("• Like and save items to improve your recommendations")
            st.write("• Use the search tab to find specific visual matches")

    with tab2:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Upload Image")

            # Upload image section
            uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                # Load and display the image
                try:
                    image = Image.open(uploaded_file)

                    # Filter selection
                    st.subheader("Apply Filter")

                    # Filter options with visual display
                    filter_options = ["Original", "Grayscale", "Warm", "Cool", "Vintage", "Blur"]

                    # Display filter options as buttons
                    filter_cols = st.columns(3)
                    for i, filter_name in enumerate(filter_options):
                        with filter_cols[i % 3]:
                            if st.button(
                                filter_name,
                                key=f"filter_{filter_name}",
                                type=(
                                    "primary"
                                    if filter_name == st.session_state.applied_filter
                                    else "secondary"
                                ),
                            ):
                                st.session_state.applied_filter = filter_name
                                st.rerun()

                    # Apply the selected filter
                    filtered_image = apply_filter(image, st.session_state.applied_filter)

                    # Display filtered image
                    st.image(
                        filtered_image,
                        caption=f"{st.session_state.applied_filter} Filter",
                        use_container_width=True,
                    )

                    # Search button
                    if st.button("🔍 Find Similar Images", type="primary"):
                        # Use the filtered image for search
                        img_byte_arr = io.BytesIO()
                        filtered_image.save(img_byte_arr, format="JPEG")
                        img_bytes = img_byte_arr.getvalue()

                        with st.spinner("Looking for similar images..."):
                            results = search_similar_images(img_bytes)

                            if results and results.get("status") == "success":
                                st.session_state.search_results = results["results"]
                                st.session_state.show_results = True
                                st.rerun()
                            else:
                                st.error("No results found or error in search")

                except Exception as e:
                    st.error(f"Error processing image: {e}")
            else:
                # Simple upload prompt
                st.info("Upload an image to find visually similar content")

        with col2:
            if st.session_state.show_results and "search_results" in st.session_state:
                st.subheader("Similar Images")

                results = st.session_state.search_results

                # Show best match
                if results and len(results) > 0:
                    st.subheader("Best Match")

                    img_path = results[0]["path"]
                    valid_path = get_image_path(img_path)

                    if valid_path and os.path.exists(valid_path):
                        # Display image
                        st.image(Image.open(valid_path), use_container_width=True)

                        # Match details
                        similarity = results[0]["similarity_score"]
                        st.markdown(f"**Match Score:** {similarity:.1%}")

                        # Like button
                        if st.button(
                            "❤️" if valid_path in st.session_state.likes else "🤍",
                            key=f"like_result_0",
                        ):
                            toggle_like(valid_path)
                            st.rerun()

                        # Save button
                        if st.button(
                            "📌" if valid_path in st.session_state.saved else "📍",
                            key=f"save_result_0",
                        ):
                            toggle_save(valid_path)
                            st.rerun()
                    else:
                        st.error(f"Image not found at path: {img_path}")

                # Show more results in a grid
                if len(results) > 1:
                    st.subheader("More Similar Images")

                    # Display in rows of 3
                    for i in range(1, min(len(results), 10), 3):
                        # Create a row of columns
                        cols = st.columns(3)

                        # Fill each column with an item
                        for j in range(3):
                            idx = i + j
                            if idx < len(results):
                                img_data = results[idx]
                                img_path = img_data["path"]
                                valid_path = get_image_path(img_path)

                                with cols[j]:
                                    if valid_path and os.path.exists(valid_path):
                                        # Image
                                        st.image(Image.open(valid_path), use_container_width=True)

                                        # Similarity score
                                        similarity = img_data["similarity_score"]
                                        st.markdown(f"**Match:** {similarity:.1%}")

                                        # Like button
                                        if st.button(
                                            "❤️" if valid_path in st.session_state.likes else "🤍",
                                            key=f"like_result_{idx}",
                                        ):
                                            toggle_like(valid_path)
                                            st.rerun()

                                        # Save button
                                        if st.button(
                                            "📌" if valid_path in st.session_state.saved else "📍",
                                            key=f"save_result_{idx}",
                                        ):
                                            toggle_save(valid_path)
                                            st.rerun()
            else:
                st.subheader("Upload an image to see similar results")
                st.write("Your search results will appear here.")


if __name__ == "__main__":
    main()
