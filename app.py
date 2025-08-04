import os
import uuid
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import your existing modules used for text analysis
from src.core.pdf_handler import PDFHandler, find_pdf_files
from src.core.embedding_generator import EmbeddingGenerator
from src.core.semantic_similarity import PDFSimilarityCalculator
from src.core.sequence_similarity import SequenceSimilarityCalculator
from src.core.exact_match import ExactMatchDetector

# Import image extraction and duplication detection modules
from src.utils.pdf_img_extractor import PDFImageExtractor
from src.utils.image_duplication_detector import build_dataset_from_results, index_and_report_cross_pdf_duplicates

# Import and setup production logging
from src.core.logging_config import setup_logging

# Configure production logging
setup_logging(
    log_level="INFO",
    log_dir="logs", 
    structured_logging=True,
    enable_console=True,
    enable_file=True
)


def create_spinner(spinner_type="loading", icon=None):
    """Create different types of spinning animations."""
    if icon:
        return f'<div class="spinner-container"><span class="spinner-icon">{icon}</span></div>'
    else:
        return f'<div class="spinner-container"><div class="spinner spinner-{spinner_type}"></div></div>'


def initialize_session_state():
    """Initialize all session state variables."""
    if "text_analysis_complete" not in st.session_state:
        st.session_state.text_analysis_complete = False
    if "image_analysis_complete" not in st.session_state:
        st.session_state.image_analysis_complete = False
    if "image_analysis_started" not in st.session_state:
        st.session_state.image_analysis_started = False
    if "current_directory" not in st.session_state:
        st.session_state.current_directory = None
    if "semantic_scores" not in st.session_state:
        st.session_state.semantic_scores = None
    if "sequence_scores" not in st.session_state:
        st.session_state.sequence_scores = None
    if "exact_matches" not in st.session_state:
        st.session_state.exact_matches = None
    if "image_analysis_logs" not in st.session_state:
        st.session_state.image_analysis_logs = []
    if "duplicate_image_pairs" not in st.session_state:
        st.session_state.duplicate_image_pairs = []


def initialize_app():
    """Setup the Streamlit app configuration and enhanced styling."""
    st.set_page_config(
        page_title="📚 Smart Document Similarity Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .main-title {
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .main-subtitle {
        color: #ffffff;
        font-size: 1.3rem;
        margin-top: 1rem;
        opacity: 0.95;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .success-card {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #22c55e;
    }
    .warning-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        border-left: 5px solid #3b82f6 !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    .info-card {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #3b82f6;
    }
    .similarity-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        margin: 0.3rem;
        font-size: 0.9rem;
    }
    .high-similarity { 
        background: linear-gradient(135deg, #fee2e2 0%, #fca5a5 100%); 
        color: #dc2626; 
        border: 2px solid #ef4444;
    }
    .medium-similarity { 
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
        color: #d97706; 
        border: 2px solid #f59e0b;
    }
    .low-similarity { 
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
        color: #059669; 
        border: 2px solid #10b981;
    }
    .match-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        border-left: 4px solid #3b82f6 !important;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: bold;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .progress-container {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Spinning animation styles */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .spinner {
        display: inline-block;
        width: 24px;
        height: 24px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 8px;
    }
    
    .spinner-icon {
        display: inline-block;
        font-size: 1.5rem;
        animation: spin 2s linear infinite;
        margin-right: 8px;
    }
    
    .spinner-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 40px;
    }
    
    /* Different colored spinners for different stages */
    .spinner-loading {
        border-top-color: #3b82f6;
    }
    .spinner-processing {
        border-top-color: #8b5cf6;
    }
    .spinner-analyzing {
        border-top-color: #06b6d4;
    }
    .spinner-finalizing {
        border-top-color: #10b981;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">📚 Smart Document Analyzer</h1>
        <p class="main-subtitle">AI-Powered Document Similarity Detection • Find Similar Content Instantly</p>
    </div>
    """, unsafe_allow_html=True)


def reset_analysis_state(new_directory):
    """Reset analysis state when directory changes."""
    st.session_state.text_analysis_complete = False
    st.session_state.image_analysis_complete = False
    st.session_state.image_analysis_started = False
    st.session_state.current_directory = new_directory
    st.session_state.semantic_scores = None
    st.session_state.sequence_scores = None
    st.session_state.exact_matches = None
    st.session_state.image_analysis_logs = []
    st.session_state.duplicate_image_pairs = []


def get_directory_input():
    """Simple sidebar for getting PDF directory path or uploading files."""
    import os
    import tempfile
    
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h3>🎛️ Analysis Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Add tabs for different input methods
    tab1, tab2 = st.sidebar.tabs(["📁 Server Path", "📤 Upload Files"])
    
    with tab1:
        st.markdown("### 📁 Server Directory")
        directory = st.text_input(
            "📂 Enter Server Directory Path", 
            value=st.session_state.current_directory or "",
            help="Enter the full path to PDFs on the server",
            placeholder="/app/pdfs or ./pdfs"
        )
    
    with tab2:
        st.markdown("### 📤 Upload Directory")
        
        # Option 1: Zip file upload (recommended for directories)
        st.markdown("**Option 1: Upload as ZIP file** (Recommended)")
        zip_file = st.file_uploader(
            "Upload ZIP file containing PDFs", 
            type="zip",
            help="Zip your PDF directory and upload it here"
        )
        
        if zip_file:
            import zipfile
            # Create temporary directory for extracted files
            if 'upload_dir' not in st.session_state:
                st.session_state.upload_dir = tempfile.mkdtemp(prefix="uploaded_pdfs_")
            
            upload_dir = st.session_state.upload_dir
            
            # Extract zip file
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(upload_dir)
                
                # Count PDF files in extracted directory
                pdf_count = 0
                for root, dirs, files in os.walk(upload_dir):
                    pdf_count += len([f for f in files if f.lower().endswith('.pdf')])
                
                directory = upload_dir
                st.success(f"✅ Extracted ZIP file with {pdf_count} PDF files")
                
            except Exception as e:
                st.error(f"❌ Error extracting ZIP file: {str(e)}")
                directory = None
        else:
            # Option 2: Multiple file selection
            st.markdown("**Option 2: Select all PDFs from your directory**")
            uploaded_files = st.file_uploader(
                "Select all PDF files from your directory", 
                type="pdf", 
                accept_multiple_files=True,
                help="Hold Ctrl/Cmd and select all PDFs from your directory"
            )
            
            if uploaded_files:
                # Create temporary directory for uploaded files
                if 'upload_dir' not in st.session_state:
                    st.session_state.upload_dir = tempfile.mkdtemp(prefix="uploaded_pdfs_")
                
                upload_dir = st.session_state.upload_dir
                
                # Save uploaded files
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(upload_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                directory = upload_dir
                st.success(f"✅ Uploaded {len(uploaded_files)} PDF files")
            else:
                directory = None
    
    # Simple directory check
    if directory != st.session_state.current_directory and directory:
        if os.path.exists(directory):
            reset_analysis_state(directory)
        
    if directory and os.path.exists(directory):
        # Use recursive search to find PDFs in subdirectories (same as ZIP extraction)
        pdf_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    # Store relative path for display
                    rel_path = os.path.relpath(os.path.join(root, file), directory)
                    pdf_files.append(rel_path)
        
        file_count = len(pdf_files)
        
        if file_count > 0:
            st.sidebar.markdown(f"""
            <div class="success-card">
                ✅ <strong>Folder Found!</strong><br>
                📊 Found {file_count} PDF files<br>
                📁 Ready for analysis
            </div>
            """, unsafe_allow_html=True)
            
            with st.sidebar.expander("📋 Files Found", expanded=False):
                for i, file in enumerate(pdf_files[:10], 1):
                    st.write(f"{i}. {file}")
                if file_count > 10:
                    st.write(f"... and {file_count - 10} more files")
        else:
            st.sidebar.markdown("""
            <div class="warning-card">
                ⚠️ <strong>No PDF Files Found</strong><br>
                The folder exists but contains no PDF files.
            </div>
            """, unsafe_allow_html=True)
    elif directory:
        st.sidebar.markdown("""
        <div class="warning-card">
            ❌ <strong>Folder Not Found</strong><br>
            Please check the path and try again.
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar.expander("ℹ️ How It Works", expanded=False):
        st.markdown("""
        **📝 Text Analysis:**<br>
        • Smart content understanding<br>
        • Text pattern matching<br>
        • Exact copy detection<br><br>
        **🖼️ Image Analysis:**<br>
        • AI-powered image comparison<br>
        • Cross-document duplicate detection<br>
        • Visual similarity scoring<br><br>
        **🎯 Complete Analysis:**<br>
        • Run both analyses together<br>
        • Comprehensive plagiarism detection
        """)
    
    return directory


def get_similarity_level(score, score_type="semantic"):
    """Determine similarity level, badge label, and description."""
    if score_type == "semantic":
        if score >= 0.7:
            return "high-similarity", "High", f"{score:.1%}"
        elif score >= 0.4:
            return "medium-similarity", "Medium", f"{score:.1%}"
        else:
            return "low-similarity", "Low", f"{score:.1%}"
    else:
        score_norm = score/100 if score > 1 else score
        if score_norm >= 0.7:
            return "high-similarity", "High", f"{score_norm:.1%}"
        elif score_norm >= 0.4:
            return "medium-similarity", "Medium", f"{score_norm:.1%}"
        else:
            return "low-similarity", "Low", f"{score_norm:.1%}"


def create_similarity_overview_charts(semantic_scores, sequence_scores, exact_matches):
    """Display charts and overview metrics for document similarity."""
    st.markdown("## 📊 Your Document Analysis Results")
    
    # Metric cards for overall statistics
    total_comparisons = len(semantic_scores)
    avg_semantic = np.mean([v[2] for v in semantic_scores.values()]) if semantic_scores else 0
    avg_sequence = np.mean([v[2] for v in sequence_scores.values()]) if sequence_scores else 0
    exact_count = len(exact_matches)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3b82f6;">📋 {total_comparisons}</h3>
            <p>Document Comparisons Made</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #8b5cf6;">🧠 {avg_semantic:.1%}</h3>
            <p>Average Content Similarity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #06b6d4;">🔤 {avg_sequence:.2f}</h3>
            <p>Average Text Pattern Match</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #f59e0b;">📊 {exact_count}</h3>
            <p>Exact Copies Found</p>
        </div>
        """, unsafe_allow_html=True)


def create_similarity_matrix_visualization(semantic_scores, sequence_scores):
    """Visualize similarity matrices as tables."""
    st.markdown("## 🗂️ Detailed Document Comparison Charts")
    
    all_pairs = list(semantic_scores.keys()) if semantic_scores else list(sequence_scores.keys())
    pdf_names = sorted(list(set([pdf for pair in all_pairs for pdf in pair])))
    n = len(pdf_names)
    
    # Initialize matrices
    semantic_mean = np.zeros((n, n))
    semantic_max = np.zeros((n, n))
    semantic_min = np.zeros((n, n))
    sequence_mean = np.zeros((n, n))
    sequence_max = np.zeros((n, n))
    sequence_min = np.zeros((n, n))
    
    # Fill semantic matrices
    if semantic_scores:
        for (pdf1, pdf2), score_tuple in semantic_scores.items():
            i, j = pdf_names.index(pdf1), pdf_names.index(pdf2)
            semantic_max[i][j] = score_tuple[0]
            semantic_max[j][i] = score_tuple[0]
            semantic_min[i][j] = score_tuple[1]
            semantic_min[j][i] = score_tuple[1]
            semantic_mean[i][j] = score_tuple[2]
            semantic_mean[j][i] = score_tuple[2]
    
    # Fill sequence matrices
    if sequence_scores:
        for (pdf1, pdf2), score_tuple in sequence_scores.items():
            i, j = pdf_names.index(pdf1), pdf_names.index(pdf2)
            sequence_max[i][j] = score_tuple[0]
            sequence_max[j][i] = score_tuple[0]
            sequence_min[i][j] = score_tuple[1]
            sequence_min[j][i] = score_tuple[1]
            sequence_mean[i][j] = score_tuple[2]
            sequence_mean[j][i] = score_tuple[2]
    
    def show_matrix(matrix, title, is_semantic=True):
        df = pd.DataFrame(
            matrix,
            index=[name.replace('.pdf', '') for name in pdf_names],
            columns=[name.replace('.pdf', '') for name in pdf_names]
        )
        if not is_semantic and df.values.max() > 1:
            df = df / 100.0
        st.markdown(f"**{title}**")
        st.dataframe(
            df.style.format("{:.1%}").background_gradient(cmap='viridis', vmin=0, vmax=1),
            use_container_width=True
        )
    
    # Semantic similarity matrices
    st.markdown("<h3 style='margin-top:2rem;'>🧠 Content Meaning Comparison</h3>", unsafe_allow_html=True)
    semantic_tabs = st.tabs(["Average", "Highest", "Lowest"])
    
    with semantic_tabs[0]:
        if semantic_scores:
            show_matrix(semantic_mean, "Average Content Similarity", True)
        else:
            st.info("No content similarity data available.")
    
    with semantic_tabs[1]:
        if semantic_scores:
            show_matrix(semantic_max, "Highest Content Similarity", True)
        else:
            st.info("No content similarity data available.")
    
    with semantic_tabs[2]:
        if semantic_scores:
            show_matrix(semantic_min, "Lowest Content Similarity", True)
        else:
            st.info("No content similarity data available.")
    
    # Sequence similarity matrices
    st.markdown("<h3 style='margin-top:2rem;'>🔤 Text Pattern Comparison</h3>", unsafe_allow_html=True)
    sequence_tabs = st.tabs(["Average", "Highest", "Lowest"])
    
    with sequence_tabs[0]:
        if sequence_scores:
            show_matrix(sequence_mean, "Average Text Pattern Match", False)
        else:
            st.info("No text pattern data available.")
    
    with sequence_tabs[1]:
        if sequence_scores:
            show_matrix(sequence_max, "Highest Text Pattern Match", False)
        else:
            st.info("No text pattern data available.")
    
    with sequence_tabs[2]:
        if sequence_scores:
            show_matrix(sequence_min, "Lowest Text Pattern Match", False)
        else:
            st.info("No text pattern data available.")


def display_exact_matches(exact_matches):
    """Display exact match results."""
    st.markdown("## 📊 Exact Copy Detection Results")
    
    if not exact_matches:
        st.markdown("""
        <div class="success-card">
            <h3>✅ No Exact Copies Found</h3>
            <p>Great! No identical text was found between your documents.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"""
    <div class="warning-card">
        <h3>🚨 Identical Text Found</h3>
        <p>Found <strong>{len(exact_matches)}</strong> cases where text appears to be copied exactly.</p>
        <p>You may want to review these matches.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Group matches by document pair
    matches_by_pair = {}
    for match in exact_matches:
        pdfA, chunkA, pdfB, chunkB, content = match
        pair_key = (pdfA, pdfB)
        if pair_key not in matches_by_pair:
            matches_by_pair[pair_key] = []
        matches_by_pair[pair_key].append((chunkA, chunkB, content))
    
    for i, ((pdfA, pdfB), matches) in enumerate(matches_by_pair.items(), 1):
        pdfA_short = pdfA.replace('.pdf', '')
        pdfB_short = pdfB.replace('.pdf', '')
        
        with st.expander(f"🔍 Duplicate Group #{i}: {pdfA_short} ↔ {pdfB_short} ({len(matches)} matches)", expanded=i==1):
            st.markdown(f"""
            <div class="match-highlight">
                <strong>Documents:</strong> {pdfA_short} and {pdfB_short}<br>
                <strong>Total Matches:</strong> {len(matches)}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Identical Content Blocks:**")
            
            # Show first 3 matches by default
            def display_match(match_data, match_num):
                chunkA, chunkB, content = match_data
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span style="background: #fef3c7; color: #b45309; font-weight: bold; border-radius: 8px; padding: 0.4rem 0.8rem; margin-right: 1rem;">
                            📄 {pdfA_short} Page {chunkA+1}
                        </span>
                        <span style="font-size: 1.5rem; margin: 0 0.5rem;">⇄</span>
                        <span style="background: #dbeafe; color: #1e40af; font-weight: bold; border-radius: 8px; padding: 0.4rem 0.8rem;">
                            📄 {pdfB_short} Page {chunkB+1}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show the actual matched content
                if content and len(content.strip()) > 0:
                    # Truncate long content for better display
                    display_content = content.strip()
                    if len(display_content) > 300:
                        display_content = display_content[:300] + "..."
                    
                    st.markdown(
                        f"""
                        <div style="background: #f8fafc; border-left: 4px solid #3b82f6; padding: 1rem; margin: 0.5rem 0 1rem 0; border-radius: 4px; color: #1f2937;">
                            <strong style="color: #374151;">📝 Matched Text:</strong><br>
                            <em style="color: #4b5563; font-size: 0.95rem; line-height: 1.5;">"{display_content}"</em>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 0.5rem; margin: 0.5rem 0 1rem 0; border-radius: 4px; font-style: italic; color: #92400e;">
                            ⚠️ Content preview not available
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Display first 3 matches
            for j, match in enumerate(matches[:3], 1):
                display_match(match, j)
            
            # If there are more than 3 matches, show them in an expander
            if len(matches) > 3:
                remaining_matches = matches[3:]
                with st.expander(f"🔍 Show {len(remaining_matches)} more identical blocks", expanded=False):
                    for j, match in enumerate(remaining_matches, 4):
                        display_match(match, j)
                        st.markdown("---")  # Add separator between matches


def create_pdf_image_extractor():
    """Create a new instance of PDFImageExtractor with temporary directory."""
    return PDFImageExtractor()  # Will use temporary directory by default


def run_image_analysis(directory):
    """Run image plagiarism analysis with proper session state management."""
    
    # If already completed, display results
    if st.session_state.image_analysis_complete:
        st.markdown("### 🖼️ Image Comparison Results")
        
        # Display logs
        for log in st.session_state.image_analysis_logs:
            st.markdown(log)
        
        # Display duplicate pairs
        if st.session_state.duplicate_image_pairs:
            st.markdown("### 🚨 Duplicate Images Detected")
            for pair in st.session_state.duplicate_image_pairs:
                st.markdown(pair)
        else:
            st.success("✅ No duplicate images were found.")
        return
    
    # Run the analysis
    progress_container = st.container()
    
    with progress_container:
        st.markdown("""
        <div class="progress-container">
            <span style='font-size:2rem;'>🖼️</span> Finding Images in Your Documents<br>
            <small>Extracting and comparing images from your PDF files...</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress with spinner for image analysis
        img_col1, img_col2 = st.columns([1, 20])
        with img_col1:
            img_spinner_placeholder = st.empty()
        with img_col2:
            progress_bar = st.progress(0)
        
        progress_text = st.empty()
        
        # Show initial spinner
        with img_spinner_placeholder:
            st.markdown('<div class="spinner-container"><div class="spinner spinner-loading"></div></div>', unsafe_allow_html=True)
        
        try:
            # Use context manager for automatic cleanup
            with create_pdf_image_extractor() as extractor:
                # Use the same PDF discovery logic as text analysis (recursive search)
                try:
                    pdf_file_paths = find_pdf_files(directory)
                    pdf_files = [os.path.basename(path) for path in pdf_file_paths]
                    total_pdfs = len(pdf_files)
                    
                    if total_pdfs == 0:
                        st.error("No PDF files found in the directory.")
                        return
                    
                    st.success(f"✅ Found {total_pdfs} PDF files for image analysis")
                except Exception as e:
                    st.error(f"Error finding PDF files: {str(e)}")
                    return
                
                all_results = []
                st.session_state.image_analysis_logs = []
                
                # Extract images from all PDFs
                for i, pdf_path in enumerate(pdf_file_paths, 1):
                    with img_spinner_placeholder:
                        st.markdown('<div class="spinner-container"><div class="spinner spinner-processing"></div></div>', unsafe_allow_html=True)
                        
                    pdf_filename = os.path.basename(pdf_path)
                    results = extractor.extract_images_from_pdf(pdf_path)
                    all_results.extend(results)
                    
                    progress_bar.progress(i / total_pdfs * 0.8)  # 80% for extraction
                    progress_text.markdown(f"📄 Looking for images in document {i}/{total_pdfs}: {pdf_filename}")
                    
                    log_msg = f"📸 Found {len(results)} images in {pdf_filename}"
                    st.session_state.image_analysis_logs.append(log_msg)
                    st.markdown(log_msg)
                
                if not all_results:
                    st.warning("No images were found in the provided PDF files.")
                    st.session_state.image_analysis_complete = True
                    return
            
                # Create dataset and analyze similarities
                with img_spinner_placeholder:
                    st.markdown('<div class="spinner-container"><div class="spinner spinner-analyzing"></div></div>', unsafe_allow_html=True)
                progress_text.markdown("📊 Organizing found images...")
                progress_bar.progress(0.85)
                
                dataset_name = f"imgs_{uuid.uuid4().hex[:8]}"
                dataset = build_dataset_from_results(all_results, dataset_name)
                
                # Enhanced embedding progress display
                embedding_container = st.container()
                with embedding_container:
                    st.markdown("""
                    <div style="padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                        <h4 style="color: #ffffff; margin: 0; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">🎨 Teaching AI to Understand Your Images</h4>
                        <p style="color: #ffffff; margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.95; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
                            Using advanced AI to analyze what's in each image...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Embedding progress with spinner
                    emb_col1, emb_col2 = st.columns([1, 20])
                    with emb_col1:
                        emb_spinner_placeholder = st.empty()
                    with emb_col2:
                        embedding_progress = st.progress(0)
                    
                    embedding_status = st.empty()
                    embedding_details = st.empty()
                    
                    # Show embedding spinner
                    with emb_spinner_placeholder:
                        st.markdown('<div class="spinner-container"><div class="spinner spinner-analyzing"></div></div>', unsafe_allow_html=True)
                
                # Progress callback function
                def embedding_progress_callback(stage, current, total, message):
                    if stage == "starting":
                        with emb_spinner_placeholder:
                            st.markdown('<div class="spinner-container"><div class="spinner spinner-loading"></div></div>', unsafe_allow_html=True)
                        embedding_progress.progress(0.1)
                        embedding_status.markdown(f"🚀 **Getting AI ready to analyze images...**")
                        embedding_details.info(f"Teaching AI to understand {total} images from your documents...")
                        
                    elif stage == "embedding_complete":
                        with emb_spinner_placeholder:
                            st.markdown('<div class="spinner-container"><div class="spinner spinner-analyzing"></div></div>', unsafe_allow_html=True)
                        embedding_progress.progress(0.8)
                        embedding_status.markdown(f"⚡ **Building image comparison system...**")
                        embedding_details.success(f"✅ AI has learned to recognize all {total} images!")
                        
                    elif stage == "finding_duplicates":
                        with emb_spinner_placeholder:
                            st.markdown('<div class="spinner-container"><div class="spinner spinner-finalizing"></div></div>', unsafe_allow_html=True)
                        embedding_progress.progress(0.9)
                        embedding_status.markdown(f"🔍 **Looking for similar images...**")
                        embedding_details.info("Comparing images to find potential matches...")
                        
                    elif stage == "complete":
                        emb_spinner_placeholder.empty()
                        embedding_progress.progress(1.0)
                        embedding_status.markdown(f"🎉 **Image analysis complete!**")
                        embedding_details.success("Successfully analyzed all images for similarities!")
                        
                    elif stage == "error":
                        emb_spinner_placeholder.empty()
                        embedding_progress.progress(0)
                        embedding_status.markdown(f"❌ **Something went wrong...**")
                        embedding_details.error("Could not analyze images. Please try again or check your files.")
                
                # Run the embedding computation with progress tracking
                try:
                    pairs_info = index_and_report_cross_pdf_duplicates(
                        dataset, 
                        brain_key="cross_pdf", 
                        thresh=0.2,
                        progress_callback=embedding_progress_callback
                    )
                    
                    # Small delay to show completion status
                    import time
                    time.sleep(1)
                    
                except Exception as e:
                    emb_spinner_placeholder.empty()
                    st.error(f"Could not analyze images: {str(e)}")
                    pairs_info = None
                
                # Clean up embedding progress display
                embedding_container.empty()
                
                with img_spinner_placeholder:
                    st.markdown('<div class="spinner-container"><div class="spinner spinner-finalizing"></div></div>', unsafe_allow_html=True)
                progress_text.markdown("✨ Finishing up image analysis...")
                progress_bar.progress(0.95)
                
                # Store results in session state
                st.session_state.duplicate_image_pairs = []
                
                if pairs_info:
                    results_container = st.container()
                    
                    with results_container:
                        st.markdown("""
                        <div class="warning-card">
                            <h3>🚨 Duplicate Images Found</h3>
                            <p>The following images show significant similarities across documents.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Convert pairs_info to match_data format and sort by similarity
                        matches = []
                        for pair in pairs_info:
                            match_data = {
                                'orig_filepath': pair['orig_path'],
                                'dup_filepath': pair['dup_path'],
                                'orig_pdf': pair['orig_pdf'],
                                'dup_pdf': pair['dup_pdf'],
                                'score': pair['similarity']
                            }
                            matches.append(match_data)
                        
                        # Sort all matches by similarity score
                        matches.sort(key=lambda x: x['score'], reverse=True)
                        
                        # Display all matches
                        for i, match in enumerate(matches, 1):
                            match_container = st.container()
                            with match_container:
                                st.markdown(f"""
                                <div style="padding: 10px; border-radius: 10px; margin-bottom: 20px; background-color: rgba(255, 255, 255, 0.1);">
                                    <h4>Match Pair #{i}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(
                                        match['orig_filepath'],
                                        caption=f"Document: {match['orig_pdf']}",
                                        use_container_width=True
                                    )
                                with col2:
                                    st.image(
                                        match['dup_filepath'],
                                        caption=f"Document: {match['dup_pdf']}",
                                        use_container_width=True
                                    )
                                
                                # Store for session state
                                pair_info = f"**Match #{i}:** {match['orig_pdf']} ↔ {match['dup_pdf']} (Score: {match['score']:.2%})"
                                st.session_state.duplicate_image_pairs.append(pair_info)
                else:
                    st.success("✅ No duplicate images found across PDF files.")
                
                progress_bar.progress(1.0)
                img_spinner_placeholder.empty()
                progress_text.markdown("✅ Image analysis complete!")
                
                # Mark as complete
                st.session_state.image_analysis_complete = True
            # Temporary directory is automatically cleaned up when exiting the context manager
            
        except Exception as e:
            st.error(f"An error occurred during image analysis: {str(e)}")
            logging.error(f"Image analysis error: {str(e)}")


def run_text_analysis(directory, chunk_size=5000, similarity_threshold=0.3):
    """Run the complete plagiarism text analysis with progress tracking."""
    import logging
    from src.core.validation import DirectoryValidator, ParameterValidator
    
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    try:
        DirectoryValidator.validate_directory_path(directory, must_exist=True, must_be_readable=True)
        ParameterValidator.validate_positive_integer(chunk_size, "chunk_size", min_value=100, max_value=50000)
        ParameterValidator.validate_positive_float(similarity_threshold, "similarity_threshold", min_value=0.0, max_value=1.0)
    except Exception as e:
        st.error(f"❌ Invalid input: {str(e)}")
        logger.error(f"Input validation failed: {str(e)}")
        return None, None, None
    
    progress_container = st.container()
    
    try:
        with progress_container:
            st.markdown("""
            <div class="progress-container">
                <span style='font-size:2rem;'>🚀</span> Analyzing Your Documents<br>
                <small>This may take a few minutes depending on document size...</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress with spinner
            col1, col2 = st.columns([1, 20])
            with col1:
                spinner_placeholder = st.empty()
            with col2:
                overall_progress = st.progress(0)
            
            status_text = st.empty()
            step_info = st.empty()
            
            # Show spinner
            with spinner_placeholder:
                st.markdown('<div class="spinner-container"><div class="spinner spinner-loading"></div></div>', unsafe_allow_html=True)
            
            # Step 1: Load PDFs
            with spinner_placeholder:
                st.markdown('<div class="spinner-container"><div class="spinner spinner-loading"></div></div>', unsafe_allow_html=True)
            status_text.info("📂 Finding your PDF documents...")
            step_info.text("Step 1/5: Looking for PDF files in your folder")
            
            pdf_handler = PDFHandler(directory)
            pdf_count = pdf_handler.get_pdf_count()
            
            if pdf_count < 2:
                spinner_placeholder.empty()
                st.error("❌ Need at least 2 PDF files for comparison analysis.")
                return None, None, None
            
            overall_progress.progress(20)
            st.success(f"✅ Found {pdf_count} PDF documents to analyze")
            
            # Step 2: Extract content
            with spinner_placeholder:
                st.markdown('<div class="spinner-container"><div class="spinner spinner-processing"></div></div>', unsafe_allow_html=True)
            status_text.info("📄 Reading document content...")
            step_info.text("Step 2/5: Extracting text from your PDF files")
            
            chunks = pdf_handler.extract_page_chunks(chunk_size=chunk_size)
            overall_progress.progress(40)
            
            # Step 3: Generate embeddings
            with spinner_placeholder:
                st.markdown('<div class="spinner-container"><div class="spinner spinner-analyzing"></div></div>', unsafe_allow_html=True)
            status_text.info("🧠 Teaching AI to understand your documents...")
            step_info.text("Step 3/5: Creating smart summaries of document content")
            
            embedder = EmbeddingGenerator()
            embeddings = embedder.generate_embeddings(chunks)
            overall_progress.progress(60)
            
            # Check if any embeddings were generated
            if not embeddings or len(embeddings) == 0:
                spinner_placeholder.empty()
                st.error("❌ No text content could be extracted from your PDF files.")
                st.error("This could be because:")
                st.error("• PDFs contain only scanned images without text")
                st.error("• PDFs are password protected")
                st.error("• PDFs are corrupted or invalid")
                st.error("• File encoding issues with Arabic/Unicode filenames")
                logger.error("No embeddings generated - cannot proceed with analysis")
                return None, None, None
            
            # Check if we have valid embeddings for at least 2 PDFs
            valid_embeddings = {k: v for k, v in embeddings.items() if v and len(v) > 0}
            if len(valid_embeddings) < 2:
                spinner_placeholder.empty()
                st.error(f"❌ Need at least 2 PDFs with extractable text. Found {len(valid_embeddings)} valid PDF(s).")
                st.info("PDF processing results:")
                for pdf_name, emb_list in embeddings.items():
                    if emb_list and len(emb_list) > 0:
                        st.success(f"✅ {pdf_name}: {len(emb_list)} text chunks")
                    else:
                        st.error(f"❌ {pdf_name}: No text content found")
                logger.error(f"Only {len(valid_embeddings)} PDFs have valid embeddings - cannot proceed")
                return None, None, None
            
            # Step 4: Compute similarities
            with spinner_placeholder:
                st.markdown('<div class="spinner-container"><div class="spinner spinner-analyzing"></div></div>', unsafe_allow_html=True)
            status_text.info("🔍 Comparing documents for similarities...")
            step_info.text("Step 4/5: Finding matching content between documents")
            
            similarity_calculator = PDFSimilarityCalculator(valid_embeddings)
            semantic_scores = similarity_calculator.compute_all_pdf_similarities()
            
            seq_similarity_calculator = SequenceSimilarityCalculator()
            sequence_scores = seq_similarity_calculator.compute_all_pdf_similarities(chunks)
            
            overall_progress.progress(80)
            
            # Step 5: Find exact matches
            with spinner_placeholder:
                st.markdown('<div class="spinner-container"><div class="spinner spinner-finalizing"></div></div>', unsafe_allow_html=True)
            status_text.info("🎯 Finding exact duplicate content...")
            step_info.text("Step 5/5: Identifying identical text passages")
            
            exact_match_detector = ExactMatchDetector()
            exact_matches = exact_match_detector.find_exact_matches(chunks)
            
            overall_progress.progress(100)
            spinner_placeholder.empty()
            progress_container.empty()
            
            st.markdown("""
            <div class="success-card">
                <h3>🎉 Analysis Complete!</h3>
                <p>Your documents have been analyzed using advanced AI techniques.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Store results in session state
            st.session_state.semantic_scores = semantic_scores
            st.session_state.sequence_scores = sequence_scores
            st.session_state.exact_matches = exact_matches
            
            return semantic_scores, sequence_scores, exact_matches
            
    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Analysis failed: {str(e)}")
        logging.error(f"Analysis error: {str(e)}")
        return None, None, None


def main():
    """Main application function."""
    initialize_app()
    directory = get_directory_input()
    
    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎮 Analysis Controls")
    
    # Analysis Options Section
    if directory and os.path.exists(directory):
        st.sidebar.markdown("### 📝 Text Analysis")
        
        # Text Analysis Controls
        if not st.session_state.text_analysis_complete:
            if st.sidebar.button("🚀 Start Text Analysis", type="primary"):
                semantic_scores, sequence_scores, exact_matches = run_text_analysis(directory)
                if semantic_scores is not None:
                    st.session_state.text_analysis_complete = True
                    st.rerun()  # Refresh to show results
        else:
            st.sidebar.success("✅ Text Analysis Complete")
            if st.sidebar.button("🔄 Analyze Text Again"):
                st.session_state.text_analysis_complete = False
                st.session_state.semantic_scores = None
                st.session_state.sequence_scores = None
                st.session_state.exact_matches = None
                st.rerun()
        
        # Image Analysis Section
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🖼️ Image Analysis")
        
        if not st.session_state.image_analysis_started:
            if st.sidebar.button("🖼️ Start Image Analysis", type="secondary"):
                st.session_state.image_analysis_started = True
                st.rerun()
        elif not st.session_state.image_analysis_complete:
            st.sidebar.info("🔄 Analyzing Images...")
        else:
            st.sidebar.success("✅ Image Analysis Complete")
            if st.sidebar.button("🔄 Analyze Images Again"):
                st.session_state.image_analysis_started = False
                st.session_state.image_analysis_complete = False
                st.session_state.image_analysis_logs = []
                st.session_state.duplicate_image_pairs = []
                st.rerun()
        
        # Combined Analysis Option
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🚀 Complete Analysis")
        
        if not st.session_state.text_analysis_complete and not st.session_state.image_analysis_started:
            if st.sidebar.button("🎯 Analyze Both Text & Images", type="primary"):
                # Start text analysis first
                semantic_scores, sequence_scores, exact_matches = run_text_analysis(directory)
                if semantic_scores is not None:
                    st.session_state.text_analysis_complete = True
                    # Then start image analysis
                    st.session_state.image_analysis_started = True
                    st.rerun()
        elif st.session_state.text_analysis_complete and not st.session_state.image_analysis_started:
            if st.sidebar.button("🖼️ Add Image Analysis", type="secondary"):
                st.session_state.image_analysis_started = True
                st.rerun()
        elif not st.session_state.text_analysis_complete and st.session_state.image_analysis_complete:
            if st.sidebar.button("📝 Add Text Analysis", type="secondary"):
                semantic_scores, sequence_scores, exact_matches = run_text_analysis(directory)
                if semantic_scores is not None:
                    st.session_state.text_analysis_complete = True
                    st.rerun()
        elif st.session_state.text_analysis_complete and st.session_state.image_analysis_complete:
            st.sidebar.success("🎉 Complete Analysis Done!")
            if st.sidebar.button("🔄 Start Fresh Analysis"):
                # Reset everything
                st.session_state.text_analysis_complete = False
                st.session_state.image_analysis_started = False
                st.session_state.image_analysis_complete = False
                st.session_state.semantic_scores = None
                st.session_state.sequence_scores = None
                st.session_state.exact_matches = None
                st.session_state.image_analysis_logs = []
                st.session_state.duplicate_image_pairs = []
                st.rerun()
    
    # Main content area
    if directory and os.path.exists(directory):
        # Display text analysis results if completed
        if st.session_state.text_analysis_complete and st.session_state.semantic_scores is not None:
            st.markdown("## 📝 Text Similarity Analysis Results")
            create_similarity_overview_charts(
                st.session_state.semantic_scores, 
                st.session_state.sequence_scores, 
                st.session_state.exact_matches
            )
            create_similarity_matrix_visualization(
                st.session_state.semantic_scores, 
                st.session_state.sequence_scores
            )
            display_exact_matches(st.session_state.exact_matches)
        
        # Display image analysis section if started (independent of text analysis)
        if st.session_state.image_analysis_started:
            if st.session_state.text_analysis_complete:
                st.markdown("---")  # Add separator if both analyses are shown
            st.markdown("## 🖼️ Image Similarity Detection")
            run_image_analysis(directory)
        
        # Welcome message when no analysis has been run
        if not st.session_state.text_analysis_complete and not st.session_state.image_analysis_started:
            st.markdown("""
            <div class="info-card">
                <h3>👋 Ready to Check Your Documents for Similarities!</h3>
                <p><strong>Choose your analysis type:</strong></p>
                <ul>
                    <li>🧠 <strong>Text Analysis:</strong> Smart content analysis, pattern matching, and copy detection</li>
                    <li>🖼️ <strong>Image Analysis:</strong> Find duplicate or similar images across documents</li>
                    <li>🎯 <strong>Complete Analysis:</strong> Both text and image analysis for comprehensive results</li>
                </ul>
                <p>👈 <strong>Get started:</strong> Choose an analysis option in the sidebar!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show analysis options info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #3b82f6;">📝 Text Analysis</h3>
                    <p><strong>What it finds:</strong></p>
                    <ul style="text-align: left; font-size: 0.9rem;">
                        <li>Similar content meaning</li>
                        <li>Matching text patterns</li>
                        <li>Exact copied text</li>
                        <li>Similarity scores & charts</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #8b5cf6;">🖼️ Image Analysis</h3>
                    <p><strong>What it finds:</strong></p>
                    <ul style="text-align: left; font-size: 0.9rem;">
                        <li>Duplicate images</li>
                        <li>Similar visual content</li>
                        <li>Cross-document matches</li>
                        <li>AI-powered comparison</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # No directory selected or invalid directory
        st.markdown("""
        <div class="info-card">
            <h3>👋 Welcome to Smart Document Analyzer!</h3>
            <p><strong>Flexible Analysis Options:</strong></p>
            <ul>
                <li>📝 <strong>Text Analysis:</strong> AI content analysis, pattern matching, and duplicate detection</li>
                <li>🖼️ <strong>Image Analysis:</strong> Find duplicate images across your documents</li>
                <li>🎯 <strong>Complete Analysis:</strong> Run both text and image analysis together</li>
                <li>📈 <strong>Visual Reports:</strong> Easy-to-understand charts and summaries</li>
            </ul>
            <p>👈 <strong>Get started:</strong> Enter your PDF folder path in the sidebar!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Sample Analysis Dashboard")
        st.info("Enter a PDF directory path to see analysis results here!")
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; margin-top: 2rem;">
        <p>🔬 <strong>Smart Document Analyzer</strong> - Find Similar Content in Your Documents</p>
        <p>Built with advanced AI technology to help you understand your document similarities</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()