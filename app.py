import os
import uuid
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import psutil
import torch

# Load environment variables from .env file
load_dotenv()

# Import your existing modules used for text analysis
from src.core.pdf_handler import PDFHandler, find_pdf_files
from src.core.embedding_generator import EmbeddingGenerator
from src.core.semantic_similarity import PDFSimilarityCalculator
from src.core.sequence_similarity import SequenceSimilarityCalculator
from src.core.exact_match import ExactMatchDetector
from src.core.validation import ContentValidator

# Import image extraction and duplication detection modules
from src.utils.pdf_img_extractor import PDFImageExtractor
from src.utils.image_duplication_detector import ImageDuplicationDetector

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

# Global configuration for concurrent processing
CONCURRENT_CONFIG = {
    'max_workers_pdf': 4,  # Max workers for PDF processing
    'max_workers_embedding': 2,  # Max workers for embedding generation
    'max_workers_similarity': 4,  # Max workers for similarity calculation
    'max_workers_image': 3,  # Max workers for image processing
    'batch_size_embedding': 32,  # Batch size for embedding generation
    'timeout_seconds': 300,  # 5 minutes timeout for operations
    'memory_limit_mb': 2048,  # 2GB memory limit
}

# Progress tracking for concurrent operations
class ProgressTracker:
    def __init__(self):
        self.current_stage = "idle"
        self.progress = 0.0
        self.status_message = ""
        self.stage_progress = {}
        self.start_time = None
        self.operation_lock = threading.Lock()
    
    def update_progress(self, stage, progress, message):
        with self.operation_lock:
            self.current_stage = stage
            self.progress = progress
            self.status_message = message
            self.stage_progress[stage] = progress
    
    def get_progress(self):
        with self.operation_lock:
            return {
                'stage': self.current_stage,
                'progress': self.progress,
                'message': self.status_message,
                'stage_progress': self.stage_progress.copy()
            }

# Global progress tracker
progress_tracker = ProgressTracker()

def check_system_resources():
    """Check if system has enough resources for concurrent processing."""
    try:
        # Check available memory
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        
        # Check disk space
        disk = psutil.disk_usage('/')
        available_gb = disk.free / (1024 * 1024 * 1024)
        
        logger = logging.getLogger(__name__)
        logger.info(f"System resources - Memory: {available_mb:.1f}MB, CPU: {cpu_count}, Disk: {available_gb:.1f}GB")
        
        # Adjust concurrent processing based on available resources
        if available_mb < 1024:  # Less than 1GB RAM
            CONCURRENT_CONFIG['max_workers_pdf'] = 2
            CONCURRENT_CONFIG['max_workers_embedding'] = 1
            CONCURRENT_CONFIG['max_workers_similarity'] = 2
            CONCURRENT_CONFIG['max_workers_image'] = 2
            logger.warning("Low memory detected, reducing concurrent workers")
        
        if cpu_count < 4:
            CONCURRENT_CONFIG['max_workers_pdf'] = min(2, cpu_count)
            CONCURRENT_CONFIG['max_workers_embedding'] = 1
            CONCURRENT_CONFIG['max_workers_similarity'] = min(2, cpu_count)
            CONCURRENT_CONFIG['max_workers_image'] = min(2, cpu_count)
            logger.warning("Limited CPU cores detected, reducing concurrent workers")
        
        if available_gb < 1:  # Less than 1GB disk space
            logger.warning("Low disk space detected, consider cleaning up")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to check system resources: {e}")
        return False

def extract_images_from_pdf_worker(extractor, pdf_path):
    """Worker function for parallel image extraction from PDF."""
    try:
        return extractor.extract_images_from_pdf(pdf_path)
    except Exception as e:
        logging.error(f"Failed to extract images from {pdf_path}: {e}")
        return []

def process_pdf_worker(pdf_handler, pdf_path, chunk_size):
    """Worker function for parallel PDF text extraction."""
    try:
        return pdf_handler._extract_chunks_from_pdf(pdf_path, chunk_size)
    except Exception as e:
        logging.error(f"Failed to process PDF {pdf_path}: {e}")
        return os.path.basename(pdf_path), []

def compute_similarity_worker(similarity_calculator, pdf_pair):
    """Worker function for parallel similarity computation."""
    try:
        pdfA, pdfB = pdf_pair
        embA = similarity_calculator.embeddings.get(pdfA, [])
        embB = similarity_calculator.embeddings.get(pdfB, [])
        
        if not embA or not embB:
            return pdf_pair, None
        
        # Convert lists to tensors if needed
        if isinstance(embA, list):
            embA = torch.stack([torch.as_tensor(e) for e in embA])
        if isinstance(embB, list):
            embB = torch.stack([torch.as_tensor(e) for e in embB])
        
        max_sim, min_sim, mean_sim = similarity_calculator.compute_pairwise_similarity(embA, embB)
        return pdf_pair, (max_sim, min_sim, mean_sim)
        
    except Exception as e:
        logging.error(f"Failed to compute similarity for pair {pdf_pair}: {e}")
        return pdf_pair, None


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
    if "skipped_pdfs" not in st.session_state:
        st.session_state.skipped_pdfs = []


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
    /* Import Google Fonts for modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root variables for easy theming */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        --secondary-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --success-gradient: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --danger-gradient: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
        --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --text-primary: #1a202c;
        --text-secondary: #4a5568;
        --text-light: #ffffff;
        --shadow-sm: 0 2px 4px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 40px rgba(0,0,0,0.15);
        --shadow-xl: 0 20px 60px rgba(0,0,0,0.2);
    }
    
    /* Base styling improvements */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Enhanced main header with glassmorphism effect */
    .main-header {
        background: var(--primary-gradient);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
        animation: headerFloat 6s ease-in-out infinite;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 8s linear infinite;
        pointer-events: none;
    }
    
    .main-title {
        color: #ffffff;
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 4px 4px 15px rgba(0,0,0,0.8), 0 0 30px rgba(255,255,255,0.6), 0 0 40px rgba(255,255,255,0.4), 0 0 50px rgba(255,255,255,0.2);
        letter-spacing: -0.02em;
        position: relative;
        z-index: 2;
        -webkit-text-stroke: 2px rgba(0,0,0,0.3);
        animation: textShimmer 3s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(255,255,255,0.9)) drop-shadow(0 0 30px rgba(255,255,255,0.5));
    }
    
    .main-subtitle {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1rem;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.6), 0 0 20px rgba(255,255,255,0.4), 0 0 25px rgba(255,255,255,0.2);
        position: relative;
        z-index: 2;
        letter-spacing: 0.01em;
        -webkit-text-stroke: 1px rgba(0,0,0,0.2);
        filter: drop-shadow(0 0 12px rgba(255,255,255,0.5));
    }
    
    /* Enhanced metric cards with hover effects */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: var(--shadow-md);
        text-align: center;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.6s;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-xl);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card h3 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        background: var(--primary-gradient);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card h3 span {
        -webkit-text-fill-color: initial;
        color: inherit;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    
    .metric-card p {
        color: #2d3748 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Enhanced success cards */
    .success-card {
        background: var(--success-gradient);
        color: #065f46 !important;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: none;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
        animation: pulseGlow 2s ease-in-out infinite;
    }
    
    .success-card::after {
        content: '✨';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 1.5rem;
        opacity: 0.7;
    }
    
    .success-card h3 {
        color: #065f46 !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: none !important;
    }
    
    /* Enhanced warning cards */
    .warning-card {
        background: var(--warning-gradient) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3) !important;
        box-shadow: var(--shadow-lg) !important;
        position: relative;
        overflow: hidden;
    }
    
    .warning-card::before {
        content: '⚠️';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 1.5rem;
        opacity: 0.8;
    }
    
    /* Enhanced info cards */
    .info-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 197, 253, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        color: var(--text-primary) !important;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: var(--shadow-md);
        position: relative;
    }
    
    .info-card h3 {
        color: #1e40af !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
    
    .info-card ul {
        color: var(--text-secondary) !important;
    }
    
    /* Enhanced similarity badges */
    .similarity-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        margin: 0.3rem;
        font-size: 0.9rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .similarity-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .similarity-badge:hover::before {
        left: 100%;
    }
    
    .high-similarity { 
        background: var(--danger-gradient);
        color: #ffffff !important;
        border: none;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .medium-similarity { 
        background: var(--warning-gradient);
        color: #ffffff !important;
        border: none;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .low-similarity { 
        background: var(--success-gradient);
        color: #065f46 !important;
        border: none;
        font-weight: 600;
    }
    
    /* Enhanced match highlight */
    .match-highlight {
        background: var(--primary-gradient) !important;
        color: var(--text-light) !important;
        border: none !important;
        padding: 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        margin: 1rem 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3) !important;
        box-shadow: var(--shadow-lg) !important;
        position: relative;
        overflow: hidden;
    }
    
    .match-highlight::before {
        content: '🔍';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 1.2rem;
        opacity: 0.8;
    }
    
    /* Enhanced sidebar header */
    .sidebar-header {
        background: var(--primary-gradient);
        padding: 2rem;
        border-radius: 16px;
        color: var(--text-light);
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-header::before {
        content: '⚙️';
        position: absolute;
        top: -10px;
        right: -10px;
        font-size: 3rem;
        opacity: 0.1;
        transform: rotate(45deg);
    }
    
    /* Enhanced progress container */
    .progress-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .progress-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--primary-gradient);
        animation: progressShimmer 2s linear infinite;
    }
    
    /* Enhanced animations */
    @keyframes headerFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(0.5deg); }
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes textShimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes pulseGlow {
        0%, 100% { box-shadow: var(--shadow-md); }
        50% { box-shadow: 0 4px 20px rgba(132, 250, 176, 0.4); }
    }
    
    @keyframes progressShimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Spinning animation styles - enhanced */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes spinGlow {
        0% { 
            transform: rotate(0deg);
            filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.5));
        }
        100% { 
            transform: rotate(360deg);
            filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.8));
        }
    }
    
    .spinner {
        display: inline-block;
        width: 28px;
        height: 28px;
        border: 3px solid rgba(255,255,255,0.3);
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spinGlow 1s linear infinite;
        margin-right: 12px;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .spinner-icon {
        display: inline-block;
        font-size: 1.8rem;
        animation: spinGlow 2s linear infinite;
        margin-right: 12px;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
    }
    
    .spinner-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 50px;
        padding: 1rem;
    }
    
    /* Enhanced colored spinners */
    .spinner-loading {
        border-top-color: #3b82f6;
        filter: drop-shadow(0 0 8px rgba(59, 130, 246, 0.6));
    }
    .spinner-processing {
        border-top-color: #8b5cf6;
        filter: drop-shadow(0 0 8px rgba(139, 92, 246, 0.6));
    }
    .spinner-analyzing {
        border-top-color: #06b6d4;
        filter: drop-shadow(0 0 8px rgba(6, 182, 212, 0.6));
    }
    .spinner-finalizing {
        border-top-color: #10b981;
        filter: drop-shadow(0 0 8px rgba(16, 185, 129, 0.6));
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-md) !important;
        text-transform: none !important;
        letter-spacing: 0.01em !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-lg) !important;
        filter: brightness(110%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Enhanced sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    /* Minimal expander styling - don't interfere with functionality */
    
    /* Enhanced dataframe styling */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        .main-subtitle {
            font-size: 1.1rem;
        }
        .metric-card {
            padding: 1.5rem;
        }
        .metric-card h3 {
            font-size: 2rem !important;
        }
    }
    
    /* Dark theme support */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #f7fafc;
            --text-secondary: #e2e8f0;
        }
        
        .stApp {
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        }
        
        .metric-card {
            background: rgba(45, 55, 72, 0.8);
            color: var(--text-primary);
        }
        
        .metric-card p {
            color: #e2e8f0 !important;
            font-weight: 600 !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        
        .metric-card h3 span {
            -webkit-text-fill-color: initial !important;
            color: inherit !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
        }
        
        .info-card {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(147, 197, 253, 0.2) 100%);
            color: var(--text-primary) !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚀 Smart Document Analyzer</h1>
        <p class="main-subtitle">✨ AI-Powered Document Similarity Detection • Find Similar Content Instantly ✨</p>
        <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="background: rgba(255,255,255,0.15); padding: 0.5rem 1rem; border-radius: 25px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                <span style="color: #ffffff; font-size: 0.85rem; font-weight: 700; text-shadow: 2px 2px 5px rgba(0,0,0,0.5), 0 0 10px rgba(255,255,255,0.3); -webkit-text-stroke: 0.5px rgba(0,0,0,0.2);">🧠 Smart AI Analysis</span>
            </div>
            <div style="background: rgba(255,255,255,0.15); padding: 0.5rem 1rem; border-radius: 25px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                <span style="color: #ffffff; font-size: 0.85rem; font-weight: 700; text-shadow: 2px 2px 5px rgba(0,0,0,0.5), 0 0 10px rgba(255,255,255,0.3); -webkit-text-stroke: 0.5px rgba(0,0,0,0.2);">🖼️ Image Detection</span>
            </div>
            <div style="background: rgba(255,255,255,0.15); padding: 0.5rem 1rem; border-radius: 25px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                <span style="color: #ffffff; font-size: 0.85rem; font-weight: 700; text-shadow: 2px 2px 5px rgba(0,0,0,0.5), 0 0 10px rgba(255,255,255,0.3); -webkit-text-stroke: 0.5px rgba(0,0,0,0.2);">⚡ Lightning Fast</span>
            </div>
        </div>
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
    st.session_state.skipped_pdfs = []


def get_directory_input():
    """Simple sidebar for uploading PDF files as ZIP."""
    import os
    import tempfile
    import zipfile
    
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h3>🎛️ Analysis Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 📤 Upload Your PDFs")
    st.sidebar.markdown("**Upload as ZIP file containing your PDF documents**")
    
    zip_file = st.sidebar.file_uploader(
        "📁 Choose ZIP file containing PDFs", 
        type="zip",
        help="Create a ZIP file with all your PDF documents and upload it here"
    )
    
    directory = None
    
    if zip_file:
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
            st.sidebar.success(f"✅ Extracted ZIP file with {pdf_count} PDF files")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error extracting ZIP file: {str(e)}")
            directory = None
    
    # Directory check and state management
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
                ✅ <strong>ZIP Extracted Successfully!</strong><br>
                📊 Found {file_count} PDF files<br>
                📁 Ready for analysis
            </div>
            """, unsafe_allow_html=True)
            
            with st.sidebar.expander("📋 Files Found", expanded=False):
                for i, file in enumerate(pdf_files[:10], 1):
                    st.sidebar.write(f"{i}. {file}")
                if file_count > 10:
                    st.sidebar.write(f"... and {file_count - 10} more files")
        else:
            st.sidebar.markdown("""
            <div class="warning-card">
                ⚠️ <strong>No PDF Files Found</strong><br>
                The ZIP file contains no PDF files.
            </div>
            """, unsafe_allow_html=True)
    
    with st.sidebar.expander("ℹ️ How to Use", expanded=False):
        st.sidebar.markdown("""
        **📁 Prepare Your Files:**
        • Put all PDF documents in a folder
        • Create a ZIP file of that folder
        • Upload the ZIP file above
        
        **📝 Text Analysis:**
        • Smart content understanding
        • Text pattern matching
        • Exact copy detection
        
        **🖼️ Image Analysis:**
        • Perceptual hash comparison
        • Cross-document duplicate detection
        • Fast similarity scoring
        
        **🎯 Complete Analysis:**
        • Run both analyses together
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
            <h3 style="color: #3b82f6;"><span>📋</span> {total_comparisons}</h3>
            <p>Document Comparisons Made</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #8b5cf6;"><span>🧠</span> {avg_semantic:.1%}</h3>
            <p>Average Content Similarity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #06b6d4;"><span>🔤</span> {avg_sequence:.2f}</h3>
            <p>Average Text Pattern Match</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #f59e0b;"><span>📊</span> {exact_count}</h3>
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


def display_semantic_similarity_with_content(semantic_scores, chunks_with_text, threshold=0.7):
    """Display semantic similarity results with actual matched content."""
    st.markdown("## 🧠 Semantic Similarity Analysis with Content")
    
    if not semantic_scores:
        st.markdown("""
        <div class="success-card">
            <h3>✅ No Semantic Similarities Found</h3>
            <p>Great! No significant semantic similarities were detected between your documents.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Filter high similarity matches
    high_similarity_matches = []
    for (pdfA, pdfB), (max_sim, min_sim, mean_sim) in semantic_scores.items():
        if mean_sim >= threshold:
            high_similarity_matches.append((pdfA, pdfB, max_sim, min_sim, mean_sim))
    
    if not high_similarity_matches:
        st.markdown("""
        <div class="success-card">
            <h3>✅ No High Similarity Content Found</h3>
            <p>No content with similarity above {threshold:.1%} was detected between your documents.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Sort by mean similarity (highest first)
    high_similarity_matches.sort(key=lambda x: x[4], reverse=True)
    
    st.markdown(f"""
    <div class="warning-card">
        <h3>🚨 High Similarity Content Found</h3>
        <p>Found <strong>{len(high_similarity_matches)}</strong> document pairs with semantic similarity above {threshold:.1%}.</p>
        <p>Review these matches to check for potential content reuse.</p>
    </div>
    """, unsafe_allow_html=True)
    
    for i, (pdfA, pdfB, max_sim, min_sim, mean_sim) in enumerate(high_similarity_matches, 1):
        pdfA_short = pdfA.replace('.pdf', '')
        pdfB_short = pdfB.replace('.pdf', '')
        
        with st.expander(f"🔍 Similarity #{i}: {pdfA_short} ↔ {pdfB_short} (Mean: {mean_sim:.1%})", expanded=i==1):
            st.markdown(f"""
            <div class="match-highlight">
                <strong>Documents:</strong> {pdfA_short} and {pdfB_short}<br>
                <strong>Similarity Scores:</strong> Max: {max_sim:.1%}, Min: {min_sim:.1%}, Mean: {mean_sim:.1%}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample content from both documents
            if pdfA in chunks_with_text and pdfB in chunks_with_text:
                chunksA = chunks_with_text[pdfA]
                chunksB = chunks_with_text[pdfB]
                
                if chunksA and chunksB:
                    # Show first chunk from each document as sample
                    sampleA = chunksA[0][:300] + "..." if len(chunksA[0]) > 300 else chunksA[0]
                    sampleB = chunksB[0][:300] + "..." if len(chunksB[0]) > 300 else chunksB[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div style="background: #fef3c7; border: 1px solid #f59e0b; padding: 1rem; border-radius: 8px;">
                            <strong>📄 {pdfA_short} (Sample Content):</strong><br>
                            <div style="background: #ffffff; padding: 0.5rem; border-radius: 4px; margin-top: 0.5rem; font-size: 0.9rem; line-height: 1.4; color: #1f2937;">
                                {sampleA}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: #dbeafe; border: 1px solid #3b82f6; padding: 1rem; border-radius: 8px;">
                            <strong>📄 {pdfB_short} (Sample Content):</strong><br>
                            <div style="background: #ffffff; padding: 0.5rem; border-radius: 4px; margin-top: 0.5rem; font-size: 0.9rem; line-height: 1.4; color: #1f2937;">
                                {sampleB}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div style="background: #e0f2fe; border-left: 4px solid #0288d1; padding: 0.75rem; margin: 1rem 0; border-radius: 4px; color: #01579b;">
                        <strong>💡 Note:</strong> These are sample content snippets. The similarity score indicates how similar the overall content and meaning are between these documents.
                    </div>
                    """, unsafe_allow_html=True)

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
                    # Clean and format the content for better display
                    display_content = content.strip()
                    
                    # Show full content with option to expand/collapse
                    if len(display_content) > 500:
                        # Create expandable content for long matches
                        with st.expander(f"📝 View Matched Text ({len(display_content)} characters)", expanded=False):
                            st.markdown(
                                f"""
                                <div style="background: #fef3c7; border: 2px solid #f59e0b; padding: 1.5rem; margin: 0.5rem 0; border-radius: 8px; color: #92400e;">
                                    <strong style="color: #92400e; font-size: 1.1rem;">🚨 IDENTICAL TEXT DETECTED:</strong><br><br>
                                    <div style="background: #ffffff; padding: 1rem; border-radius: 6px; border-left: 4px solid #f59e0b; font-family: 'Courier New', monospace; font-size: 0.9rem; line-height: 1.6; color: #1f2937; max-height: 400px; overflow-y: auto;">
                                        {display_content}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        # Show shorter content directly
                        st.markdown(
                            f"""
                            <div style="background: #fef3c7; border: 2px solid #f59e0b; padding: 1.5rem; margin: 0.5rem 0; border-radius: 8px; color: #92400e;">
                                <strong style="color: #92400e; font-size: 1.1rem;">🚨 IDENTICAL TEXT DETECTED:</strong><br><br>
                                <div style="background: #ffffff; padding: 1rem; border-radius: 6px; border-left: 4px solid #f59e0b; font-family: 'Courier New', monospace; font-size: 0.9rem; line-height: 1.6; color: #1f2937;">
                                    {display_content}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Add character count and match statistics
                    word_count = len(display_content.split())
                    st.markdown(
                        f"""
                        <div style="background: #e0f2fe; border-left: 4px solid #0288d1; padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px; color: #01579b;">
                            <strong>📊 Match Statistics:</strong> {len(display_content)} characters, {word_count} words
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


@st.cache_resource(show_spinner=False)
def get_cached_embedding_generator(model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', batch_size: int | None = None):
    """Return a cached EmbeddingGenerator instance to avoid reloading the model on reruns."""
    return EmbeddingGenerator(model_name=model_name, batch_size=batch_size)


def run_image_analysis_concurrent(directory):
    """Run image plagiarism analysis with enhanced concurrent processing."""
    
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
    # Container to show persistent scanned/empty PDF notices
    skipped_info_container = st.container()
    progress_container = st.container()
    
    with progress_container:
        st.markdown("""
        <div class="progress-container">
            <span style='font-size:2rem;'>🖼️</span> Finding Images in Your Documents (Concurrent Mode)<br>
            <small>Using parallel processing for faster image analysis...</small>
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
                
                # Extract images from all PDFs using parallel processing
                with ThreadPoolExecutor(max_workers=CONCURRENT_CONFIG['max_workers_image']) as executor:
                    # Submit all PDF processing tasks
                    future_to_pdf = {
                        executor.submit(extract_images_from_pdf_worker, extractor, pdf_path): pdf_path 
                        for pdf_path in pdf_file_paths
                    }
                    
                    completed_pdfs = 0
                    for future in as_completed(future_to_pdf):
                        pdf_path = future_to_pdf[future]
                        pdf_filename = os.path.basename(pdf_path)
                        
                        try:
                            results = future.result()
                            all_results.extend(results)
                            
                            completed_pdfs += 1
                            progress_bar.progress(completed_pdfs / total_pdfs * 0.8)  # 80% for extraction
                            progress_text.markdown(f"📄 Processing images in document {completed_pdfs}/{total_pdfs}: {pdf_filename}")
                            
                            log_msg = f"📸 Found {len(results)} images in {pdf_filename}"
                            st.session_state.image_analysis_logs.append(log_msg)
                            st.markdown(log_msg)
                            
                        except Exception as e:
                            st.error(f"Failed to process images from {pdf_filename}: {str(e)}")
                            completed_pdfs += 1
                
                if not all_results:
                    st.warning("No images were found in the provided PDF files.")
                    st.session_state.image_analysis_complete = True
                    return
            
                # Initialize the image duplication detector with parallel processing
                with img_spinner_placeholder:
                    st.markdown('<div class="spinner-container"><div class="spinner spinner-analyzing"></div></div>', unsafe_allow_html=True)
                progress_text.markdown("📊 Setting up perceptual hash analysis (parallel processing)...")
                progress_bar.progress(0.85)
                
                # Initialize detector and load images
                detector = ImageDuplicationDetector()
                detector.load_images_parallel(all_results, max_workers=CONCURRENT_CONFIG['max_workers_image'])
                
                # Enhanced embedding progress display
                embedding_container = st.container()
                with embedding_container:
                    st.markdown("""
                    <div style="padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                        <h4 style="color: #ffffff; margin: 0; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">🔍 Analyzing Image Fingerprints (Parallel)</h4>
                        <p style="color: #ffffff; margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.95; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
                            Using parallel perceptual hashing to find duplicate and similar images...
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
                        embedding_status.markdown(f"🚀 **Getting AI ready to analyze images (parallel)...**")
                        embedding_details.info(f"Teaching AI to understand {total} images from your documents using multiple workers...")
                        
                    elif stage == "embedding_complete":
                        with emb_spinner_placeholder:
                            st.markdown('<div class="spinner-container"><div class="spinner spinner-analyzing"></div></div>', unsafe_allow_html=True)
                        embedding_progress.progress(0.8)
                        embedding_status.markdown(f"⚡ **Building parallel image comparison system...**")
                        embedding_details.success(f"✅ AI has learned to recognize all {total} images using parallel processing!")
                        
                    elif stage == "finding_duplicates":
                        with emb_spinner_placeholder:
                            st.markdown('<div class="spinner-container"><div class="spinner spinner-finalizing"></div></div>', unsafe_allow_html=True)
                        embedding_progress.progress(0.9)
                        embedding_status.markdown(f"🔍 **Looking for similar images (parallel)...**")
                        embedding_details.info("Comparing images to find potential matches using parallel computation...")
                        
                    elif stage == "complete":
                        emb_spinner_placeholder.empty()
                        embedding_progress.progress(1.0)
                        embedding_status.markdown(f"🎉 **Image analysis complete!**")
                        embedding_details.success("Successfully analyzed all images for similarities using parallel processing!")
                        
                    elif stage == "error":
                        emb_spinner_placeholder.empty()
                        embedding_progress.progress(0)
                        embedding_status.markdown(f"❌ **Something went wrong...**")
                        embedding_details.error("Could not analyze images. Please try again or check your files.")
                
                # Run the hash-based duplicate detection with parallel processing
                try:
                    # Manual progress updates
                    embedding_progress_callback("starting", 0, len(all_results), "Starting analysis")
                    
                    st.info("🔍 Analyzing image fingerprints for duplicates (parallel processing)...")
                    embedding_progress_callback("finding_duplicates", 0, 0, "Computing perceptual hashes")
                    
                    # Find duplicates using perceptual hashing with parallel processing
                    pairs_info = detector.detect_cross_pdf_duplicates_parallel(threshold=0.85, max_workers=CONCURRENT_CONFIG['max_workers_image'])
                    
                    embedding_progress_callback("complete", 0, 0, "Analysis complete")
                    
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
                                'orig_page': pair.get('orig_page', 0),
                                'dup_page': pair.get('dup_page', 0),
                                'orig_image_index': pair.get('orig_image_index', 0),
                                'dup_image_index': pair.get('dup_image_index', 0),
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
                                        caption=f"Document: {match['orig_pdf']} (Page {match['orig_page']})",
                                        use_container_width=True
                                    )
                                with col2:
                                    st.image(
                                        match['dup_filepath'],
                                        caption=f"Document: {match['dup_pdf']} (Page {match['dup_page']})",
                                        use_container_width=True
                                    )
                                
                                # Store for session state with page information
                                pair_info = f"**Match #{i}:** {match['orig_pdf']} (Page {match['orig_page']}) ↔ {match['dup_pdf']} (Page {match['dup_page']}) (Score: {match['score']:.2%})"
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

def run_image_analysis(directory):
    """Run image plagiarism analysis with proper session state management."""
    # Use the enhanced concurrent version by default
    return run_image_analysis_concurrent(directory)


def run_text_analysis_concurrent(directory, chunk_size=5000, similarity_threshold=0.3):
    """Run the complete plagiarism text analysis with enhanced concurrent processing."""
    import logging
    from src.core.validation import DirectoryValidator, ParameterValidator
    
    logger = logging.getLogger(__name__)
    
    # Check system resources and adjust configuration
    check_system_resources()
    
    # Validate inputs
    try:
        DirectoryValidator.validate_directory_path(directory, must_exist=True, must_be_readable=True)
        ParameterValidator.validate_positive_integer(chunk_size, "chunk_size", min_value=100, max_value=50000)
        ParameterValidator.validate_positive_float(similarity_threshold, "similarity_threshold", min_value=0.0, max_value=1.0)
    except Exception as e:
        st.error(f"❌ Invalid input: {str(e)}")
        logger.error(f"Input validation failed: {str(e)}")
        return None, None, None
    
    # Container to persist scanned/empty PDF notices outside progress UI
    skipped_info_container = st.container()
    progress_container = st.container()
    
    try:
        with progress_container:
            st.markdown("""
            <div class="progress-container">
                <span style='font-size:2rem;'>🚀</span> Analyzing Your Documents (Concurrent Mode)<br>
                <small>Using parallel processing for faster analysis...</small>
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
            
            # Step 2: Extract content with parallel processing
            with spinner_placeholder:
                st.markdown('<div class="spinner-container"><div class="spinner spinner-processing"></div></div>', unsafe_allow_html=True)
            status_text.info("📄 Reading document content (parallel processing)...")
            step_info.text("Step 2/5: Extracting text from your PDF files using multiple workers")
            
            # Use enhanced parallel PDF processing
            chunks = pdf_handler.extract_page_chunks_enhanced(chunk_size=chunk_size, max_workers=CONCURRENT_CONFIG['max_workers_pdf'])
            overall_progress.progress(40)
            
            # Step 3: Validate per-PDF content and generate embeddings with parallel processing
            with spinner_placeholder:
                st.markdown('<div class="spinner-container"><div class="spinner spinner-analyzing"></div></div>', unsafe_allow_html=True)
            status_text.info("🧠 Checking which PDFs contain readable text and preparing AI (parallel)...")
            step_info.text("Step 3/5: Validating text content and preparing embeddings with parallel processing")
            
            # Filter out PDFs that have no extractable text
            chunks_with_text = ContentValidator.filter_pdfs_with_text(chunks, min_total_chars=20)
            skipped_pdfs = [name for name in chunks.keys() if name not in chunks_with_text]

            # User-friendly feedback about skipped PDFs (persist during and after analysis)
            st.session_state.skipped_pdfs = skipped_pdfs
            if skipped_pdfs:
                with skipped_info_container:
                    st.markdown(
                        """
                        <div class="warning-card">
                            <h3>⚠️ Some PDFs were skipped</h3>
                            <p>These files appear to be scanned or contain no selectable text:</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    for name in skipped_pdfs:
                        st.markdown(f"- {name}")

            # Ensure at least 2 PDFs remain for comparison
            if len(chunks_with_text) < 2:
                spinner_placeholder.empty()
                st.error("❌ Not enough documents with readable text to compare.")
                st.info("You need at least 2 PDFs that contain selectable text. Consider OCR for scanned PDFs.")
                return None, None, None

            # Use enhanced parallel embedding generation
            embedder = get_cached_embedding_generator()
            embeddings = embedder.generate_embeddings_enhanced(
                chunks_with_text, 
                max_workers=CONCURRENT_CONFIG['max_workers_embedding'],
                batch_size=CONCURRENT_CONFIG['batch_size_embedding']
            )
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
                st.error(f"❌ Not enough documents produced embeddings. Found {len(valid_embeddings)} valid PDF(s).")
                st.info("Document processing summary:")
                for pdf_name in chunks.keys():
                    if pdf_name in skipped_pdfs:
                        st.error(f"❌ {pdf_name}: No readable text (likely scanned).")
                    elif pdf_name in valid_embeddings:
                        st.success(f"✅ {pdf_name}: Ready for comparison")
                    else:
                        st.warning(f"⚠️ {pdf_name}: Text found but embeddings could not be generated.")
                logger.error(f"Only {len(valid_embeddings)} PDFs have valid embeddings - cannot proceed")
                return None, None, None
            
            # Step 4: Compute similarities with parallel processing
            with spinner_placeholder:
                st.markdown('<div class="spinner-container"><div class="spinner spinner-analyzing"></div></div>', unsafe_allow_html=True)
            status_text.info("🔍 Comparing documents for similarities (parallel processing)...")
            step_info.text("Step 4/5: Finding matching content between documents using parallel computation")
            
            similarity_calculator = PDFSimilarityCalculator(valid_embeddings)
            semantic_scores = similarity_calculator.compute_all_pdf_similarities_parallel(max_workers=CONCURRENT_CONFIG['max_workers_similarity'])
            
            # Use only the PDFs with readable text for sequence similarity too
            chunks_for_sequence = {k: chunks_with_text[k] for k in valid_embeddings.keys()}
            seq_similarity_calculator = SequenceSimilarityCalculator(chunks_for_sequence)
            sequence_scores = seq_similarity_calculator.compute_all_pdf_similarities_parallel(max_workers=CONCURRENT_CONFIG['max_workers_similarity'])
            
            overall_progress.progress(80)
            
            # Step 5: Find exact matches
            with spinner_placeholder:
                st.markdown('<div class="spinner-container"><div class="spinner spinner-finalizing"></div></div>', unsafe_allow_html=True)
            status_text.info("🎯 Finding exact duplicate content...")
            step_info.text("Step 5/5: Identifying identical text passages")
            
            exact_match_detector = ExactMatchDetector()
            exact_matches = exact_match_detector.find_exact_matches(chunks_with_text)
            
            overall_progress.progress(100)
            spinner_placeholder.empty()
            progress_container.empty()
            
            st.markdown("""
            <div class="success-card">
                <h3>🎉 Analysis Complete!</h3>
                <p>Your documents have been analyzed using advanced AI techniques with parallel processing.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Store results in session state
            st.session_state.semantic_scores = semantic_scores
            st.session_state.sequence_scores = sequence_scores
            st.session_state.exact_matches = exact_matches
            st.session_state.chunks_with_text = chunks_with_text  # Store chunks for content display
            
            return semantic_scores, sequence_scores, exact_matches
            
    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Analysis failed: {str(e)}")
        logging.error(f"Analysis error: {str(e)}")
        return None, None, None

def run_text_analysis(directory, chunk_size=5000, similarity_threshold=0.3):
    """Run the complete plagiarism text analysis with progress tracking."""
    # Use the enhanced concurrent version by default
    return run_text_analysis_concurrent(directory, chunk_size, similarity_threshold)


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
        # Persistently show skipped PDFs info if any
        if 'skipped_pdfs' in st.session_state and st.session_state.skipped_pdfs:
            with st.container():
                st.markdown(
                    """
                    <div class="warning-card">
                        <h3>⚠️ Some PDFs were skipped</h3>
                        <p>These files appear to be scanned or contain no selectable text:</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for name in st.session_state.skipped_pdfs:
                    st.markdown(f"- {name}")

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
            
            # Add toggle for enhanced content display
            st.markdown("---")
            st.markdown("### 🔍 Detailed Content Analysis")
            
            # Create tabs for different types of analysis
            tab1, tab2, tab3 = st.tabs(["📊 Exact Matches", "🧠 Semantic Similarity", "📈 Overview"])
            
            with tab1:
                display_exact_matches(st.session_state.exact_matches)
            
            with tab2:
                # Get chunks data from session state if available
                chunks_data = getattr(st.session_state, 'chunks_with_text', {})
                if chunks_data:
                    display_semantic_similarity_with_content(
                        st.session_state.semantic_scores, 
                        chunks_data, 
                        threshold=0.7
                    )
                else:
                    st.info("📝 Semantic similarity content analysis requires chunks data. Run a fresh analysis to see detailed content.")
            
            with tab3:
                st.markdown("### 📊 Analysis Summary")
                st.markdown("""
                **Analysis Types:**
                - **📊 Exact Matches:** Identical text blocks found across documents
                - **🧠 Semantic Similarity:** Content with similar meaning and context
                - **📈 Overview:** Statistical summary and similarity matrices
                
                **Content Display Features:**
                - ✅ Shows actual matched text content
                - ✅ Expandable sections for long content
                - ✅ Character and word count statistics
                - ✅ Sample content from similar documents
                """)
        
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
                <p><strong>Your ZIP file has been uploaded successfully! Choose your analysis type:</strong></p>
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
                        <li>Fast hash-based comparison</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # No ZIP file uploaded
        st.markdown("""
        <div class="info-card">
            <h3>👋 Welcome to Smart Document Analyzer!</h3>
            <p><strong>Analysis Options:</strong></p>
            <ul>
                <li>📝 <strong>Text Analysis:</strong> AI content analysis, pattern matching, and duplicate detection</li>
                <li>🖼️ <strong>Image Analysis:</strong> Find duplicate images across your documents</li>
                <li>🎯 <strong>Complete Analysis:</strong> Run both text and image analysis together</li>
                <li>📈 <strong>Visual Reports:</strong> Easy-to-understand charts and summaries</li>
                <li>⚡ <strong>Fast & Reliable:</strong> Efficient algorithms for accurate results</li>
            </ul>
            <p>👈 <strong>Get started:</strong> Upload a ZIP file containing your PDF documents in the sidebar!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Sample Analysis Dashboard")
        st.info("Upload a ZIP file containing your PDF documents to see analysis results here!")
    
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