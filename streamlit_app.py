import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from typing import List, Dict
import base64
from test import DonutInvoiceParser, InvoiceData

# Page configuration
st.set_page_config(
    page_title="Invoice Parser",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #FF5252;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'parser' not in st.session_state:
        st.session_state.parser = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'results' not in st.session_state:
        st.session_state.results = []


def check_model_cache():
    """Check if the model is already downloaded/cached"""
    try:
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
        if cache_dir.exists():
            # Look for the specific model
            model_dirs = list(cache_dir.glob('*donut-base-finetuned-docvqa*'))
            return len(model_dirs) > 0
    except:
        pass
    return False


def load_parser_with_progress():
    """Load the Donut parser with detailed progress feedback"""
    if st.session_state.parser is None:
        # Create placeholder for progress updates
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        # Check if model is cached
        model_cached = check_model_cache()

        if not model_cached:
            # Model needs to be downloaded
            progress_placeholder.markdown(
                '<div class="warning-box">📥 <b>First-time setup:</b> Downloading AI model (~500MB). This may take 2-5 minutes depending on your internet connection...</div>',
                unsafe_allow_html=True
            )
            status_placeholder.info("⏳ Please wait... The model will be cached for future use.")
        else:
            # Model is cached
            progress_placeholder.markdown(
                '<div class="info-box">🔄 Loading AI model from cache...</div>',
                unsafe_allow_html=True
            )

        # Show progress bar
        progress_bar = st.progress(0)

        try:
            # Update progress - Starting
            progress_bar.progress(10)
            status_placeholder.text("Initializing model configuration...")

            # Load the parser
            progress_bar.progress(30)
            status_placeholder.text("Loading processor and model weights...")

            st.session_state.parser = DonutInvoiceParser()

            # Update progress - Complete
            progress_bar.progress(100)
            status_placeholder.text("✅ Model loaded successfully!")

            st.session_state.model_loaded = True

            # Clear progress indicators after a short delay
            import time
            time.sleep(1)
            progress_placeholder.empty()
            status_placeholder.empty()
            progress_bar.empty()

            # Show success message
            st.success("✅ AI Model loaded and ready!")

        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.empty()
            progress_bar.empty()
            st.error(f"❌ Error loading model: {str(e)}")
            st.info("💡 Try restarting the application or check your internet connection.")
            return None

    return st.session_state.parser


def save_uploaded_file(uploaded_file, temp_dir):
    """Save uploaded file to temporary directory"""
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def display_invoice_card(data: Dict, index: int = None):
    """Display invoice data in a nice card format"""
    extracted = data.get('extracted_data', {})

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### 📄")
        if index:
            st.markdown(f"**Invoice #{index}**")

    with col2:
        if 'pdf_file' in data:
            st.markdown(f"**📁 File:** `{data['pdf_file']}`")

        # Create columns for invoice details
        detail_cols = st.columns(2)

        with detail_cols[0]:
            st.markdown(f"**🏢 Party Name:**")
            party_name = extracted.get('party_name', 'Not Found')
            if party_name != "Not Found":
                st.success(party_name)
            else:
                st.error(party_name)

            st.markdown(f"**🔢 Invoice Number:**")
            inv_number = extracted.get('invoice_number', 'Not Found')
            if inv_number != "Not Found":
                st.success(inv_number)
            else:
                st.error(inv_number)

        with detail_cols[1]:
            st.markdown(f"**📅 Invoice Date:**")
            inv_date = extracted.get('invoice_date', 'Not Found')
            if inv_date != "Not Found":
                st.success(inv_date)
            else:
                st.error(inv_date)

            st.markdown(f"**💰 Invoice Amount:**")
            inv_amount = extracted.get('invoice_amount', 'Not Found')
            if inv_amount != "Not Found":
                st.success(inv_amount)
            else:
                st.error(inv_amount)

        # Success rate
        if 'success_rate' in data:
            success_rate = data['success_rate']
            fields_found = int(success_rate.split('/')[0])
            total_fields = int(success_rate.split('/')[1])
            percentage = (fields_found / total_fields) * 100

            st.progress(percentage / 100)
            st.markdown(f"**Success Rate:** {success_rate} fields ({percentage:.0f}%)")

    st.markdown("---")


def main():
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">🧾 Invoice Parser</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Extract invoice data using AI-powered OCR</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        processing_mode = st.radio(
            "Processing Mode",
            ["Single Invoice", "Batch Processing"],
            help="Choose how you want to process invoices"
        )

        st.markdown("---")



        st.header("ℹ️ Instructions")
        st.markdown("""
        1. **Select processing mode**
        2. **Upload PDF file(s)**
        3. **Click 'Process'** button
        4. **View results** and download Excel

        **Supported Fields:**
        - 🏢 Party Name
        - 🔢 Invoice Number
        - 📅 Invoice Date
        - 💰 Invoice Amount

        **Note:** First-time usage will download the AI model (~500MB).
        """)

    # Main content area
    if processing_mode == "Single Invoice":
        st.header("📄 Single Invoice Processing")

        uploaded_file = st.file_uploader(
            "Upload Invoice PDF",
            type=['pdf'],
            help="Select a single PDF invoice file"
        )

        if uploaded_file is not None:
            st.markdown(f'<div class="info-box">📁 Selected file: <b>{uploaded_file.name}</b></div>',
                        unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                process_button = st.button("🚀 Process Invoice", key="single_process")

            if process_button:
                # Load parser (with progress feedback)
                parser = load_parser_with_progress()

                if parser is None:
                    st.error("❌ Failed to load AI model. Please try again.")
                    return

                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded file
                    pdf_path = save_uploaded_file(uploaded_file, temp_dir)
                    output_excel = os.path.join(temp_dir, "invoice_data.xlsx")

                    # Process invoice
                    with st.spinner("🔄 Extracting data from invoice..."):
                        result = parser.process_invoice(pdf_path, output_excel)

                    if result:
                        st.markdown('<div class="success-box">✅ Invoice processed successfully!</div>',
                                    unsafe_allow_html=True)

                        # Display results
                        st.subheader("📊 Extracted Data")
                        display_invoice_card(result)

                        # Download button
                        st.subheader("💾 Download Results")
                        with open(output_excel, "rb") as f:
                            st.download_button(
                                label="📥 Download Excel File",
                                data=f.read(),
                                file_name="invoice_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    else:
                        st.markdown('<div class="error-box">❌ Failed to process invoice</div>',
                                    unsafe_allow_html=True)

    elif processing_mode == "Batch Processing":
        st.header("📚 Batch Invoice Processing")

        uploaded_files = st.file_uploader(
            "Upload Multiple Invoice PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select multiple PDF invoice files"
        )

        if uploaded_files:
            st.markdown(f'<div class="info-box">📁 Selected <b>{len(uploaded_files)}</b> file(s)</div>',
                        unsafe_allow_html=True)

            # Show file list
            with st.expander("📋 View file list"):
                for idx, file in enumerate(uploaded_files, 1):
                    st.write(f"{idx}. {file.name}")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                process_button = st.button("🚀 Process All Invoices", key="batch_process")

            if process_button:
                # Load parser (with progress feedback)
                parser = load_parser_with_progress()

                if parser is None:
                    st.error("❌ Failed to load AI model. Please try again.")
                    return

                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_excel = os.path.join(temp_dir, "all_invoices.xlsx")

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results = []
                    success_count = 0

                    # Process each file
                    for idx, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")

                        # Save uploaded file
                        pdf_path = save_uploaded_file(uploaded_file, temp_dir)

                        try:
                            result = parser.process_invoice(pdf_path, output_excel)
                            if result:
                                results.append(result)
                                fields_found = int(result['success_rate'].split('/')[0])
                                if fields_found >= 3:
                                    success_count += 1
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                        # Update progress
                        progress_bar.progress((idx + 1) / len(uploaded_files))

                    status_text.text("✅ Processing complete!")

                    # Display summary
                    st.markdown(
                        f'<div class="success-box">✅ Processed {len(results)}/{len(uploaded_files)} invoices successfully</div>',
                        unsafe_allow_html=True)

                    # Display results
                    st.subheader("📊 Extracted Data")
                    for idx, result in enumerate(results, 1):
                        display_invoice_card(result, idx)

                    # Download button
                    st.subheader("💾 Download Results")
                    with open(output_excel, "rb") as f:
                        st.download_button(
                            label=f"📥 Download Excel File ({len(results)} invoices)",
                            data=f.read(),
                            file_name="all_invoices.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    # Statistics
                    st.subheader("📈 Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Processed", len(results))
                    with col2:
                        st.metric("Successful", success_count)
                    with col3:
                        success_rate = (success_count / len(results) * 100) if results else 0
                        st.metric("Success Rate", f"{success_rate:.1f}%")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p style="font-size: 0.8rem;">🤖 AI-powered invoice extraction using Vision Transformer technology</p>
        <p style="font-size: 0.7rem;">Model:Document Understanding Transformer by Naver Clova</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()