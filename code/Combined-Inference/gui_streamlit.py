import streamlit as st
import torch
import os
import glob
import sys
import tempfile
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Ensure we can import from the inference script
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from inference import TiledInferencePipeline
except ImportError:
    # Handle direct execution vs module execution cases
    sys.path.append(str(current_dir / 'code' / 'Combined-Inference'))
    from inference import TiledInferencePipeline

# --- Page Config ---
st.set_page_config(
    page_title="AI Image Enhancer | Streamlit",
    page_icon="🎬",
    layout="wide"
)

# --- Checkpoint Discovery ---
DEFAULT_STAGE1_SEARCH_PATH = "../Stage1-Smartphone-to-DSLR/checkpoints/**/*.pth"
DEFAULT_STAGE2_SEARCH_PATH = "../Stage2-Film-Enhancement/checkpoints/**/*.pth"

def find_checkpoints(search_pattern):
    """Find .pth files and prioritize best/latest models."""
    abs_pattern = str(current_dir / search_pattern)
    files = glob.glob(abs_pattern, recursive=True)
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.abspath(f) for f in files]

def get_best_checkpoint(files, keyword):
    """Find the best checkpoint for a specific model type."""
    matches = [f for f in files if keyword.lower() in f.lower()]
    if not matches:
        return None
    # Prioritize 'best' in name
    best = [f for f in matches if 'best' in os.path.basename(f).lower()]
    if best:
        return best[0]
    return matches[0]

# --- Sidebar / Settings ---
st.sidebar.title("🎬 Pipeline Settings")

with st.sidebar.expander("🤖 Model Selection", expanded=True):
    # Stage 1: Targeted 3-choice logic
    dped_folder_pattern = "../Stage1-Smartphone-to-DSLR/checkpoints/stage1_dped/*.pth"
    zrr_folder_pattern = "../Stage1-Smartphone-to-DSLR/checkpoints/stage1_ZRR/*.pth"
    prod_folder_pattern = "../Stage1-Smartphone-to-DSLR/checkpoints/stage1_production/*.pth"
    
    dped_files = find_checkpoints(dped_folder_pattern)
    zrr_files = find_checkpoints(zrr_folder_pattern)
    prod_files = find_checkpoints(prod_folder_pattern)
    
    # Get best model for each
    dped_cp = get_best_checkpoint(dped_files, "best") or (dped_files[0] if dped_files else None)
    zrr_cp = get_best_checkpoint(zrr_files, "best") or (zrr_files[0] if zrr_files else None)
    prod_cp = get_best_checkpoint(prod_files, "best") or (prod_files[0] if prod_files else None)
    
    # User requested exactly 3 choices
    s1_choice_map = {
        "DPED ResNet": dped_cp,
        "ZRR U-Net": zrr_cp,
        "Failed ZRR U-Net": prod_cp
    }
    
    stage1_choice = st.radio(
        "Stage 1: Smartphone → DSLR", 
        options=list(s1_choice_map.keys()),
        index=0,  # DPED is default
        help="Select the Stage 1 architecture. Radios are used to prevent accidental text editing/deletion."
    )
    stage1_path = s1_choice_map[stage1_choice]
    
    if not stage1_path:
        st.sidebar.warning(f"⚠️ No checkpoint found for {stage1_choice}. Training might be needed.")

    # Stage 2: Automatic selection (No dropdown as requested)
    stage2_options = find_checkpoints(DEFAULT_STAGE2_SEARCH_PATH)
    stage2_path = stage2_options[0] if stage2_options else None
    
    if stage2_path:
        enable_stage2 = st.checkbox("Enable Cinematic Film Look", value=True, help="Applies the Stage 2 FilmNet/LUT enhancement.")
        if not enable_stage2:
            stage2_path = None
    else:
        st.sidebar.warning("⚠️ Stage 2 model not found. Only Stage 1 will be applied.")
        stage2_path = None

# High-quality defaults (User requested to remove manual tiling/performance selection)
tile_size = 224
overlap = 168

# Hardware Info
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
hw_name = "Apple Silicon (MPS)" if device == 'mps' else ("NVIDIA GPU (CUDA)" if device == 'cuda' else "CPU")
st.sidebar.info(f"**Hardware:** {hw_name}")

# --- Pipeline Loading (Cached in Session State) ---
@st.cache_resource
def load_pipeline(s1, s2, t_size, ov):
    s1_val = s1 if s1 and s1 != "None" else None
    s2_val = s2 if s2 and s2 != "None" else None
    
    return TiledInferencePipeline(
        stage1_checkpoint=s1_val,
        stage2_checkpoint=s2_val,
        device=device,
        tile_size=int(t_size),
        overlap=int(ov)
    )

# --- Main UI ---
st.title("🎞️ AI Image Enhancer")
st.write("Transform your smartphone photos into professional cinematic masterpieces.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "heic", "dng"])

if uploaded_file is not None:
    # Initial setup
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(image, width="stretch")
        
        process_btn = st.button("✨ Enhance Image", type="primary", width="stretch")
    
    if process_btn:
        if stage1_path == "None" and stage2_path == "None":
            st.error("Please select at least one model checkpoint in the sidebar!")
        else:
            with st.spinner("🚀 Processing with Tiled Inference..."):
                try:
                    # Initialize/Get pipeline
                    pipeline = load_pipeline(stage1_path, stage2_path, tile_size, overlap)
                    
                    # Save to temp and process
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input:
                        image.save(tmp_input.name)
                        input_path = tmp_input.name
                    
                    results = pipeline.process(input_path, output_dir=None)
                    os.unlink(input_path)
                    
                    # Convert results
                    to_pil = lambda t: transforms.ToPILImage()(t.squeeze(0).cpu())
                    
                    img_input = results['input']
                    img_dslr = to_pil(results['dslr']) if 'dslr' in results else img_input
                    img_film = to_pil(results['film']) if 'film' in results else img_dslr
                    
                    with col2:
                        st.subheader("Enhanced Result")
                        st.image(img_film, width="stretch")
                        
                        # Comparison slider or just side-by-side? 
                        # Streamlit standard doesn't have a slider, so side-by-side is best.
                        
                        # Download button
                        # Convert PIL to bytes for download
                        import io
                        buf = io.BytesIO()
                        img_film.save(buf, format="JPEG", quality=95)
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="📥 Download Enhanced Image",
                            data=byte_im,
                            file_name=f"enhanced_{uploaded_file.name}",
                            mime="image/jpeg",
                            width="stretch"
                        )
                        
                    st.success("✅ Enhancement Complete!")
                    
                    # Technical summary
                    with st.expander("📝 Processing Details"):
                        st.write(f"**Resolution:** {image.size[0]}x{image.size[1]}")
                        st.write(f"**Tiling:** {tile_size}x{tile_size} (overlap {overlap})")
                        if 'dslr' in results: st.write("✅ Stage 1: Smartphone → DSLR Applied")
                        if 'film' in results: st.write("✅ Stage 2: DSLR → Film Applied")

                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)

else:
    st.info("💡 Upload a photo to begin the enhancement process.")
    
    # Show examples if available
    st.subheader("Sample Workflow")
    st.markdown("""
    1. **Upload**: Select any smartphone photo (JPG, PNG, HEIC).
    2. **Configure**: Choose between 'Standard' or 'DPED' models in the sidebar.
    3. **Enhance**: Watch as the AI applies high-resolution tiled processing.
    4. **Download**: Save your cinematic-quality results.
    """)
