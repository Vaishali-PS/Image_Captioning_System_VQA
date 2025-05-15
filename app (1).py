import streamlit as st
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import time
import torch
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import pickle

FEATURE_OUTPUT_DIR = 'testing_features'
VQA_MODEL_NAME = "Salesforce/blip-vqa-base"


@st.cache_resource
def load_vqa_model_and_processor():
    print("Loading BLIP VQA model and processor...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processor = BlipProcessor.from_pretrained(VQA_MODEL_NAME)
        model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_NAME).to(device)

        print("BLIP VQA model and processor loaded.")
        return model, processor, device

    except Exception as e:
        st.error(f"Error loading BLIP VQA model: {e}")
        return None, None, None



def get_vqa_answer(vqa_model, vqa_processor, device, image_pil, question):
    """Gets answer to a question using the BLIP VQA model."""
    start_time = time.time()
    print("Getting VQA answer using BLIP...")
    try:
        inputs = vqa_processor(image_pil, question, return_tensors="pt").to(device)

        with torch.no_grad():
            out = vqa_model.generate(**inputs)

        predicted_answer = vqa_processor.decode(out[0], skip_special_tokens=True)

        print(f"BLIP VQA prediction took {time.time() - start_time:.2f} seconds.")
        return predicted_answer

    except Exception as e:
        st.error(f"Error getting BLIP VQA answer: {e}")
        return "Error during VQA prediction."


st.set_page_config(layout="wide", page_title="Visual AI Assistant", page_icon="🤖")

st.title("📷 Visual AI Assistant: Captioning & VQA 🤖")
st.markdown("""Upload an image and interact with it! Generate descriptive captions or ask questions about the visual content."""
            , unsafe_allow_html=True)

@st.cache_resource
def load_captioning_components():
    print("Loading Captioning components (VGG16, Model, Tokenizer)...")
    caption_model_path = 'caption_generator_model.h5'
    tokenizer_path = 'tokenizer.pkl'
    try:
        with open(tokenizer_path, 'rb') as f:
            caption_tokenizer = pickle.load(f)

        caption_model = load_model(caption_model_path)

        vgg_model = VGG16()
        vgg_feature_extractor = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

        caption_max_length = 35

        print("Captioning components loaded.")
        return vgg_feature_extractor, caption_model, caption_tokenizer, caption_max_length

    except FileNotFoundError as e:
        st.error(f"Error loading captioning assets: {e}. Make sure '{caption_model_path}' and '{tokenizer_path}' exist in the 'Temp' directory.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading captioning components: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def extract_features(image_pil, model):
    """Extract features from an image using the VGG16 model."""
    try:
        image = image_pil.resize((224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        features = model.predict(image, verbose=0)
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def convert_to_word(integer, tokenizer):
    """Convert an integer prediction back to a word."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image_features, tokenizer, max_length):
    """Generate a caption for the image features."""
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = convert_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split(' ')[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


vqa_model, vqa_processor, device = (None,) * 3
vgg_feature_extractor, caption_model, caption_tokenizer, caption_max_length = (None,) * 4

try:
    vqa_model, vqa_processor, device = load_vqa_model_and_processor()
except Exception as e:
    st.error(f"Failed to load VQA components: {e}")

try:
    vgg_feature_extractor, caption_model, caption_tokenizer, caption_max_length = load_captioning_components()
except Exception as e:
    st.error(f"Failed to load Captioning components: {e}")


models_loaded = all([
    vqa_model, vqa_processor, device,
    vgg_feature_extractor, caption_model, caption_tokenizer, caption_max_length
])


if not models_loaded:
    st.error("One or more essential models failed to load. Cannot proceed. Please check logs and file paths.")
else:
    st.success("✅ All models loaded successfully!")

    uploaded_file = st.file_uploader("🖼️ Choose an image...", type=["jpg", "jpeg", "png"])

    if 'caption' not in st.session_state:
        st.session_state.caption = None
    if 'vqa_question' not in st.session_state:
        st.session_state.vqa_question = ""
    if 'vqa_answer' not in st.session_state:
        st.session_state.vqa_answer = None

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([0.55, 0.45])

        with col1:
            st.image(image_pil, caption='Uploaded Image', use_column_width=True)

        with col2:
            st.subheader("✨ Actions & Results")
            st.divider()

            with st.container():
                st.markdown("**📝 Generate Image Caption**")
                if st.button("✨ Generate Caption"):
                    with st.spinner("✍️ Generating caption..."):
                        features = extract_features(image_pil, vgg_feature_extractor)

                        if features is not None:
                            st.session_state.caption = predict_caption(caption_model, features, caption_tokenizer, caption_max_length)
                            st.rerun()
                        else:
                            st.error("Could not generate caption due to feature extraction error.")

                if st.session_state.caption:
                    st.info(f"**Generated Caption:** {st.session_state.caption}")

            st.divider()

            with st.container():
                st.markdown("**❓ Ask a Question**")

                with st.form(key='vqa_form'):
                    question = st.text_input("Enter your question about the image:", key="vqa_question_input", placeholder="e.g., What color is the car?")
                    submitted = st.form_submit_button("💬 Get Answer")

                    if submitted and question:
                        with st.spinner("🤔 Thinking..."):
                            answer = get_vqa_answer(vqa_model, vqa_processor, device, image_pil, question)
                            st.session_state.vqa_question = question
                            st.session_state.vqa_answer = answer
                            st.rerun()
                    elif submitted and not question:
                        st.warning("⚠️ Please enter a question.")

                if st.session_state.vqa_answer:
                    st.success(f"**Q:** {st.session_state.vqa_question}")
                    st.success(f"**A:** {st.session_state.vqa_answer}")


        if FEATURE_OUTPUT_DIR and os.path.exists(FEATURE_OUTPUT_DIR):
             temp_image_path = os.path.join(FEATURE_OUTPUT_DIR, uploaded_file.name)
             if os.path.exists(temp_image_path):
                 try:
                     os.remove(temp_image_path)
                 except OSError as e:
                     print(f"Error removing temporary file {temp_image_path}: {e}")
        elif FEATURE_OUTPUT_DIR:
             print(f"Warning: Directory '{FEATURE_OUTPUT_DIR}' does not exist. Cannot remove temp file.")


    else:
        st.session_state.caption = None
        st.session_state.vqa_question = ""
        st.session_state.vqa_answer = None