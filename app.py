import time
import sys
import streamlit as st
from transformers import pipeline, set_seed
import re
import torch # Make sure torch is installed if using PyTorch backend

# --- Configuration ---
MODEL_NAME = "distilgpt2" # Or "gpt2", "gpt2-medium" etc.
DEFAULT_CUISINES = ["Italian", "Mexican", "Indian", "Chinese", "Japanese", "French", "American", "Thai", "Greek", "Spanish", "Arabic"]

# --- Model Loading (Cached) ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_generator(model_name):
    print(f"[{time.strftime('%H:%M:%S')}] ENTERING load_generator for {model_name}", flush=True) # Print immediately
    st.info(f"Attempting to load model: {model_name}... This may take several minutes on first run.") # Show message in UI early
    try:
        device = 0 if torch.cuda.is_available() else -1
        print(f"[{time.strftime('%H:%M:%S')}] Using device: {'GPU' if device == 0 else 'CPU'}", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] Calling pipeline('text-generation', model='{model_name}', ...)", flush=True)

        # === THIS IS THE SLOW STEP ===
        generator = pipeline('text-generation', model=model_name, device=device)
        # =============================

        print(f"[{time.strftime('%H:%M:%S')}] SUCCESS: pipeline() call finished.", flush=True)
        st.success(f"Model {model_name} loaded successfully!") # Update UI message
        return generator
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR loading model {model_name}: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        st.error(f"CRITICAL ERROR loading model {model_name}: {e}")
        return None
    finally:
         print(f"[{time.strftime('%H:%M:%S')}] EXITING load_generator for {model_name}", flush=True) # See if it finishes


# --- Helper Function for Cleaning Output ---
# (Same as before)
def clean_generated_names(generated_text, prompt_ending):
    """
    Cleans the raw output from the model to extract potential restaurant names.
    """
    prompt_ending_index = generated_text.find(prompt_ending)
    if prompt_ending_index != -1:
        useful_text = generated_text[prompt_ending_index + len(prompt_ending):].strip()
    else:
        useful_text = generated_text # Fallback

    lines = useful_text.split('\n')
    names = []
    for line in lines:
        cleaned_line = re.sub(r'^\s*[\d\.\-\*]+\s*', '', line).strip()
        if cleaned_line and len(cleaned_line.split()) <= 6 and len(cleaned_line) > 2: # Allow slightly longer names
             if not any(kw in cleaned_line.lower() for kw in ["cuisine:", "generate:", "names:", "restaurant:", "example:", "input:", "output:", "prompt:", "here are"]):
                 if cleaned_line not in names:
                    # Basic check for sentence-like structure (optional, can be refined)
                    if '.' not in cleaned_line[-2:]: # Avoid lines ending like sentences
                        names.append(cleaned_line)
    return names


# --- Core Generation Function ---
# (Modified slightly for Streamlit feedback)
def generate_restaurant_names(generator_pipeline, cuisine, num_names_to_generate=5, max_length=70):
    """
    Generates restaurant names for a given cuisine using the loaded pipeline.
    """
    if not generator_pipeline:
        st.error("Generator pipeline not available.")
        return []

    prompt_ending = f"Here are {num_names_to_generate} creative names for a {cuisine} restaurant:"
    prompt = f"""
Task: Generate a list of creative and authentic restaurant names.
Cuisine Focus: {cuisine}
Considerations: Think about common ingredients (like pasta, tacos, spices), famous locations or cities (like Rome, Oaxaca, Mumbai), cultural terms (like 'trattoria', 'cantina', 'dhaba'), and words evoking positive feelings (like 'delicious', 'aroma', 'fiesta', 'amore').
Desired Output Format: A numbered list of {num_names_to_generate} unique names.

{prompt_ending}
1."""

    # Set a seed for reproducibility during a single session if desired
    # set_seed(42) # Comment out for more randomness each time

    try:
        # Calculate max_new_tokens instead of max_length for better control
        # Estimate prompt tokens (rough) and add desired output length
        # Note: A more precise tokenizer-based calculation is better but adds complexity
        prompt_token_estimate = len(prompt.split()) # Very rough estimate
        # max_len_param = prompt_token_estimate + max_length
        max_new_tokens_param = max_length # Generate up to this many *new* tokens

        raw_outputs = generator_pipeline(
            prompt,
            # max_length=max_len_param, # Use max_new_tokens instead if available/preferred
            max_new_tokens=max_new_tokens_param, # More direct control over output length
            num_return_sequences=num_names_to_generate, # Generate multiple attempts
            do_sample=True,
            temperature=0.85, # Slightly higher temp for more creativity
            top_k=50,
            pad_token_id=generator_pipeline.tokenizer.eos_token_id
        )
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return []

    all_generated_names = []
    # st.write("--- Raw Model Output (for debugging) ---") # Optional: uncomment for debug
    for i, output in enumerate(raw_outputs):
        generated_text = output['generated_text']
        # st.text(f"Sequence {i+1}:\n{generated_text}\n-----------------------") # Optional: uncomment for debug
        names_from_sequence = clean_generated_names(generated_text, prompt_ending)
        all_generated_names.extend(names_from_sequence)

    final_names = []
    for name in all_generated_names:
        if name not in final_names:
            final_names.append(name)

    return final_names[:num_names_to_generate]

# --- Streamlit App UI ---

st.set_page_config(layout="wide") # Use wide layout

# st.title("🍽️ Restaurant Name Generator")
# st.markdown("Powered by Hugging Face Transformers")

# # Load the generator pipeline (will be cached after first run)
# generator = load_generator(MODEL_NAME)
st.title("🍽️ Restaurant Name Generator")
st.markdown("Powered by Hugging Face Transformers")

print(f"[{time.strftime('%H:%M:%S')}] Script main body: Attempting to call load_generator...", flush=True)
generator = load_generator(MODEL_NAME)
print(f"[{time.strftime('%H:%M:%S')}] Script main body: Returned from load_generator. Result is None: {generator is None}", flush=True)

# Rest of your script... (ensure the check for generator is None remains)
if generator is None:

   st.error("Model could not be loaded. Check terminal for detailed errors.")
   st.stop() # Stop further execution if model failed
else:
   # ... your UI code using the generator ...
   st.sidebar.header("Generator Options")
   # ... etc ...

# --- Sidebar Controls ---
# st.sidebar.header("Generator Options")

selected_cuisine = st.sidebar.selectbox(
    "Pick a Cuisine:",
    options=DEFAULT_CUISINES,
    index=0 # Default to the first cuisine in the list
)

num_names = st.sidebar.number_input(
    "Number of names to generate:",
    min_value=1,
    max_value=10,
    value=5, # Default value
    step=1
)

max_len = st.sidebar.slider(
    "Max Generation Length (Tokens):",
    min_value=20,
    max_value=150,
    value=70, # Default value
    step=10,
    help="Controls the maximum number of tokens (roughly words/punctuation) the AI generates *after* the prompt. Longer values might yield more results but can also lead to less relevant text."
)

generate_button = st.sidebar.button("✨ Generate Names", type="primary", disabled=(generator is None))

# --- Main Area for Results ---

if generator is None:
    st.warning("Model could not be loaded. Please check the console for errors.")
else:
    if generate_button:
        with st.spinner(f"Thinking of {num_names} creative {selected_cuisine} restaurant names..."):
            generated_names = generate_restaurant_names(
                generator_pipeline=generator,
                cuisine=selected_cuisine,
                num_names_to_generate=num_names,
                max_length=max_len
            )

        st.subheader(f"Generated Names for a {selected_cuisine} Restaurant:")
        if generated_names:
            for i, name in enumerate(generated_names):
                # Using markdown for potential formatting later, looks cleaner
                st.markdown(f"**{i+1}. {name}**")
                # You could add more details here, like dummy menu items
                # st.caption(f" - Sample dish: {generate_dummy_dish(selected_cuisine)}") # Example
            st.success("Generation complete!")
        else:
            st.warning("Could not generate distinct names with the current settings. Try adjusting length or cuisine.")

    else:
        st.info("Select a cuisine and click 'Generate Names' in the sidebar to start.")

# Optional: Add footer or more info
st.sidebar.markdown("---")
st.sidebar.markdown(f"*Using model: `{MODEL_NAME}`*")