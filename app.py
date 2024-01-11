import google.generativeai as genai
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
import os

# load environment variables
load_dotenv()

# configure API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

generation_config = {
    "temperature": 0.8,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096
}

safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
     for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

# initialize model
model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# function to read image
def read_image_data(filepath):
    imagepath = Path(filepath)
    if not imagepath.exists():
        raise FileNotFoundError(f"Could not find image: {imagepath}")
    return {"mime_type": "image/jpeg", "data": imagepath.read_bytes()}

# function to generate response
def generate_gemini_response(prompt, imagepath):
    image_data = read_image_data(imagepath)
    response = model.generate_content([prompt, image_data])
    return response.text

# Initial input prompt for the plant pathologist
input_prompt = """
As a highly skilled plant pathologist, your expertise is indispensable in our pursuit of maintaining optimal plant health. You will be provided with information or samples related to plant diseases, and your role involves conducting a detailed analysis to identify the specific issues, propose solutions, and offer recommendations.

**Analysis Guidelines:**

1. **Disease Identification:** Examine the provided information or samples to identify and characterize plant diseases accurately.

2. **Detailed Findings:** Provide in-depth findings on the nature and extent of the identified plant diseases, including affected plant parts, symptoms, and potential causes.

3. **Next Steps:** Outline the recommended course of action for managing and controlling the identified plant diseases. This may involve treatment options, preventive measures, or further investigations.

4. **Recommendations:** Offer informed recommendations for maintaining plant health, preventing disease spread, and optimizing overall plant well-being.

5. **Important Note:** As a plant pathologist, your insights are vital for informed decision-making in agriculture and plant management. Your response should be thorough, concise, and focused on plant health.

**Disclaimer:**
*"Please note that the information provided is based on plant pathology analysis and should not replace professional agricultural advice. Consult with qualified agricultural experts before implementing any strategies or treatments."*

Your role is pivotal in ensuring the health and productivity of plants. Proceed to analyze the provided information or samples, adhering to the structured 
"""

def process_uploaded_files(files):
    filepath = files[0].name if files else None
    response = generate_gemini_response(input_prompt, filepath) if filepath else None
    return filepath, response

with gr.Blocks() as demo:
    file_output = gr.Textbox()
    image_output = gr.Image()
    combined_output = [image_output, file_output]

    upload_button = gr.UploadButton(
        "click to Upload an Image",
        file_types=["Image"],
        file_count = "multiple"
    )

    upload_button.upload(process_uploaded_files, upload_button, combined_output)

demo.launch(debug=True)