import os
import google.generativeai as genai

# Configure the API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Define generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
)

# Start a chat session
chat_session = model.start_chat(history=[])

# Function to handle user input and generate a response
def get_response(user_input=None, image_path=None):
    if image_path:
        # If an image is provided, process the image
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Send the image to the model and get a response
        response = chat_session.send_message(image=image_data)
    elif user_input:
        # If text input is provided, process the text
        response = chat_session.send_message(user_input)
    else:
        return "Please provide either text input or an image."

    return response.text

# Example usage
user_input = "Describe the content of the image."
image_path = "path/to/your/image.png"  # Set the image path if you want to send an image

response = get_response(user_input=user_input, image_path=image_path)
print(response)
