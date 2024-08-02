import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")
print(response.text)



import mido
import pygame
import time

# Initialize pygame's mixer for MIDI playback
pygame.mixer.init()

# Create a new MIDI file and a track
midi = mido.MidiFile()
track = mido.MidiTrack()
midi.tracks.append(track)

# Define the notes for a C major scale
notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C, D, E, F, G, A, B, C

# Add notes to the track
for note in notes:
    track.append(mido.Message('note_on', note=note, velocity=64, time=480))
    track.append(mido.Message('note_off', note=note, velocity=64, time=480))

# Convert MIDI data to a playable format
midi_data = midi.save(file=None)

# Play the MIDI data
pygame.mixer.music.load(mido.MidiFile(file=midi_data))
pygame.mixer.music.play()

# Wait for the playback to finish
while pygame.mixer.music.get_busy():
    time.sleep(1)

print("Playback finished")

print("MIDI file saved as c_major_scale.mid")














# # Configure the API key
# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# # Define generation configuration
# generation_config = {
#     "temperature": 1,
#     "top_p": 0.95,
#     "top_k": 64,
#     "max_output_tokens": 8192,
#     "response_mime_type": "text/plain",
# }

# # Create the model
# model = genai.GenerativeModel(
#     model_name="gemini-1.5-flash",
#     generation_config=generation_config,
#     # safety_settings = Adjust safety settings
#     # See https://ai.google.dev/gemini-api/docs/safety-settings
# )

# # Start a chat session
# chat_session = model.start_chat(history=[])

# # Function to handle user input and generate a response
# def get_response(user_input=None, image_path=None):
#     if image_path:
#         # If an image is provided, process the image
#         with open(image_path, "rb") as image_file:
#             image_data = image_file.read()
        
#         # Send the image to the model and get a response
#         response = chat_session.send_message(image=image_data)
#     elif user_input:
#         # If text input is provided, process the text
#         response = chat_session.send_message(user_input)
#     else:
#         return "Please provide either text input or an image."

#     return response.text

# # Example usage
# user_input = "Describe the content of the image."
# image_path = "path/to/your/image.png"  # Set the image path if you want to send an image

# response = get_response(user_input=user_input, image_path=image_path)
# print(response)
