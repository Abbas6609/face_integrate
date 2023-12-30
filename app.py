import streamlit as st
import insightface
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import cv2
import io

# wellcomingMessage = """
#     <h1>Face Swapping</h1>
#     <p>If you like this app, please take a look at my <a href="https://www.meetup.com/tech-web3-enthusiasts-united-insightful-conversations/" target="_blank">Meetup Group</a>! There will be more interesting apps and events soon.</p>
#     <p>Happy <span style="font-size:500%;color:red;">&hearts;</span> coding!</p>
# """


# Initialize the Face Analysis model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

# Function to process the uploaded image
def load_image(image):
    image = Image.open(image).convert('RGB')  # Convert to RGB
    return np.array(image)

# Function to swap faces
def swap_faces(faceSource, sourceFaceId, faceDestination, destFaceId):
    faces = app.get(faceSource)
    faces = sorted(faces, key=lambda x: x.bbox[0])
    if len(faces) < sourceFaceId or sourceFaceId < 1:
        st.error(f"Source image only contains {len(faces)} faces, but you requested face {sourceFaceId}")
        return None

    source_face = faces[sourceFaceId - 1]

    res_faces = app.get(faceDestination)
    res_faces = sorted(res_faces, key=lambda x: x.bbox[0])
    if len(res_faces) < destFaceId or destFaceId < 1:
        st.error(f"Destination image only contains {len(res_faces)} faces, but you requested face {destFaceId}")
        return None

    res_face = res_faces[destFaceId - 1]
    result = swapper.get(faceDestination, res_face, source_face, paste_back=True)
    return result

st.title("Facial Integrtaion App")
st.markdown("Wellcome to Facial Integration" , unsafe_allow_html=True)

# File uploader for source image
source_image = st.file_uploader("Choose a Source Image", type=["png", "jpg", "jpeg"])
if source_image is not None:
    # Display the source image
    st.image(source_image, caption='Uploaded Source Image', use_column_width=True)
    source_image = load_image(source_image)  # Convert the uploaded file to an image
source_face_id = st.number_input('Source Face Position', value=1, min_value=1)

# File uploader for destination image
dest_image = st.file_uploader("Choose a Destination Image", type=["png", "jpg", "jpeg"])
if dest_image is not None:
    # Display the destination image
    st.image(dest_image, caption='Uploaded Destination Image', use_column_width=True)
    dest_image = load_image(dest_image)  # Convert the uploaded file to an image
dest_face_id = st.number_input('Destination Face Position', value=1, min_value=1)

# Variable to store the output image
output_image = None

# Button to perform the face swap action
if st.button('Integrate Faces'):
    if source_image is not None and dest_image is not None:
        # Perform face swap and store the result in output_image
        result = swap_faces(source_image, source_face_id, dest_image, dest_face_id)
        if result is not None:
            output_image = result  # Store the result in output_image

# Check if there is an output image to display and download
if output_image is not None:
    # Display the output image
    st.image(output_image, caption='Face Swapped Image', use_column_width=True)

    # Convert the output image to a byte array for download
    buffer = io.BytesIO()
    Image.fromarray(output_image.astype('uint8')).save(buffer, format="JPEG")
    buffer.seek(0)

    # Create a download button
    st.download_button(
        label="Download Image",
        data=buffer,
        file_name="face_output.jpg",
        mime="image/jpeg"
    )



# with Streamlit .. 1 out put