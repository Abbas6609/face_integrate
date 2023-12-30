import streamlit as st
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from PIL import Image
import numpy as np
import cv2

# Define functions to load images and perform face swapping
def load_image(image):
    image = Image.open(image).convert('RGB')  # Convert to RGB
    return np.array(image)

def swap_faces(faceSource, sourceFaceId, faceDestination, destFaceId, app, swapper):
    try:
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

    except Exception as e:
        st.error(f"An error occurred during face swapping: {e}")
        return None

# Main function to run the Streamlit app
def main():
    # Initialize the app only when the script is run directly (not on import)
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_model('inswapper_128.onnx', download=True, download_zip=True)

    st.title("Facial Integration App")
    st.markdown("Welcome to Facial Integration", unsafe_allow_html=True)

    source_image = st.file_uploader("Choose a Source Image", type=["png", "jpg", "jpeg"])
    source_face_id = st.number_input('Source Face Position', value=1, min_value=1)
    dest_image = st.file_uploader("Choose a Destination Image", type=["png", "jpg", "jpeg"])
    dest_face_id = st.number_input('Destination Face Position', value=1, min_value=1)

    if source_image is not None:
        source_image_array = load_image(source_image)
        st.image(source_image, caption='Uploaded Source Image', use_column_width=True)
    else:
        source_image_array = None

    if dest_image is not None:
        dest_image_array = load_image(dest_image)
        st.image(dest_image, caption='Uploaded Destination Image', use_column_width=True)
    else:
        dest_image_array = None

    if st.button('Integrate Faces'):
        if source_image_array is not None and dest_image_array is not None:
            result = swap_faces(source_image_array, source_face_id, dest_image_array, dest_face_id, app, swapper)
            if result is not None:
                st.image(result, caption='Face Swapped Image', use_column_width=True)

# Run the main function
if __name__ == "__main__":
    main()
