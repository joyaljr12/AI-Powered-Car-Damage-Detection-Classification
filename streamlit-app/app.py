import streamlit as st
from model_helper import predict


st.title('Vechicle Damage Detection')

uploaded_file = st.file_uploader('upload the file', type=['jpg', 'png', 'dicom'])

if uploaded_file:
    image_path = 'temp_file.jpg' # write this bytes to a temporary file
    with open(image_path, 'wb') as f: #wb(w means write and b means binary)
        f.write(uploaded_file.getbuffer()) #getbuffer gives the bytes/binary of that uploaded file
        st.image(uploaded_file, caption='uploaded file', use_container_width=True)
        prediction = predict(image_path)
        st.info(f'predicted_class : {prediction}')

#streamlit run e:/final_project/streamlit-app/app.py