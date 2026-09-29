import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

st.title("Parking Space Detection")
st.write("Upload a parking image and click Predict.")

uploaded_file = st.file_uploader(
    "Upload Parking Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        results = model.predict(image, conf=0.25)

        result = results[0]

        output_image = result.plot()

        st.subheader("Prediction Result")

        st.image(
            output_image,
            channels="BGR",
            use_container_width=True
        )

        empty = 0
        occupied = 0

        if result.boxes is not None:

            for cls in result.boxes.cls.tolist():

                class_name = model.names[int(cls)]

                if class_name == "space-empty":
                    empty += 1

                elif class_name == "space-occupied":
                    occupied += 1

        st.subheader("Parking Summary")

        st.write("Empty Spaces:", empty)
        st.write("Occupied Spaces:", occupied)
        st.write("Total Detected:", empty + occupied) 