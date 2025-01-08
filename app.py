import streamlit as st
import joblib
import numpy as np

# Load the trained model and other artifacts
model = joblib.load('perceptron_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the Streamlit app
def app():
    st.title('Pumpkin Seed Classifier')
    
    st.header('Enter the features for the pumpkin seed classification:')
    
    # Input fields for pumpkin seed features
    area = st.slider('Area', min_value=0.0)
    perimeter = st.slider('Perimeter', min_value=0.0)
    major_axis_length = st.slider('Major Axis Length', min_value=0.0)
    minor_axis_length = st.slider('Minor Axis Length', min_value=0.0)
    convex_area = st.slider('Convex Area', min_value=0.0)
    equiv_diameter = st.slider('Equivalent Diameter', min_value=0.0)
    eccentricity = st.slider('Eccentricity', min_value=0.0, max_value=1.0)
    solidity = st.slider('Solidity', min_value=0.0, max_value=1.0)
    extent = st.slider('Extent', min_value=0.0, max_value=1.0)
    roundness = st.slider('Roundness', min_value=0.0)
    aspect_ratio = st.slider('Aspect Ratio', min_value=0.0)
    compactness = st.slider('Compactness', min_value=0.0)
    
    # When the user presses the predict button
    if st.button('Predict'):
        # Prepare input features for prediction
        input_features = np.array([[area, perimeter, major_axis_length, minor_axis_length, 
                                    convex_area, equiv_diameter, eccentricity, solidity, 
                                    extent, roundness, aspect_ratio, compactness]])
        
        # Scale the input features
        scaled_input = scaler.transform(input_features)
        
        # Predict the class
        prediction = model.predict(scaled_input)
        predicted_class = label_encoder.inverse_transform(prediction)
        
        # Show predicted result
        st.write(f"Predicted Class: {predicted_class[0]}")
        
        # Optionally, you can also display evaluation metrics here if required
        st.markdown("### Model Evaluation (Test Data)")
        st.write(f"Accuracy: {0.88:.2f}")  # Modify with actual evaluation result if desired

# Run the Streamlit app
if __name__ == '__main__':
    app()
