import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved RandomForestClassifier model
model = pickle.load(open('crop_recommendation.pkl', 'rb'))

# Function for making crop recommendations
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features).reshape(1, -1)
    return prediction[0]

# Crop dictionary for mapping crop numbers to crop names
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Streamlit app
def main():
    st.title('Crop Recommendation System')

    st.sidebar.header('Input Parameters')
    N = st.sidebar.number_input('N value', min_value=0.0, step=0.1)
    P = st.sidebar.number_input('P value', min_value=0.0, step=0.1)
    K = st.sidebar.number_input('K value', min_value=0.0, step=0.1)
    temperature = st.sidebar.number_input('Temperature', min_value=0.0, step=0.1)
    humidity = st.sidebar.number_input('Humidity', min_value=0.0, step=0.1)
    ph = st.sidebar.number_input('pH value', min_value=0.0, step=0.1)
    rainfall = st.sidebar.number_input('Rainfall', min_value=0.0, step=0.1)

    if st.sidebar.button('Recommend Crop'):
        prediction = recommendation(N, P, K, temperature, humidity, ph, rainfall)
        if prediction[0] in crop_dict:
            recommended_crop = crop_dict[prediction[0]]
            st.success(f"Recommended Crop: {recommended_crop}")
        else:
            st.warning("Sorry, we are not able to recommend a proper crop for this environment.")

if __name__ == '__main__':
    main()
