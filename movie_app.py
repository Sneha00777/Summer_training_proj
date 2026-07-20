import streamlit as st
import pickle
import numpy as np
st.set_page_config(page_title="Movie Matcher", layout="wide")

st.title("Movie Recommandations")
st.markdown("---")

with open('movie_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('genre_encoder.pkl', 'rb') as file:
    genre_encoder = pickle.load(file)

with open('star_encoder.pkl', 'rb') as file:
    star_encoder = pickle.load(file)

with open('period_encoder.pkl', 'rb') as file:
    period_encoder = pickle.load(file)

with open('remake_encoder.pkl', 'rb') as file:
    remake_encoder = pickle.load(file)

with st.sidebar:
    st.header("Movies characteristics")
    st.write("Select your movie features below:")

    genre = st.selectbox("Select Genre", genre_encoder.classes_)
    lead_star = st.selectbox("Select Lead Star", star_encoder.classes_)
    release_period = st.selectbox("Select Release Period", period_encoder.classes_)
    whether_remake = st.selectbox("Is it a Remake?", remake_encoder.classes_)

    st.markdown("---")
    predict_button = st.button("Find Matching Movies", use_container_width=True, type="primary")

st.subheader("Results Dashboard")

if predict_button:

    encoded_genre = genre_encoder.transform([genre])[0]
    encoded_star = star_encoder.transform([lead_star])[0]
    encoded_period = period_encoder.transform([release_period])[0]
    encoded_remake = remake_encoder.transform([whether_remake])[0]

    input_data = np.array([[encoded_genre, encoded_star, encoded_period, encoded_remake]])

    probabilities = model.predict_proba(input_data)[0]


    top_indices = np.argsort(probabilities)[::-1][:5]

    top_movies = model.classes_[top_indices]
    with st.container(border=True):
        st.write("### Top 5 Recommended Matches")
        st.markdown("Based on the selected filters, here are the most relevant movie choices:")

        for movie in top_movies:
            st.write(f"-**{movie}**")
