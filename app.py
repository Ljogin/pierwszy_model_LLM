import json
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

MODEL_NAME = 'welcome_survey_clustering_pipeline_v1.pkl'
DATA = 'welcome_survey_simple_v1.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'

@st.cache_data
def get_model():
    return joblib.load(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def get_all_participants(model):
    all_df = pd.read_csv(DATA, sep=';')
    all_df["Cluster"] = model.predict(all_df)
    return all_df

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])

model = get_model()
all_df = get_all_participants(model)
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = model.predict(person_df)[0]
predicted_cluster_data = cluster_names_and_descriptions[str(predicted_cluster_id)]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")
for column, title in [
    ("age", "Rozkład wieku w grupie"),
    ("edu_level", "Rozkład wykształcenia w grupie"),
    ("fav_animals", "Rozkład ulubionych zwierząt w grupie"),
    ("fav_place", "Rozkład ulubionych miejsc w grupie"),
    ("gender", "Rozkład płci w grupie"),
]:
    fig = px.histogram(same_cluster_df, x=column)
    fig.update_layout(title=title, xaxis_title=column, yaxis_title="Liczba osób")
    st.plotly_chart(fig)
