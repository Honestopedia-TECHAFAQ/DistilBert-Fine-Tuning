import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import sqlite3

conn = sqlite3.connect('profiles.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS profiles
             (name TEXT, interests TEXT, professional_background TEXT, location TEXT, event_preferences TEXT)''')
def save_profile_to_db(profile):
    c.execute("INSERT INTO profiles VALUES (?, ?, ?, ?, ?)", (profile["Info"], profile["Interests"], profile["Professional Background"], profile["Location"], profile["Event Preferences"]))
    conn.commit()
def close_db_connection():
    conn.close()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def process_input(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad(): 
        outputs = model(input_ids)
    last_hidden_states = outputs.last_hidden_state
    features = last_hidden_states[:, 0, :].squeeze().tolist()
    return features

def generate_profile(input_text, name_column, profession, interests, location, event_preferences):
    features = process_input(input_text)
    profile = {
        "Info": name_column,
        "Interests": interests,
        "Professional Background": profession,
        "Location": location,
        "Event Preferences": event_preferences
    }
    save_profile_to_db(profile)
    return profile

def main():
    st.title("Profile Creation Model")
    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
    else:
        df = pd.DataFrame()
    
    input_text = st.text_area("Enter your description:")
    name_column = st.selectbox("Select the column containing names", df.columns.tolist(), key='name_column')

    selected_columns = st.multiselect("Select columns to display info", df.columns.tolist())

    if st.button("Generate Profile"):
        if input_text:
            if name_column:
                new_profile = generate_profile(input_text, name_column, "", "", "", "")
                df = df.append(new_profile, ignore_index=True)
                if uploaded_file.name.endswith('.csv'):
                    df.to_csv('profiles.csv', index=False)
                elif uploaded_file.name.endswith('.xlsx'):
                    df.to_excel('profiles.xlsx', index=False)
                st.success("Profile generated and saved successfully.")
            else:
                st.warning("Please select a column containing names.")
        else:
            st.warning("Please enter a description.")

    for idx, row in df.iterrows():
        st.write(f"## Profile {idx+1}")
        description = row["Description"]
        if name_column:
            for column in selected_columns:
                st.write(f"### {column} Info:")
                info = row[column] if column in df.columns else "Not Available"
                st.write(info)
            st.write("---")
    close_db_connection()

if __name__ == "__main__":
    main()
