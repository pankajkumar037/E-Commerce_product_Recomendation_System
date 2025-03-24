import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Define functions for recommendations
def content_based_recommendations(train_data, item_name, top_n=10):
    if item_name not in train_data['Name'].values:
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    return train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]
    similar_users_indices = user_similarities.argsort()[::-1][1:]
    recommended_items = []

    for user_index in similar_users_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

    return train_data[train_data['ProdID'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']].head(10)

def hybrid_recommendations(train_data, target_user_id, item_name, top_n=10):
    content_based = content_based_recommendations(train_data, item_name, top_n)
    collaborative = collaborative_filtering_recommendations(train_data, target_user_id, top_n)
    return pd.concat([content_based, collaborative]).drop_duplicates().head(10)

# Streamlit App Configuration
st.set_page_config(layout='wide', page_title='E-commerce Recommendation System')
st.title('E-commerce Product Recommendation System')

# Load Data
cleaned_data = pd.read_csv('data/clean_data.csv')
trending_products = pd.read_csv('data/trending_products.csv')

random_image_urls = [
        "static/img_1.png",
        "static/img_2.png",
        "static/img_3.png",
        "static/img_4.png",
        "static/img_5.png",
        "static/img_6.png",
        "static/img_7.png",
        "static/img_8.png",
    ]

# Sidebar Navigation
st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'User History'])

# Home Page
if page == 'Home':
    st.header('Trending Products')
    col1, col2, col3 = st.columns(3)

    for i, row in trending_products.iterrows():
        image_path = random.choice(random_image_urls)
        with [col1, col2, col3][i % 3]:
            st.image(image_path, use_column_width=True)
            st.markdown(f"**{row['Name']}**")
            st.caption(f"Brand: {row['Brand']} | Rating: {row['Rating']} | Reviews: {row['ReviewCount']}")

    # Input Fields
    st.sidebar.subheader('Get Recommendations')
    user_id = st.sidebar.number_input('Enter User ID', min_value=1, step=1)

    # Dropdown for product names
    unique_products = cleaned_data['Name'].unique().tolist()
    item_name = st.sidebar.selectbox('Select Product Name', unique_products)

    if st.sidebar.button('Get Recommendations'):
        if item_name and user_id:
            recommendations = hybrid_recommendations(cleaned_data, user_id, item_name)
            if not recommendations.empty:
                st.header('Recommended Products')
                for i, row in recommendations.iterrows():
                    image_path = row['ImageURL'] if pd.notna(row['ImageURL']) else random.choice(random_image_urls)
                    st.image(image_path, use_column_width=True)
                    st.markdown(f"**{row['Name']}**")
                    st.caption(f"Brand: {row['Brand']} | Rating: {row['Rating']} | Reviews: {row['ReviewCount']}")

                # Save history
                with open('user_history.txt', 'a') as file:
                    file.write(f"User {user_id} - {item_name}\n")
            else:
                st.warning('No recommendations found. Please try another product.')

# User History Page
elif page == 'User History':
    st.header('User Recommendation History')
    try:
        with open('user_history.txt', 'r') as file:
            history = file.read()
            st.text_area('History', history, height=300)
    except FileNotFoundError:
        st.info('No history found.')
