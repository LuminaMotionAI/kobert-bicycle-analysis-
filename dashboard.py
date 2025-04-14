import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 페이지 설정
st.set_page_config(
    page_title="Bicycle Data Analysis Dashboard",
    page_icon="🚲",
    layout="wide"
)

# 데모 데이터 생성
def create_demo_data():
    np.random.seed(42)
    return pd.DataFrame({
        'region': ['Seoul', 'Gyeonggi', 'Busan'] * 10,
        'age': np.random.randint(20, 60, 30),
        'gender': ['Male', 'Female'] * 15,
        'rating': np.random.randint(1, 6, 30)
    })

def main():
    # 사이드바
    st.sidebar.title("Bicycle Data Analysis")
    st.sidebar.image("https://img.freepik.com/free-vector/flat-design-bicycle-silhouette_23-2149156381.jpg", width=200)
    
    menu = st.sidebar.radio(
        "Menu",
        ["Home", "Data Analysis"]
    )
    
    # 데모 데이터 생성
    data = create_demo_data()
    
    if menu == "Home":
        st.title("Bicycle Market Analysis Dashboard")
        st.markdown("""
        ## Welcome! 👋
        This dashboard provides analysis and visualization of bicycle-related data.
        
        ### Main Features
        - 📊 Data Overview
        - 📈 Data Visualization
        
        Please select a menu from the sidebar.
        """)
        
        st.info("Currently using demo data.")
        
    elif menu == "Data Analysis":
        st.title("Data Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Data Overview", "Regional Analysis", "Age/Gender Analysis"])
        
        with tab1:
            st.header("Data Overview")
            st.dataframe(data)
            st.write("### Basic Statistics")
            st.write(data.describe())
        
        with tab2:
            st.header("Regional Analysis")
            fig, ax = plt.subplots(figsize=(10, 6))
            region_counts = data['region'].value_counts()
            region_counts.plot(kind='bar')
            plt.title("Regional Distribution")
            plt.xlabel("Region")
            plt.ylabel("Count")
            st.pyplot(fig)
            
            st.write("### Average Rating by Region")
            region_ratings = data.groupby('region')['rating'].mean()
            st.write(region_ratings)
        
        with tab3:
            st.header("Age/Gender Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Age Distribution")
                fig, ax = plt.subplots()
                sns.histplot(data=data, x='age', bins=20)
                plt.title("Age Distribution")
                st.pyplot(fig)
            
            with col2:
                st.write("### Gender Distribution")
                fig, ax = plt.subplots()
                data['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title("Gender Distribution")
                st.pyplot(fig)
            
            st.write("### Average Rating by Gender")
            gender_ratings = data.groupby('gender')['rating'].mean()
            st.write(gender_ratings)

if __name__ == "__main__":
    main() 