import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import io
from PIL import Image,ImageEnhance,ImageFilter
from PIL import ImageOps
import easyocr
import numpy as np
import pandas as pd
import cv2 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams 
from sklearn.naive_bayes import MultinomialNB
import spacy
from spacy import displacy
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Download spaCy model
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer # Term Frequency 
from sklearn.metrics import classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, accuracy_score,recall_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import pickle
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split



st.header('Final Project')


with st.sidebar:

    st.sidebar.image("data.jpeg",use_column_width=True)
    selected = option_menu("Task",
                      ["Image Processing","NLP Detailing","EDA", "Prediction","Evaluation Metrics","Customer Recomendation"],
                      menu_icon="cast",
                      styles={
                          "container": {"padding":"4!important", "background-color":"lightgreen"},
                          "icon":{"color":"#01A982","font-size":"20px"},
                          "nav-link": {"font-size": "20px", "text-align":"left"},
                          "nav-link-selected": {"background-color": "blue"}
                      }
                      )


if selected == "Image Processing":
    
    with st.container(border=True):     
        with st.container():
            
            st.markdown("<h4 style='text-align:center;color:#000000;'> Upload the Image </h4>",unsafe_allow_html=True)
            upload_image=st.file_uploader("",type=["jpg","PNG","jpeg"])
            col1,col2=st.columns([1,1])
            with col1:    
                if upload_image is not None:
                    image_bytes = io.BytesIO(upload_image.read())
                    image1 = Image.open(image_bytes)
                    st.image(image1, width=200)
                    st.write("Image format:", image1.format,"-------", "Image size:", image1.size," ","Image mode:", image1.mode)
                    st.markdown("<h4 style='text-align:center;color:#000000;'> Uploaded Image </h4>",unsafe_allow_html=True)
                   
                        
            with col2:
                st.markdown("<h4 style='text-align:center;color:#000000;'>Text Detection on Image</h4>",unsafe_allow_html=True)
                on=st.toggle("Image to text Conversion")
                if on:
                    image_array = np.asarray(image1)
                    if image1.mode != 'RGB':
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                    
                    reader=easyocr.Reader(["en"])
                    extracted_text=reader.readtext(gray,detail=0)
                    st.write("Extracted_text :",extracted_text) 
            
    with st.container(border=True):
    
        st.markdown("<h4 style='text-align:center;color:#949494;'>Basic Resizing and Cropping</h4>",unsafe_allow_html=True)
        cl1,cl2,cl3=st.columns([1,1,1])
        with cl1:
            tog=st.toggle("Cropping")
            if tog:
                col1,col2=st.columns([1,1])
                with col1:
                    sd=image1.crop((0,0,800,600))
                    sd1=st.button("Standard Photo size 4:3")
                    if sd1:
                        st.image(sd,width=250)
                with col2:
                    sd=image1.crop((0,0,1920,1080))
                    sd1=st.button("HD Photo size 16:9")
                    if sd1:
                        st.image(sd,width=250)

        with cl2:
            on=st.toggle("Resize")
            if on:
                height=st.number_input("Height",min_value=1,max_value=2580,step=100)
                width=st.number_input("Width",min_value=1,max_value=2580,step=100)
                resize=image1.resize((width,height))
                st.image(resize)
                
        with cl3:
            on=st.toggle("Rotation")
            if on:
                flip_lr=st.button("Flip left right")
                if flip_lr:
                    flip_lr1=image1.transpose(Image.FLIP_LEFT_RIGHT)
                    st.image(flip_lr1)
                flip_TB=st.button("Flip Top Bottom")
                if flip_TB:
                    flip_lr1=image1.transpose(Image.FLIP_TOP_BOTTOM)
                    st.image(flip_lr1)
                flip_TP=st.button("Transpose")
                if flip_TP:
                    flip_lr1=image1.transpose(Image.TRANSPOSE)
                    st.image(flip_lr1)
                flip_TV=st.button("Transverse")
                if flip_TV:
                    flip_lr1=image1.transpose(Image.TRANSVERSE)
                    st.image(flip_lr1)
    
    with st.container(border=True):
         
        st.markdown("<h4 style='text-align:center;color:#949494;'>Basic Manipulation</h4>",unsafe_allow_html=True)
        cl2,cl3=st.columns([1,1])
        
                    
        with cl2:
            on=st.toggle("Change Image Format")
            if on:
                g=st.button("Grayscale")
                if g:
                    gray=image1.convert("L")
                    st.write("Image Formart : ",image1.format)
                    st.image(gray)
                r=st.button("Monochrome")
                if r:
                    rgba=image1.convert("1")
                    st.write("Image Format :",image1.format)
                    st.image(rgba)
                c=st.button("CMYK")
                if c:
                    cmyk=image1.convert("CMYK")
                    st.write("Image Format : ",image1.format)
                    st.image(cmyk)
        with cl3:
            on=st.toggle("Color convertion and Equalizer")
            if on:
                eq=st.button("Equalizer")
                if eq:
                    e=ImageOps.equalize(image1)
                    st.image(e)
                iv=st.button("Invert")
                if iv:
                    i=ImageOps.invert(image1)
                    st.image(i)
                mi=st.button("Mirror")
                if mi:
                    mir=ImageOps.mirror(image1)
                    st.image(mir)
    with st.container(border=True):
        st.markdown("<h4 style='text-align:center;color:#949494;'>Basic Enhancement and Filter</h4>",unsafe_allow_html=True)
        cl01,cl02=st.columns([1,1])
        with cl01:
            on=st.toggle("Image Enhancement")
            if on:
                bi=st.button("Enhance Brightness")
                #fl=st.number_input("Enter the value",1.0,20.0,step=1.0)
                if bi:
                    bi=ImageEnhance.Brightness(image1)
                    en1=bi.enhance(2.0)
                    st.image(en1)
            if on:
                bi=st.button("Enhance Color")
                #fl=st.number_input("Enter the value",1.0,20.0,step=1.0)
                if bi:
                    bi=ImageEnhance.Color(image1)
                    en1=bi.enhance(2.0)
                    st.image(en1)
            if on:
                bi=st.button("Enhance Contrast")
                #fl=st.number_input("Enter the value",1.0,20.0,step=1.0)
                if bi:
                    bi=ImageEnhance.Contrast(image1)
                    en1=bi.enhance(2.0)
                    st.image(en1)
        with cl02:
            on=st.toggle("Image Filter")
            if on:
                bi=st.button("Image blur")
                if bi:
                    dt=image1.filter(ImageFilter.BLUR)
                    st.image(dt)
                bi=st.button("Image Boxblur")
                if bi:
                    dt=image1.filter(ImageFilter.BoxBlur(2.0))
                    st.image(dt)
                bi=st.button("Image Edge Enhance")
                if bi:
                    dt=image1.filter(ImageFilter.EDGE_ENHANCE)
                    st.image(dt)
                bi=st.button("Image Details")
                if bi:
                    dt=image1.filter(ImageFilter.DETAIL)
                    st.image(dt)
                bi=st.button("Image Find Edges")
                if bi:
                    dt=image1.filter(ImageFilter.FIND_EDGES)
                    st.image(dt)
                bi=st.button("Image Smooth")
                if bi:
                    dt=image1.filter(ImageFilter.SMOOTH)
                    st.image(dt)
        
       
if selected == "NLP Detailing":

        df = pd.read_csv("test.csv")

        X_train=df["text"]
        y_train=df["label"]

        # Streamlit app
        st.title("NLP Processing with Streamlit")

        # Text input
        text_input = st.text_area("Enter text for NLP processing:")

        # Tokenization
        if st.checkbox("Tokenization"):
            tokens = word_tokenize(text_input)
            st.write("Tokens:", tokens)

        # Stopword Removal
        if st.checkbox("Stopword Removal"):
            stop_words = set(stopwords.words("english"))
            filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
            st.write("Tokens after stopword removal:", filtered_tokens)

        # Number Removal
        if st.checkbox("Number Removal"):
            filtered_tokens = [word for word in filtered_tokens if not word.isdigit()]
            st.write("Tokens after number removal:", filtered_tokens)

        # Special Character Removal
        if st.checkbox("Special Character Removal"):
            filtered_tokens = [word for word in filtered_tokens if word.isalnum()]
            st.write("Tokens after special character removal:", filtered_tokens)

        # Stemming
        if st.checkbox("Stemming"):
            porter_stemmer = PorterStemmer()
            stemmed_tokens = [porter_stemmer.stem(word) for word in filtered_tokens]
            st.write("Tokens after stemming:", stemmed_tokens)

        # Lemmatization
        if st.checkbox("Lemmatization"):
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
            st.write("Tokens after lemmatization:", lemmatized_tokens)

        # Parts of Speech (POS)
        if st.checkbox("Parts of Speech (POS)"):
            doc = nlp(text_input)
            pos_tags = [(token.text, token.pos_) for token in doc]
            st.write("Parts of Speech:", pos_tags)

        # N-gram
        if st.checkbox("N-gram"):
            n = st.slider("Select N for N-gram", min_value=2, max_value=5, value=2, step=1)
            ngram_vectorizer = CountVectorizer(ngram_range=(n, n))
            X_ngram = ngram_vectorizer.fit_transform([text_input])
            st.write(f"{n}-gram representation:", X_ngram.toarray())

        # Text Classification
        if st.checkbox("Text Classification"):
        # Create a pipeline with CountVectorizer and MultinomialNB

            model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
            ])

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Make predictions on the testing data
            y_pred = model.predict(X_train)

            # Display evaluation metrics
            accuracy = accuracy_score(y_train, y_pred)
            class_report_df = pd.DataFrame(classification_report(y_train, y_pred, output_dict=True))
            st.write("Classification Report:")
            st.write(f"Accuracy: {accuracy:.2f}")
            st.dataframe(class_report_df)


        if st.checkbox("Sentiment Analysis"):
            #Assuming binary sentiment classification (positive and negative)
            sentiment = "Positive" if model.predict([text_input])[0] == "positive" else "Negative"
            st.write(f"Sentiment: {sentiment}")

        # Word Cloud
        if st.checkbox("Word Cloud"):

            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_input)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.axis("off")
            st.pyplot()

        # Keyword Extraction
        if st.checkbox("Keyword Extraction"):
            keywords = nlp(text_input).ents
            st.write("Keywords:", [keyword.text for keyword in keywords])

        # Named Entity Recognition (NER)
        if st.checkbox("Named Entity Recognition (NER)"):
            doc_ner = nlp(text_input)
            ner_displacy = displacy.render(doc_ner, style="ent", page=True)
            st.write(ner_displacy, unsafe_allow_html=True)
            
if selected == "Customer Recomendation":
    
    def load_data():
        """Loads data from a CSV file"""
        df = pd.read_csv('Mareket_data.csv')
        return df


    def create_surprise_dataset(data):
        """Creates a Surprise dataset from the loaded data"""
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(data[['CustomeId', 'Products', 'Rating']], reader)
        return dataset


    def build_collaborative_filtering_model(dataset):
        """Builds and trains a collaborative filtering model"""
        trainset, testset = train_test_split(dataset, test_size=0.2)
        model = SVD()
        model.fit(trainset)
        return model, testset


    def recommend_products(model, data, customer_id, num_recommendations):
        """Recommends products for a specific customer"""
        all_products = data['ProductsID'].unique()
        purchased_products = data[data['CustomeId'] == customer_id]['ProductsID'].values
        remaining_products = [p for p in all_products if p not in purchased_products]

        predictions = [(p, model.predict(customer_id, p).est) for p in remaining_products]
        top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:num_recommendations]

        recommended_product_names = [data[data['ProductsID'] == pid]['Products'].values[0]
                                    for pid, _ in top_predictions]
        return recommended_product_names


    def main():
        st.title('Customer Recommendation App with Surprise')

        data = load_data()

        st.write('### Original Data (Sample)')
        st.write(data.head())  # Display a sample of the data

        dataset = create_surprise_dataset(data)
        model, _ = build_collaborative_filtering_model(dataset)

        customer_id = st.selectbox('Select a CustomerId for recommendations:', data['CustomeId'].unique())
        num_recommendations = st.number_input('Number of Recommendations:', min_value=1, max_value=10, value=2)

        if st.button("Recommend Products"):
            recommendations = recommend_products(model, data.copy(), customer_id, num_recommendations)
            st.subheader(f'Top {num_recommendations} Product Recommendations for CustomerId {customer_id}:')
            st.write(recommendations)


    if __name__ == '__main__':
        main()


   
if selected == "EDA":
    # Step 1: Load CSV File
        class SessionState:
            def __init__(self, **kwargs):
                for key, val in kwargs.items():
                    setattr(self, key, val)

# Create an instance of SessionState
        session_state = SessionState(df=None)
        
        # Step 1: Load CSV File
        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            session_state.df = df  # Save the data in the session state
        
        # Step 2: Display DataFrame
        if session_state.df is not None and st.button("Show DataFrame"):
            st.dataframe(session_state.df)
      
        if session_state.df is not None:
            st.write("### DataFrame")
            st.dataframe(session_state.df)
        
        # Drop Duplicates and NaN Values
        if session_state.df is not None:
            st.write("### Drop Duplicates and NaN Values")
            session_state.df = session_state.df.drop_duplicates()
            session_state.df = session_state.df.dropna()
            st.dataframe(session_state.df)
            st.success("Duplicates and NaN values dropped successfully!")
        
        if session_state.df is not None:
            st.write("### DataFrame Info")
            st.text(session_state.df.info())
            
        if session_state.df is not None:
            st.write("### Summary Statistics")
            st.text(session_state.df.describe())

        
        # Label Encoding
        if session_state.df is not None:
            st.write("### Label Encoding")
            le = LabelEncoder()
            for col in session_state.df.columns:
                if session_state.df[col].dtype == 'object' or session_state.df[col].dtype == 'bool':
                    session_state.df[col] = le.fit_transform(session_state.df[col])
            st.dataframe(session_state.df)
            st.success("Label Encoding completed successfully!")
        
        # One-Hot Encoding for categorical columns
        if session_state.df is not None:
            st.write("### One-Hot Encoding")
            categorical_columns = session_state.df.select_dtypes(include=['object']).columns
            session_state.df = pd.get_dummies(session_state.df, columns=categorical_columns)
            st.dataframe(session_state.df)
            st.success("One-Hot Encoding completed successfully!")
        
        # DateTime Format Conversion
        if session_state.df is not None:
            st.write("### DateTime Format Conversion")
            session_state.df['target_date'] = pd.to_datetime(session_state.df['target_date'])
            st.dataframe(session_state.df)
            st.success("DateTime Format Conversion completed successfully!")
        
        # Plot Relationship Curve
        if session_state.df is not None:
            st.write("### Plot Relationship Curve")
            sampled_df = pd.DataFrame(session_state.df["avg_visit_time"].sample(min(1000, len(session_state.df))))
            sns.pairplot(sampled_df)
            st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)
        
        # Detect and Treat Outliers
        if session_state.df is not None:
            st.write("### Detect and Treat Outliers")
            Q1 = session_state.df['transactionRevenue'].quantile(0.25)
            Q3 = session_state.df['transactionRevenue'].quantile(0.75)
            IQR = Q3 - Q1
            session_state.df = session_state.df[~((session_state.df['transactionRevenue'] < (Q1 - 1.5 * IQR)) | (session_state.df['transactionRevenue'] > (Q3 + 1.5 * IQR)))]
            st.dataframe(session_state.df)
            st.success("Outliers detected and treated successfully!")
        
        # Plot Normalization Curve
        if session_state.df is not None:
            st.write("### Plot Normalization Curve")
            sns.histplot(session_state.df['avg_session_time'], kde=True)
            st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)
        
        # Treat Skewness
        if session_state.df is not None:
            st.write("### Treat Skewness")
            session_state.df['latest_visit_number'] = np.log1p(session_state.df['latest_visit_number'])
            st.dataframe(session_state.df)
            st.success("Skewness treated successfully!")
        
        # Calculate Correlation and Plot Heatmap
        if session_state.df is not None:
            st.write("### Correlation Heatmap")
            correlation_matrix = session_state.df.corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
            st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)
                
        
        # Box Plot for Outlier Detection
        if session_state.df is not None:
            st.write("### Box Plot for Outlier Detection")
            numeric_columns = session_state.df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                sns.boxplot(x=col, data=session_state.df)
                st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)
        
        # scatter plot matric 
        if session_state.df is not None:
            st.write("### Scatter Plot Matrix")
            sns.set(style="ticks", rc={"figure.autolayout": False})
            sampled_df = pd.DataFrame(session_state.df.sample(min(1000, len(session_state.df))))
            
            progress_bar = st.progress(0)
            for i in range(len(sampled_df.columns)):
                sns.pairplot(sampled_df, vars=[sampled_df.columns[i]], diag_kind='hist')
                progress_bar.progress((i + 1) / len(sampled_df.columns))
        
            st.pyplot()
            # Disable the warning about PyplotGlobalUse
            st.set_option('deprecation.showPyplotGlobalUse', False)


if selected=="Prediction":
    c1,c2,c3,c4 = st.columns(4)
    st.markdown("""
    <style>
        .st-ax {
                background-color: lightblue;
        }

        .stTextInput input{
                background-color: lightblue;
        }
                
        .stNumberInput input{
                background-color: lightblue;
        }
        .stDateInput input{
                background-color: lightblue;
        }
                
    </style>
    """,unsafe_allow_html=True)
    with open("model_rf.pkl", "rb") as mf:
        new_model = pickle.load(mf)

    # Input form
    with st.form("user_inputs"):
        with st.container():
            count_session = st.number_input("count_session")
            time_earliest_visit = st.number_input("time_earliest_visit")
            avg_visit_time = st.number_input("avg_visit_time")
            days_since_last_visit = st.number_input("days_since_last_visit")
            days_since_first_visit = st.number_input("days_since_first_visit")
            visits_per_day = st.number_input("visits_per_day")
            bounce_rate = st.number_input("bounce_rate")
            earliest_source = st.number_input("earliest_source")
            latest_source = st.number_input("latest_source")
            earliest_medium = st.number_input("earliest_medium")
            latest_medium = st.number_input("latest_medium")
            earliest_keyword = st.number_input("earliest_keyword")
            latest_keyword = st.number_input("latest_keyword")
            earliest_isTrueDirect = st.number_input("earliest_isTrueDirect")
            latest_isTrueDirect = st.number_input("latest_isTrueDirect")
            num_interactions = st.number_input("num_interactions")
            bounces = st.number_input("bounces")
            time_on_site = st.number_input("time_on_site")
            time_latest_visit = st.number_input("time_latest_visit")
            
        submit_button = st.form_submit_button(label="Submit")
    
    # Predict using the model
    if submit_button:
        test_data = np.array([
            [
                count_session, time_earliest_visit, avg_visit_time, days_since_last_visit, 
                days_since_first_visit, visits_per_day, bounce_rate, earliest_source, 
                latest_source, earliest_medium, latest_medium, earliest_keyword, 
                latest_keyword, earliest_isTrueDirect, latest_isTrueDirect, num_interactions, 
                bounces, time_on_site, time_latest_visit
            ]
        ])
            
        # Convert the data to float
        test_data = test_data.astype(float)


        # Make predictions using the loaded model
        predicted = new_model.predict(test_data)[0]
        prediction_proba = new_model.predict_proba(test_data)
        st.write("Model Succusfully Processed")
        # Display the prediction results on the Streamlit app
        #st.write("Prediction:", predicted)
        #st.write("Prediction Probability:", prediction_proba)
    
        if predicted == 0 :
            st.write("Not Converted ")
        else:
            st.write(" Converted")
        

if selected=="Evaluation Metrics":

    # Step 1: Load CSV File
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # EDA and Preprocessing Steps

        # Duplicate Removal
        st.write("### Duplicate Removal")
        df1 = df.drop_duplicates()
        st.success("Duplicates removed successfully!")

        # NaN Value Fill
        st.write("### NaN Value Fill")
        df2 = df1.fillna(0)  # You can replace 0 with the desired value
        st.success("NaN values filled successfully!")

        # DateTime Format Conversion
        st.write("### DateTime Format Conversion")
        date_columns = df2.select_dtypes(include=['datetime']).columns
        for col in date_columns:
            df2[col] = pd.to_datetime(df2[col])
        st.success("DateTime Format Conversion completed successfully!")

        # Display DataFrame
        st.dataframe(df2)

        # Summary Statistics
        st.write("### Summary Statistics")
        st.write(df2.describe())

        # Feature Importance with Random Forest
        le = LabelEncoder()
        for col in df2.columns:
            if df2[col].dtype == 'object' or df2[col].dtype == 'bool':
                df2[col] = le.fit_transform(df2[col])

        X_train = df2.drop('has_converted',axis=1)
        y_train = df2['has_converted']

        # Plot feature importance
        st.write("### Feature Importance with Random Forest")
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        feature_importances = rf.feature_importances_
        
        feature_importance_df=pd.DataFrame({
            "Feature":X_train.columns,
            "Impotance":feature_importances
            })
        top_10_features=feature_importance_df.sort_values(by="Impotance",ascending=False).head(10)["Feature"].tolist()
        extra_feature="has_converted"
        df3 = df2[top_10_features + [extra_feature]]
        #columns=['count_session','time_earliest_visit','avg_visit_time','days_since_last_visit','days_since_first_visit','visits_per_day','bounce_rate','earliest_source','latest_source','earliest_medium','has_converted']
        
        # Streamlit code
        st.title('Top 10 Features Importance')
        st.bar_chart(top_10_features)
        
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Pie Chart using Feature Importance
        st.write("### Pie Chart using Feature Importance")
        fig, ax = plt.subplots()
        ax.pie(feature_importances, labels=feature_importances, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
        
            
        # Random Forest Model Build
        model = RandomForestClassifier(n_estimators=50,random_state=42)
        
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        rf_predict=rf_model.predict(X_train)
        rf_accuracy = accuracy_score(y_train,rf_predict)
        rf_Precision=precision_score(y_train,rf_predict)
        rf_recall=recall_score(y_train,rf_predict)
        rf_f1=f1_score(y_train,rf_predict)
        
        # Display Random Forest Model results
        st.write("# Random Forest Model")
        st.write("Accuracy:", rf_accuracy)
        st.write("Precision:", rf_Precision)
        st.write("Recall:", rf_recall)
        st.write("F1_score:", rf_f1)
        
        # Decision Tree Model Build
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        dt_predict=dt_model.predict(X_train)
        dt_accuracy = accuracy_score(y_train,dt_predict)
        dt_Precision=precision_score(y_train,dt_predict)
        dt_recall=recall_score(y_train,dt_predict)
        dt_f1=f1_score(y_train,dt_predict)
        
        # Display Decision Tree Model results
        st.write("# Decision Tree Model")
        st.write("Accuracy:", dt_accuracy)
        st.write("Precision:", dt_Precision)
        st.write("Recall:", dt_recall)
        st.write("F1_score:", dt_f1)

        
        # KNN Model Build
        knn_model = KNeighborsClassifier()
        knn_model.fit(X_train, y_train)  # Use the X_train, y_train from the first block
        knn_predict=knn_model.predict(X_train)
        knn_accuracy = accuracy_score(y_train, knn_predict)
        knn_Precision=precision_score(y_train,knn_predict)
        knn_recall=recall_score(y_train,knn_predict)
        knn_f1=f1_score(y_train,knn_predict)
        
        
        # Display KNN Model results
        st.write("# KNN Model")
        st.write("Accuracy:", knn_accuracy)
        st.write("Precision:",knn_Precision)
        st.write("Recall:", knn_recall)
        st.write("F1_score:",knn_f1)

    # Display results in a table
        results_data = {
            'Model': ['Random Forest', 'Decision Tree', 'KNN'],
            'Accuracy': [rf_accuracy, dt_accuracy, knn_accuracy],
            'Precision': [rf_Precision, dt_Precision, knn_Precision],
            'Recall': [rf_recall, dt_recall, knn_recall],
            'F1_score': [rf_f1, dt_f1, knn_f1]
        }
        
        results_table = st.table(results_data)
            

        # Plotly Visualization
        fig = px.bar(
            x=['Random Forest', 'Decision Tree', 'KNN'],
            y=[rf_accuracy, dt_accuracy, knn_accuracy],
            labels={'y': 'Accuracy', 'x': 'Models'},
            title='Model Accuracy Comparison'
        )

        st.plotly_chart(fig)
