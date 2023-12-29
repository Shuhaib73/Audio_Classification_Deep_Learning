
import pickle
from pathlib import Path
import streamlit as st 
import streamlit_authenticator as stauth 
from PIL import Image 
import pandas as pd 
import numpy as np 
import time as t 
import base64
import webbrowser
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt 
import seaborn as sns 
import librosa
from sklearn.preprocessing import LabelEncoder
import io 
import resampy
from keras.models import load_model


def get_background(filename):

    # # Set the background image format
    main_bg_ext = "png"
        
    # Main page Background
    with open(filename, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Set the background using CSS
    return st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{encoded_image});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def web_customes():
    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """

    return st.markdown(hide_st_style, unsafe_allow_html=True)

@st.cache_resource
def loading_msg():
    success_message = st.success("Logging in")
    t.sleep(0.2)
    success_message.empty()


def app():

    st.set_page_config(page_title="Audio_Classification", page_icon=":Info", layout="centered")

    @st.cache_data
    def get_data():
        df = pd.read_csv("merged_df.csv", index_col=None)

        return df

    df = get_data()
    
    web_customes()

    get_background("robo.png")

    #--------------------------------------------------
    st.markdown("<h2 style='text-align: center; color: #6c757d; font-size: 1px;'>Welcome Back!</h2>", unsafe_allow_html=True)

    # User-Authentication

    usernames = ['Admin','Shuhaib']
    names = ['name1','name2']
    
    file_path = Path(__file__).parent / "hashed_pw.pkl"
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)


    credentials = {"usernames":{}}
            
    for uname, name, pwd in zip(usernames, names, hashed_passwords):
        user_dict = {"name": name, "password": pwd}
        credentials["usernames"].update({uname: user_dict})

    
    authenticator = stauth.Authenticate(credentials, "Audio_Classification", "random_key", cookie_expiry_days=0)
    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status == None:
        st.markdown("<h3 style='text-align: center; color: #6c757d; font-size: 16px;'>Please Enter the Username and Password</h3>", unsafe_allow_html=True)

    if authentication_status == False:
        st.error("Username / Password is incorrect")
    elif authentication_status == True:
        loading_msg()
    

    if authentication_status:

        selected = option_menu(
            menu_title=None,
            options=["Model", "Info", "Insights", "Contact"],
            icons=["house", "book", "bookmark-star","envelope"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )

        if selected == "Model":

            get_background("robo1.png")
            
            # Description
            options = st.selectbox(
                'Would you like to know more about this project?',
                (' ', 'Description', 'Take me to the GitHub repository'))
            
            if options == 'Description':
                with open('Description.html', 'r', encoding='utf-8') as f:
                    data = f.read()
                    st.markdown(f'<div style="background-color:black;">{data}</div>', unsafe_allow_html=True)

            elif options == 'Take me to the GitHub repository':
                webbrowser.open_new_tab('https://github.com/TrainerArshdeep/SoundClassification')

            # --------------------------------------------------------------------

            # Audio file uploading
            with st.form(key='form', clear_on_submit=False):
                
                Audio_f = st.file_uploader("Upload Audio", type=["wav"])

                submit = st.form_submit_button("Predict")

                # Load the keras model
                models = load_model('FModel.h5')

                class_labels = ['dog_bark', 'children_playing', 'car_horn', 'air_conditioner','street_music', 'gun_shot', 'siren', 'engine_idling', 'jackhammer','drilling']
                
                #  Initialize and fit the LabelEncoder
                encoder = LabelEncoder()
                encoder.fit(class_labels)

                if submit:

                    if Audio_f is not None:
                        st.audio(Audio_f)
                        # Get the audio file content as bytes
                        filename = Audio_f.read()
                        
                        audio, sample_rate = librosa.load(io.BytesIO(filename), res_type='kaiser_fast') 
                        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)  # Change n_mfcc to 128
                        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

                        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
                        
                        predicted_probabilities = models.predict(mfccs_scaled_features)

                        rounded_probabilities = np.round(predicted_probabilities, decimals=15)
                        prediction_class = encoder.inverse_transform(np.argmax(rounded_probabilities, axis=1))

                        prediction_class = ','.join(prediction_class)
                        
                        lighter_background_color = "#e0e4ff"  # Adjust the color code as needed

                        st.markdown(f'<div style="background-color: {lighter_background_color}; padding: 10px; text-align: center; font-weight: bold; color: #1d3557; border-radius: 10px;">The Audio is predicted to be: {prediction_class}</div>', unsafe_allow_html=True)
                    
                    else:
                        st.warning("Please upload the Audio file")
        
            # ----------------------------------------------------------

            authenticator.logout("Logout", "main")


        if selected == "Info":

            get_background("tech_bg4.png")
            
            # st.info("Table")
            if 'number_of_rows' not in st.session_state:
                st.session_state['number_of_rows'] = 5

            # Input for the number of rows
            increment = st.text_input('Specify the number of rows to be displayed')
            if increment:
                increment = int(increment)
                st.session_state['number_of_rows'] = increment

            # Input for target classes in the sidebar
            
            st.sidebar.markdown("<h2 style='text-align: center; color: #ff758f; font-size: 18px;'>Please Filter Here:</h2>", unsafe_allow_html=True)

            target_class = st.sidebar.multiselect(
                "Select the Class: ",
                options=df['class'].unique(),
                default=['car_horn', 'siren', 'gun_shot', 'dog_bark',]
            )

            # Apply filters to the DataFrame
            filtered_df = df[df['class'].isin(target_class)].head(st.session_state['number_of_rows'])

            # Display the filtered DataFrame
            st.dataframe(filtered_df)

        if selected == "Insights":

            get_background("tech_bg4.png")
            
            st.markdown("<h1 style='text-align: left; font-size: 20px;'>📊 Dashboard</h1>", unsafe_allow_html=True)

            st.sidebar.markdown("<h2 style='text-align: center; color: #ff758f; font-size: 18px;'>Please Filter Here:</h2>", unsafe_allow_html=True)

            t_class = st.sidebar.multiselect(
                "Select the Class: ",
                options=df['class'].unique(),
                default=['car_horn', 'siren', 'gun_shot', 'dog_bark',]
            )
            
            # Visuals
            fl_df = df[df['class'].isin(t_class)]
            v1 = fl_df['class'].value_counts().reset_index().sort_values(by='class', ascending=False)

            v1.rename(columns={'class':'Sound', 'count':'Frequency'}, inplace=True)

            # Visual 1
            fig1, ax1 = plt.subplots(facecolor='none')
            ax1.set_facecolor('none')
            sns.barplot(
                x='Frequency', 
                y='Sound', 
                data=v1,
                ax=ax1,
                color='#FF6B7A'
                )
            plt.title("Frequency Distribution", fontdict={'fontsize':12, 'fontweight': 'bold', 'color': '#ffffff'})
            ax1.set_xlabel("Frequency", fontdict={'fontsize': 12, 'fontweight': 'bold', 'color': '#f5f3f4'})
            ax1.set_ylabel("Sound", fontdict={'fontsize': 12, 'fontweight': 'bold', 'color': '#f5f3f4'})

            cc = '#1DC6F6'
            ax1.tick_params(axis='x', colors=cc)
            ax1.tick_params(axis='y', colors=cc)
            
            ax1.set_xticklabels(ax1.get_xticklabels(), fontweight='bold')
            ax1.set_yticklabels(ax1.get_yticklabels(), fontweight='bold')

            plt.tight_layout()
            sns.despine()
            
            # Visual 2

            fig2, ax2 = plt.subplots(facecolor='none')
            ax2.set_facecolor('none')
            sns.barplot(
                y='Frequency', 
                x='Sound', 
                data=v1,
                ax=ax2,
                color='#5465ff'
                )
            plt.title("Frequency Distribution", fontdict={'fontsize':12, 'fontweight': 'bold', 'color': '#ffffff'})
            ax2.set_xlabel("Frequency", fontdict={'fontsize': 12, 'fontweight': 'bold', 'color': '#f5f3f4'})
            ax2.set_ylabel("Sound", fontdict={'fontsize': 12, 'fontweight': 'bold', 'color': '#f5f3f4'})

            cc = '#1DC6F6'
            ax2.tick_params(axis='x', colors=cc)
            ax2.tick_params(axis='y', colors=cc)
            
            ax2.set_xticklabels(ax2.get_xticklabels(), fontweight='bold')
            ax2.set_yticklabels(ax2.get_yticklabels(), fontweight='bold')

            plt.tight_layout()
            sns.despine()

            # Separation
            left_column, right_column = st.columns(2)
            left_column.pyplot(fig1, use_container_width=True)
            right_column.pyplot(fig2, use_container_width=True)

            # Visual 3

            cls_name = df['class'].unique()
            explode = [0.03 for cls in cls_name]

            colors = ['#5465ff','#5465ff','#5465ff','#5465ff','#42a5f5','#42a5f5','#42a5f5','#ffd6ff','#ffd6ff','#ffd6ff']
        
            fig3, ax3 = plt.subplots(facecolor='none')
            ax3.set_facecolor('none')
            pie_result = plt.pie(
                df['class'].value_counts(),
                labels=cls_name,
                explode=explode,
                autopct='%1.2f%%',
                colors=colors,
                textprops={'color': 'white', 'weight': 'bold'}
            )
            
            plt.title("Frequency Distribution", fontdict={'fontsize':12, 'fontweight': 'bold', 'color': '#ffffff'}) 
            
            for text in pie_result[1]:
                text.set_color('#1DC6F6')
                text.set_fontweight('bold')

            plt.tight_layout()
            sns.despine()


            left_column.pyplot(fig3, use_container_width=True)
            
            # v1_plt = px.bar(
            #     v1,
            #     x="Frequency",  # Assuming 'class' is in the index after reset_index
            #     y="Sound",
            #     title="<b>Frequency Distribution</b>",
            #     color_discrete_sequence=["#48cae4"],
            #     template="plotly_white",
            # )

            # v1_plt.update_layout(
            #     plot_bgcolor="rgba(0,0,0,0)",
            #     paper_bgcolor="rgba(0,0,0,0)",
            #     xaxis=dict(showgrid=False),  # Updated to use 'dict' instead of '('
            # )
            # st.plotly_chart(v1_plt)


        if selected == "Contact":

            get_background("robo1.png")
            
            st.markdown("<h2 style='text-align: center; color: #ced4da; font-size: 22px;'><span style='margin-right: 10px;'>📬</span>Get In Touch with us!</h2>", unsafe_allow_html=True)

            contact_form = """
            <form action="https://formsubmit.co/bursins77@gmail.com" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Name" required>
                <input type="email" name="email" placeholder="Email" pattern="[a-zA-Z]+@[a-zA-Z]+\.[a-zA-Z]{2,}" required>
                <textarea name="Message" placeholder="Your message here"></textarea>
                <button type="Submit">Send</button>
            </form>            
            """

            st.markdown(contact_form, unsafe_allow_html=True)

            #Use css style
            def Style_css(file_name):
                with open(file_name) as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

            Style_css("Style_css.css")

def main():
    app()


if __name__  == '__main__':
    main()


