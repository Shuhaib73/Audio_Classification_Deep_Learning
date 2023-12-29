import pickle
from pathlib import Path

import streamlit_authenticator as stauth

usernames = ['Admin','Shuhaib']
names = ['name1','name2']
passwords = ['admin@@','shuhaib@@']


hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"

with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)

