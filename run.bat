@echo off
echo Activating Python 3.10 Virtual Environment and starting Streamlit...
call venv310\Scripts\activate.bat
streamlit run app.py
pause
