@echo off
setlocal
set "BASE_DIR=%~dp0.."
set "FFMPEG_BINARY=%~dp0ffmpeg\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"
set "FFPROBE_BINARY=%~dp0ffmpeg\ffmpeg-8.1.1-essentials_build\bin\ffprobe.exe"
set "PATH=%~dp0ffmpeg\ffmpeg-8.1.1-essentials_build\bin;%PATH%"
cd /d "%BASE_DIR%"
if exist .\venv310\Scripts\python.exe (
  .\venv310\Scripts\python.exe -m streamlit run app.py
) else (
  echo Virtual environment not found at .\venv310\Scripts\python.exe
  echo Please create or activate your virtual environment first.
)
endlocal
