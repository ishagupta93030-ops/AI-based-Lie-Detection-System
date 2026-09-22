# Tools Directory

This folder contains helper files for local tool support.

## FFmpeg

A portable FFmpeg build is included under `tools/ffmpeg/ffmpeg-8.1.1-essentials_build/bin`.

The Streamlit app automatically detects the local FFmpeg binaries from the repository and configures `pydub`.

## Launching the App

To launch the Streamlit app with the included FFmpeg tools, run:

```powershell
tools\start_streamlit_with_ffmpeg.bat
```
