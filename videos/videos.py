import yt_dlp
import os

# Create output directory
output_dir = './data_science_videos'
os.makedirs(output_dir, exist_ok=True)

# Playlist URL
playlist_url = 'https://www.youtube.com/watch?v=t2_Q2BRzeEE&list=PLGjplNEQ1it8-0CmoljS5yeV-GlKSUEt0'

# Download options
ydl_opts = {
    'format': 'best',  # Best quality
    'outtmpl': f'{output_dir}/%(playlist_index)s - %(title)s.%(ext)s',
    'writesubtitles': True,  # Download subtitles
    'writeautomaticsub': True,  # Download auto-generated subtitles
    'subtitleslangs': ['en'],
    'ignoreerrors': True,  # Continue on errors
    'no_warnings': False,
}

# Download the playlist
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    print("Starting download...")
    ydl.download([playlist_url])
    print("Download complete!")