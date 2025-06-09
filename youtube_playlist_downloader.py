# youtube_playlist_downloader.py
#
# Purpose:
# This script downloads audio from one or more YouTube video URLs.
# It uses 'yt-dlp' to fetch video information and perform downloads,
# and 'ffmpeg' (required by yt-dlp) for audio extraction.
# The script can save audio files to a specified output directory
# and will skip downloading if an audio file for a given video ID
# already exists in that directory.
#
# Dependencies:
# - Python 3.x
# - yt-dlp: Must be installed and accessible in the system PATH.
#   (e.g., via `pip install yt-dlp`)
# - ffmpeg: Must be installed and accessible in the system PATH.
#   (e.g., via `sudo apt-get install ffmpeg` on Debian/Ubuntu)
#
# Usage Examples:
# 1. Download audio from a single URL to the default 'input_audio' directory:
#    python youtube_playlist_downloader.py https://www.youtube.com/watch?v=VIDEO_ID_1
#
# 2. Download audio from multiple URLs:
#    python youtube_playlist_downloader.py https://www.youtube.com/watch?v=VIDEO_ID_1 https://www.youtube.com/watch?v=VIDEO_ID_2
#
# 3. Specify a custom output directory:
#    python youtube_playlist_downloader.py https://www.youtube.com/watch?v=VIDEO_ID_1 -o my_downloaded_audio
#    OR
#    python youtube_playlist_downloader.py https://www.youtube.com/watch?v=VIDEO_ID_1 --output_dir my_downloaded_audio
#

import subprocess
import argparse # For command-line argument parsing
import os       # For directory creation (os.makedirs) and path joining (os.path.join)
import glob     # For checking if files already exist using wildcard patterns

# json module is imported but not currently used. Kept for potential future enhancements.
# import json

def get_video_id(video_url):
    """
    Gets the video ID from a YouTube URL using yt-dlp.
    This helps in naming files and checking for existing downloads.

    Args:
        video_url: The URL of the YouTube video.

    Returns:
        The video ID as a string, or None if an error occurs.
    """
    try:
        command = ["yt-dlp", "--print", "id", video_url]
        # Using utf-8 encoding for consistent output handling
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        video_id = process.stdout.strip()

        # Basic sanity check for the video ID. YouTube IDs are typically 11 characters.
        if not video_id or len(video_id) > 20 or ' ' in video_id:
            print(f"Warning: yt-dlp --print id returned an unusual value: '{video_id}' for URL: {video_url}")
            # Returning None as the ID is suspect or empty.
            return None
        return video_id
    except subprocess.CalledProcessError as e:
        print(f"Error getting video ID for {video_url}: {e.stderr.strip()}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while getting video ID for {video_url}: {e}")
        return None

def download_audio(video_url, output_dir): # output_dir is now a required parameter
    """
    Downloads the audio from a YouTube video using yt-dlp,
    skipping if a file with the same video ID already exists in the output_dir.

    Args:
        video_url: The URL of the YouTube video.
        output_dir: The directory to save the downloaded audio.
    """
    print(f"\nProcessing video: {video_url}")
    video_id = get_video_id(video_url)

    if not video_id:
        print(f"Could not determine video ID for {video_url}. Skipping download.")
        return

    # Ensure the output directory exists; create it if it doesn't.
    # exist_ok=True means os.makedirs won't raise an error if the directory already exists.
    os.makedirs(output_dir, exist_ok=True)

    # Check if a file with this video_id (any extension) already exists in the output directory.
    # yt-dlp video IDs are generally safe for direct use in file paths (alphanumeric, underscore, hyphen).
    existing_files_pattern = os.path.join(output_dir, video_id + ".*")
    found_files = glob.glob(existing_files_pattern)

    if found_files:
        # glob.glob returns a list. We typically expect at most one match for an audio file.
        print(f"Skipping {video_url}, file already exists: {found_files[0]}")
        return

    try:
        # Define the output template for yt-dlp.
        # %(id)s will be replaced by the video ID by yt-dlp.
        # %(ext)s will be replaced by the appropriate audio file extension.
        output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

        command = [
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "best", # Request best available audio format
            "--output", output_template, # Specify output directory and filename template
            video_url
        ]

        print(f"Attempting to download audio for: {video_url} (ID: {video_id}) into {output_dir}")
        # Using utf-8 encoding for consistent output handling
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')

        print(f"Successfully downloaded audio for: {video_url}")

        # yt-dlp often prints progress and other useful info to stderr.
        if process.stderr.strip():
            # Filter out common non-error messages if desired, or print all for verbosity.
            # For now, printing all non-empty stderr.
            print(f"yt-dlp messages (stderr):\n{process.stderr.strip()}")

        # stdout might contain info if specific --print options were used,
        # but usually not much when just extracting audio with -x.
        if process.stdout.strip() and not "[ExtractAudio]" in process.stdout:
             print(f"yt-dlp output (stdout):\n{process.stdout.strip()}")

    except subprocess.CalledProcessError as e:
        # This block catches errors specifically from the subprocess.run call (e.g., yt-dlp returning non-zero).
        print(f"Error downloading audio for {video_url}: {e}")
        print(f"Command: {' '.join(e.cmd)}") # Show the exact command that failed.
        print(f"Stderr: {e.stderr.strip()}")
        print(f"Stdout: {e.stdout.strip()}")
    except Exception as e:
        # Catch any other unexpected errors during the download process.
        print(f"An unexpected error occurred while trying to download {video_url}: {e}")

if __name__ == "__main__":
    # Setup command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Downloads audio from YouTube video URLs using yt-dlp. "\
                    "Checks for existing files to avoid re-downloads."
    )
    parser.add_argument(
        'video_urls',
        nargs='+', # Requires at least one URL, accepts multiple.
        help='One or more YouTube video URLs separated by spaces.'
    )
    parser.add_argument(
        '-o', '--output_dir',
        default="input_audio", # Default output directory if not specified.
        help='Directory to save the downloaded audio files. Defaults to "input_audio".'
    )

    args = parser.parse_args()

    # Note: A check for ffmpeg's presence could be added here for user convenience,
    # but yt-dlp will also report if ffmpeg is missing.

    print(f"Received {len(args.video_urls)} video URL(s) to process.")
    print(f"Audio files will be saved to: {os.path.abspath(args.output_dir)}")

    for url in args.video_urls:
        # Pass the user-specified (or default) output directory to the download function.
        download_audio(url, args.output_dir)

    print("\nAll processing finished.")
