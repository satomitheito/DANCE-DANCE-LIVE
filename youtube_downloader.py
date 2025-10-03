import argparse
import os
import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp is not installed. Please install it using:")
    print("pip install yt-dlp")
    sys.exit(1)


def download_video(url, output_dir="downloads", quality="best"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        'format': f'{quality}[ext=mp4]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'merge_output_format': 'mp4',
        'ffmpeg_location': '/opt/homebrew/bin/ffmpeg',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown')
            print(f"Downloading: {video_title}")
            
            ydl.download([url])
            print(f" Successfully downloaded: {video_title}")
            print(f" Saved to: {output_dir}")
            
    except Exception as e:
        print(f" Error downloading video: {str(e)}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube videos as MP4 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    
    parser.add_argument(
        'url',
        help='YouTube video URL to download'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='downloads',
        help='Output directory for downloaded videos (default: downloads)'
    )
    
    parser.add_argument(
        '--quality', '-q',
        default='best',
        choices=['best', 'worst', '720p', '480p', '360p', '240p'],
        help='Video quality preference (default: best)'
    )
    
    args = parser.parse_args()
    
    if not args.url.startswith(('https://www.youtube.com/', 'https://youtu.be/')):
        print(" Error: Please provide a valid YouTube URL")
        print("Examples:")
        print("  https://www.youtube.com/watch?v=VIDEO_ID")
        print("  https://youtu.be/VIDEO_ID")
        sys.exit(1)
    
    print("🎬 YouTube Video Downloader")
    print("=" * 40)
    print(f"URL: {args.url}")
    print(f"Output: {args.output}")
    print(f"Quality: {args.quality}")
    print("=" * 40)
    
    success = download_video(args.url, args.output, args.quality)
    if success:
        print("\n Download completed successfully!")
    else:
        print("\n Download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
