#!/usr/bin/env python3
"""
Video streaming script
Supports reading multiple video files and streaming to RTMP server with infinite loop playback
"""

import argparse
import subprocess
import sys
import os
import time
from typing import List, Optional


def check_ffmpeg():
    """Check if ffmpeg is installed"""
    result = subprocess.run(['ffmpeg', '-version'], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL, 
                          check=False)
    return result.returncode == 0


def get_video_info(video_path: str) -> dict:
    """Get video information"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = result.stdout.strip().split('\n')
    
    width = int(lines[0]) if len(lines) > 0 and lines[0] else None
    height = int(lines[1]) if len(lines) > 1 and lines[1] else None
    fps_str = lines[2] if len(lines) > 2 and lines[2] else None
    duration = float(lines[3]) if len(lines) > 3 and lines[3] else None
    
    # Parse fps
    fps = None
    if fps_str and '/' in fps_str:
        parts = fps_str.split('/')
        if len(parts) == 2:
            num, den = int(parts[0]), int(parts[1])
            fps = num / den if den > 0 else None
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'duration': duration
    }


def stream_video(video_path: str, rtmp_url: str, loop: bool = False, 
                 video_codec: str = 'libx264', audio_codec: str = 'aac',
                 preset: str = 'veryfast', bitrate: str = '2000k',
                 fps: Optional[float] = None, resolution: Optional[str] = None):
    """
    Stream a single video to RTMP server
    
    Args:
        video_path: Video file path
        rtmp_url: RTMP streaming URL
        loop: Whether to loop playback
        video_codec: Video codec
        audio_codec: Audio codec
        preset: Encoding preset
        bitrate: Bitrate
        fps: Output frame rate (None means use original video frame rate)
        resolution: Output resolution, format like '1920x1080' (None means use original video resolution)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"\nStarting stream: {video_path}")
    print(f"Streaming URL: {rtmp_url}")
    
    # Build ffmpeg command
    cmd = ['ffmpeg']
    
    # Input options
    if loop:
        cmd.extend(['-stream_loop', '-1'])  # -1 means infinite loop
    
    cmd.extend(['-re', '-i', video_path])  # -re means read at original frame rate
    
    # Video encoding options
    cmd.extend(['-c:v', video_codec])
    cmd.extend(['-preset', preset])
    cmd.extend(['-b:v', bitrate])
    cmd.extend(['-maxrate', bitrate])
    cmd.extend(['-bufsize', str(int(bitrate.replace('k', '')) * 2) + 'k'])
    
    # Frame rate settings
    if fps:
        cmd.extend(['-r', str(fps)])
    
    # Resolution settings
    if resolution:
        cmd.extend(['-s', resolution])
    
    # Audio encoding options
    cmd.extend(['-c:a', audio_codec])
    cmd.extend(['-b:a', '128k'])
    cmd.extend(['-ar', '44100'])
    
    # RTMP output options
    cmd.extend(['-f', 'flv'])
    cmd.extend(['-flvflags', 'no_duration_filesize'])
    
    # Output URL
    cmd.append(rtmp_url)
    
    print(f"Executing command: {' '.join(cmd)}\n")
    
    # Execute streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Real-time output stderr (ffmpeg outputs to stderr)
    for line in process.stderr:
        print(line, end='')
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\nError during streaming, return code: {process.returncode}")
        return False
    
    return True


def parse_stream_list(file_path: str) -> List[tuple]:
    """
    Parse stream list from text file
    Each line format: video_path rtmp_url
    
    Args:
        file_path: Path to text file
        
    Returns:
        List of tuples (video_path, rtmp_url)
    """
    streams = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(None, 1)  # Split by whitespace, max 2 parts
            if len(parts) < 2:
                print(f"Warning: Line {line_num} invalid format, skipping: {line}")
                continue
            
            video_path, rtmp_url = parts[0], parts[1]
            streams.append((video_path, rtmp_url))
    
    return streams


def stream_multiple_videos(streams: List[tuple], loop: bool = False, **kwargs):
    """
    Stream multiple videos in loop
    
    Args:
        streams: List of tuples (video_path, rtmp_url)
        loop: Whether to infinitely loop all videos
        **kwargs: Other parameters passed to stream_video
    """
    if not streams:
        print("Error: No video streams provided")
        return
    
    print(f"Preparing to stream {len(streams)} video file(s)")
    for i, (video_path, rtmp_url) in enumerate(streams, 1):
        print(f"  {i}. {video_path} -> {rtmp_url}")
    
    while True:
        for video_path, rtmp_url in streams:
            success = stream_video(video_path, rtmp_url, loop=False, **kwargs)
            if not success:
                print(f"Video {video_path} streaming failed, skipping...")
                time.sleep(1)  # Brief wait before continuing to next
        
        if not loop:
            print("\nAll videos streaming completed")
            break
        
        print("\nAll videos playback completed, starting new cycle...")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description='Stream videos to RTMP server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream videos from text file with infinite loop
  python make_stream.py -f stream_list.txt --loop
  
  # Stream videos from text file, play once
  python make_stream.py -f stream_list.txt
  
  # Specify resolution and frame rate
  python make_stream.py -f stream_list.txt --resolution 1920x1080 --fps 30

Text file format (stream_list.txt):
  # Each line: video_path rtmp_url
  /path/to/video1.mp4 rtmp://localhost/live/stream1
  /path/to/video2.mp4 rtmp://localhost/live/stream2
  # Comments start with #
        """
    )
    
    parser.add_argument('-f', '--file', dest='input_file', required=True,
                       help='Input text file containing video paths and RTMP URLs (one per line: video_path rtmp_url)')
    parser.add_argument('--loop', action='store_true',
                       help='Infinitely loop all videos')
    parser.add_argument('--video-codec', default='libx264',
                       help='Video codec (default: libx264)')
    parser.add_argument('--audio-codec', default='aac',
                       help='Audio codec (default: aac)')
    parser.add_argument('--preset', default='veryfast',
                       choices=['ultrafast', 'superfast', 'veryfast', 'faster', 
                               'fast', 'medium', 'slow', 'slower', 'veryslow'],
                       help='Encoding preset (default: veryfast)')
    parser.add_argument('--bitrate', default='2000k',
                       help='Video bitrate (default: 2000k)')
    parser.add_argument('--fps', type=float, default=None,
                       help='Output frame rate (default: use original video frame rate)')
    parser.add_argument('--resolution', default=None,
                       help='Output resolution, format: WIDTHxHEIGHT, e.g.: 1920x1080')
    
    args = parser.parse_args()
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg not found, please install ffmpeg first")
        print("Installation: sudo apt-get install ffmpeg (Ubuntu/Debian)")
        sys.exit(1)
    
    # Parse stream list from file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    streams = parse_stream_list(args.input_file)
    
    if not streams:
        print("Error: No valid streams found in input file")
        sys.exit(1)
    
    # Validate video files
    valid_streams = []
    for video_path, rtmp_url in streams:
        if os.path.exists(video_path):
            valid_streams.append((video_path, rtmp_url))
        else:
            print(f"Warning: Video file not found, skipping: {video_path}")
    
    if not valid_streams:
        print("Error: No valid video files")
        sys.exit(1)
    
    # Display video information
    print("\nVideo information:")
    for video_path, rtmp_url in valid_streams:
        info = get_video_info(video_path)
        fps = info.get('fps')
        duration = info.get('duration')
        width = info.get('width') or '?'
        height = info.get('height') or '?'
        fps_str = f"{fps:.2f}" if fps is not None else "?"
        duration_str = f"{duration:.2f}s" if duration is not None else "?"
        print(f"  {os.path.basename(video_path)}: "
              f"{width}x{height}, "
              f"FPS: {fps_str}, "
              f"Duration: {duration_str}")
        print(f"    -> {rtmp_url}")
    
    # Start streaming
    if len(valid_streams) == 1:
        # Single video
        video_path, rtmp_url = valid_streams[0]
        stream_video(
            video_path,
            rtmp_url,
            loop=args.loop,
            video_codec=args.video_codec,
            audio_codec=args.audio_codec,
            preset=args.preset,
            bitrate=args.bitrate,
            fps=args.fps,
            resolution=args.resolution
        )
    else:
        # Multiple videos
        stream_multiple_videos(
            valid_streams,
            loop=args.loop,
            video_codec=args.video_codec,
            audio_codec=args.audio_codec,
            preset=args.preset,
            bitrate=args.bitrate,
            fps=args.fps,
            resolution=args.resolution
        )


if __name__ == '__main__':
    main()
