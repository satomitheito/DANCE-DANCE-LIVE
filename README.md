# Dance Dance Live 💃

A Just Dance-style game using computer vision to track and score your dance moves in real-time.

## Features

- 🎥 Real-time pose tracking using MediaPipe
- 🎯 Live scoring with detailed breakdowns (angles, orientation, extension)
- 📊 Performance analytics with score graphs
- 🎬 Native HTML5 video playback (smooth, full-speed)
- 📱 Responsive design

## Architecture

```
├── app.py                      # Flask backend with RESTful API
├── pose_comparison.py          # Pose matching algorithms
├── static/
│   ├── css/style.css          # Styling
│   └── js/app.js              # Frontend logic
├── templates/
│   └── index.html             # Main UI
├── analyzed_pose_video.mp4    # Reference video (required)
└── analyzed_pose_video_landmarks.json  # Pre-computed landmarks (required)
```

## Setup

### 1. Install Dependencies

```bash
# Using venv
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Reference Video

Make sure you have:
- `analyzed_pose_video.mp4` - Your reference dance video
- `analyzed_pose_video_landmarks.json` - Pre-computed landmarks

Generate these using `full_body_analysis.py`:

```bash
python full_body_analysis.py
```

### 3. Run the App

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## Usage

1. **Click "Start Dancing"** - Video plays automatically
2. **Match the moves** - Your webcam captures your performance
3. **See your score** - Updated every second in real-time
4. **View results** - Detailed performance graph when complete

## API Endpoints

### `GET /api/video-info`
Get video metadata (fps, duration, dimensions)

### `POST /api/analyze-pose`
Analyze a single frame
```json
{
  "frame": "data:image/jpeg;base64,...",
  "timestamp": 1.5
}
```

### `POST /api/batch-analyze`
Analyze multiple frames for post-processing

## How It Works

1. **Reference Processing** (pre-computed):
   - Extract pose landmarks from reference video using MediaPipe
   - Store frame-by-frame landmarks in JSON

2. **Real-time Matching**:
   - Capture webcam at 30 FPS
   - Extract pose every 30th frame (~1 second intervals)
   - Normalize both poses (scale/position invariant)
   - Compare using:
     - Joint angles (60% weight)
     - Body orientation (30% weight)
     - Limb extension (10% weight)

3. **Scoring**:
   - Calculate similarity score (0-100)
   - Display live feedback
   - Store for post-analysis

## Performance

- **Video playback**: Native HTML5 (full 30 FPS with audio)
- **Pose analysis**: ~60ms per frame
- **Live scoring**: Every 1 second
- **Post-processing**: All frames analyzed after completion

## Technologies

- **Backend**: Flask, MediaPipe, OpenCV, NumPy
- **Frontend**: Vanilla JS, HTML5 Video, Chart.js
- **Pose Detection**: MediaPipe Pose (33 landmarks)

## License

MIT
