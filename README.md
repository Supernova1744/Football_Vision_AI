
---

# ⚽ Football Vision AI

Welcome to **Football Vision AI**! This project is all about bringing the excitement of football to the world of AI. Our system uses cutting-edge computer vision techniques to detect players, referees, goalkeepers, and the ball, providing real-time visualizations and insights into the state of the game. Whether you're a football enthusiast, a data scientist, or just curious about AI, this project has something for you!

## 🌟 Features

- **Player and Referee Detection**: Our AI can accurately identify and track all players, referees and the ball on the field using YOLOv8n and BoTSORT.

- **Goalkeeper Identification**: Distinguish goalkeepers from other players for specialized analysis.

- **Two teams Identification**: Differentiate the two teams for targeted analysis using PCA and KMeans only.

- **Real-time Visualizations**: Generate dynamic visual representations of the match state, making it easy to understand the flow of the game.

## 📌 Note

This project is an implementation based on a YouTube tutorial with some modifications and enhancements. You can watch the original tutorial [here](https://www.youtube.com/watch?v=aBVGKoNZQUw). Below are the key differences:

- Improved player and referee detection speed while maintaining accuracy by utilizing YOLOv8n instead of YOLOv8x.
- Enhanced keypoint detection efficiency without compromising accuracy by employing YOLOv8n Pose instead of YOLOv8x Pose.
- Improved players clustering algorithm by using PCA and K-Means on the cropped players without ReID.
- Optimized real-time visualizations.

Many visualization were taken from [Supervision](https://github.com/roboflow/supervision) Repo.

Tracker used [BoTSort](https://github.com/NirAharon/BoT-SORT/)
## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- OpenCV
- NumPy
- scikit-learn
- ONNXRuntime

### Installation

1. Clone the repository:
   ```bash
   git clone  https://github.com/Supernova1744/Football_AI.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Football_AI
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Prepare your video files.
2. Run the main script:
   ```bash
   python main.py --video videos\match1.mp4
   ```
3. View the results in the `output` directory.

### Demo Video

Check out a demo video of our system in action! Place your video oath in the `--videos` flag for the main script to visualize the analysis.

![Demo Video](videos/demo_video.mp4)

## 💬 Contact

If you have any questions or suggestions, feel free to open an issue or contact us at [ali.samir.1744@gmail.com](mailto:ali.samir.1744@gmail.com).

---
