# Smart Accident Detection System

A Streamlit-based accident detection application that uses computer vision and object detection models to detect accidents in video streams and optionally send SMS alerts.

## Features
- Real-time or video-based accident detection
- YOLOv8 / RT-DETR object detection support
- Streamlit dashboard for monitoring
- Accident logging and analytics pages
- Optional Twilio SMS alerts
- PDF report generation

## Project Structure
- `app.py` - Main Streamlit application
- `main.py` - Video inference script
- `pages/` - Streamlit page modules
- `database.py` - SQLite database helpers
- `report_generator.py` - PDF report generation
- `requirements.txt` - Python dependencies

## Requirements
Python 3.10+ is recommended.

## Setup
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your configuration:

```env
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_FROM_NUMBER=your_twilio_number
TWILIO_TO_NUMBER=your_recipient_number
HF_TOKEN=your_huggingface_token
```

## Run the App
Start the Streamlit application:

```bash
streamlit run app.py
```

## Notes
- Large model files such as `.pt` files and video/data assets are not included by default.
- Make sure your environment has access to the required model files before running inference.

## License
This project is for educational and demo purposes.
