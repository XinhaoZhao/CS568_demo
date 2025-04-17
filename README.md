# Meeting Transcript Analyzer

This application allows you to analyze meeting transcripts using ChatGPT. You can ask questions about the meeting content, get summaries, and create todo lists based on the discussion.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Place your meeting transcripts in the `meeting_transcripts` folder. The transcripts should be in .txt format.

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Select a meeting transcript from the dropdown menu
2. Ask questions about the meeting content in the chat interface
3. The AI will analyze the transcript and provide relevant answers

Example questions you can ask:
- What were the main topics discussed in the meeting?
- Create a todo list based on the meeting discussion
- What were the key decisions made?
- Who was assigned what tasks?
- Summarize the meeting in bullet points