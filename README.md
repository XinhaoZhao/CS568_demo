# Meeting Transcript Analyzer

A web application that analyzes meeting transcripts using AI to provide summaries, action items, and schedule information. It also integrates company resources for comprehensive context-aware responses.

## Features

### Meeting Analysis
- **Summary Generation**: Automatically generates concise summaries of meeting transcripts
- **To-Do List**: Extracts action items and tasks from meetings
- **Schedule & Deadlines**: Identifies and lists any mentioned dates, deadlines, or timeframes
- **Interactive Chat**: Ask questions about the meeting content and get AI-powered responses

### Company Resources Management
- Upload and manage company resources (documents, policies, guidelines)
- Resources are used as context for chat responses
- View and delete uploaded resources

### Transcript Management
- Upload new meeting transcripts (replaces existing ones)
- Use transcripts from the local folder
- Automatic analysis of uploaded transcripts

## Technical Requirements

- Python 3.7 or higher
- Flask
- OpenAI API key
- Required Python packages (listed in requirements.txt)

## Setup Instructions

1. Clone the repository:
```bash
git clone [repository-url]
cd CS568_demo
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Create necessary directories:
```bash
mkdir meeting_transcripts
mkdir company_resources
```

5. Start the application:
```bash
python app.py
```

6. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

### Starting the Application
1. Open the application in your web browser
2. You'll see the welcome screen with options to:
   - Start Analysis (if transcripts exist)
   - Upload New Transcript
   - Manage Company Resources

### Uploading Transcripts
1. Click "Upload New Meeting Transcript"
2. Select a .txt file containing the meeting transcript
3. Click "Upload Transcript"
4. Note: This will replace any existing transcripts

### Managing Company Resources
1. In the "Company Resources" section:
   - Upload new resources using the file upload interface
   - View existing resources with their details
   - Delete resources using the delete button

### Using the Analysis Features
1. Click "Start Analysis" to view:
   - Meeting Summary
   - To-Do List
   - Schedule & Deadlines

### Chat Interface
1. Type your question in the chat input
2. The AI will respond based on:
   - Meeting transcript content
   - Company resources (if relevant)
3. Questions can be about:
   - Meeting content
   - Company policies
   - Both meeting and company information

## Notes

- Meeting transcripts should be in .txt format
- Company resources can be in various formats (txt, pdf, doc, etc.)
- The application will automatically analyze new transcripts upon upload
- Chat responses consider both meeting content and company resources
- The analysis (summary, todos, schedule) only considers meeting transcripts

## Troubleshooting

1. If the application fails to start:
   - Check if all required packages are installed
   - Verify the OpenAI API key is correctly set in .env
   - Ensure the required directories exist

2. If uploads fail:
   - Check file permissions
   - Verify file format is supported
   - Ensure directories have write permissions

3. If analysis fails:
   - Check if transcripts are in the correct format
   - Verify the OpenAI API key is valid
   - Check server logs for specific errors
