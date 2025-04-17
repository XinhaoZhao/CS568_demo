from flask import Flask, request, jsonify, render_template
import os
import openai
from dotenv import load_dotenv
import glob

app = Flask(__name__)
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_transcript_content(transcript_path):
    with open(transcript_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_all_transcripts_content():
    transcript_files = glob.glob('meeting_transcripts/*.txt')
    all_content = []
    for file in transcript_files:
        content = get_transcript_content(file)
        filename = os.path.basename(file)
        all_content.append(f"Transcript from {filename}:\n{content}\n")
    return "\n".join(all_content)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    
    if not message:
        return jsonify({'error': 'Missing message'}), 400
    
    try:
        all_transcripts = get_all_transcripts_content()
        
        # Create a prompt that includes all transcripts and the user's question
        prompt = f"""You are a helpful assistant analyzing multiple meeting transcripts. Here are all the transcripts:

{all_transcripts}

User's question: {message}

Please provide a helpful response based on all the transcripts content."""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes meeting transcripts and provides insights."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return jsonify({
            'response': response.choices[0].message.content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 