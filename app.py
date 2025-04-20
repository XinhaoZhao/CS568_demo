from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import openai
from dotenv import load_dotenv
import glob
from werkzeug.utils import secure_filename

app = Flask(__name__)
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure upload folders
UPLOAD_FOLDER = 'meeting_transcripts'
COMPANY_RESOURCES_FOLDER = 'company_resources'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['COMPANY_RESOURCES_FOLDER'] = COMPANY_RESOURCES_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def get_company_resources_content():
    resource_files = glob.glob('company_resources/*')
    all_content = []
    for file in resource_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                filename = os.path.basename(file)
                all_content.append(f"Resource from {filename}:\n{content}\n")
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}")
    return "\n".join(all_content)

def generate_summary_and_todos(transcripts_content):
    try:
        # Generate summary
        summary_prompt = f"""Please provide a concise summary of the following meeting transcripts:

{transcripts_content}

Focus on the key points, decisions made, and important discussions."""
        
        summary_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise meeting summaries."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        summary = summary_response.choices[0].message.content

        # Generate to-do list
        todos_prompt = f"""Based on the following meeting transcripts, create a clear to-do list with action items:

{transcripts_content}

List each action item with a clear description and, if possible, assignee."""
        
        todos_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates clear to-do lists from meetings."},
                {"role": "user", "content": todos_prompt}
            ]
        )
        todos = todos_response.choices[0].message.content

        # Generate schedule/deadlines
        schedule_prompt = f"""Based on the following meeting transcripts, extract and list any mentioned schedules, deadlines, or timeframes:

{transcripts_content}

Format the response as a clear list of dates, deadlines, and timeframes mentioned in the meeting. If no specific dates or deadlines are mentioned, respond with "No specific deadlines or schedules mentioned in the meeting." """
        
        schedule_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts schedules and deadlines from meeting transcripts."},
                {"role": "user", "content": schedule_prompt}
            ]
        )
        schedule = schedule_response.choices[0].message.content

        return {
            'summary': summary,
            'todos': todos,
            'schedule': schedule
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/initial-analysis')
def initial_analysis():
    transcript_files = glob.glob('meeting_transcripts/*.txt')
    if not transcript_files:
        return jsonify({
            'status': 'empty',
            'message': 'No meeting transcripts found. Please upload meeting transcripts to get started.'
        })
    
    all_transcripts = get_all_transcripts_content()
    analysis = generate_summary_and_todos(all_transcripts)
    
    if 'error' in analysis:
        return jsonify({'error': analysis['error']}), 500
    
    return jsonify({
        'status': 'success',
        'summary': analysis['summary'],
        'todos': analysis['todos'],
        'schedule': analysis['schedule']
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Clear existing transcripts
        existing_files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.txt'))
        for existing_file in existing_files:
            try:
                os.remove(existing_file)
            except Exception as e:
                print(f"Error deleting file {existing_file}: {str(e)}")
        
        # Save new transcript
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Generate analysis for the new file
        all_transcripts = get_all_transcripts_content()
        analysis = generate_summary_and_todos(all_transcripts)
        
        if 'error' in analysis:
            return jsonify({'error': analysis['error']}), 500
        
        return jsonify({
            'message': 'File uploaded successfully',
            'summary': analysis['summary'],
            'todos': analysis['todos'],
            'schedule': analysis['schedule']
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/company-resources', methods=['GET'])
def get_company_resources():
    try:
        files = []
        for filename in os.listdir(app.config['COMPANY_RESOURCES_FOLDER']):
            file_path = os.path.join(app.config['COMPANY_RESOURCES_FOLDER'], filename)
            if os.path.isfile(file_path):
                files.append({
                    'name': filename,
                    'size': os.path.getsize(file_path),
                    'upload_date': os.path.getctime(file_path)
                })
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/company-resources/upload', methods=['POST'])
def upload_company_resource():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['COMPANY_RESOURCES_FOLDER'], filename))
        return jsonify({'message': 'File uploaded successfully'})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/company-resources/<filename>', methods=['DELETE'])
def delete_company_resource(filename):
    try:
        file_path = os.path.join(app.config['COMPANY_RESOURCES_FOLDER'], secure_filename(filename))
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'message': 'File deleted successfully'})
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    source = data.get('source', 'folder')  # Default to folder if not specified
    
    if not message:
        return jsonify({'error': 'Missing message'}), 400
    
    try:
        if source == 'folder':
            all_transcripts = get_all_transcripts_content()
        else:
            # Get the most recently uploaded file
            transcript_files = glob.glob('meeting_transcripts/*.txt')
            if not transcript_files:
                return jsonify({'error': 'No transcripts available'}), 400
            latest_file = max(transcript_files, key=os.path.getctime)
            all_transcripts = get_transcript_content(latest_file)
        
        # Get company resources content
        company_resources = get_company_resources_content()
        
        # Create a prompt that includes both transcripts and company resources
        prompt = f"""You are a helpful assistant analyzing meeting transcripts and company resources. Here are the meeting transcripts:

{all_transcripts}

And here are the company resources:

{company_resources}

User's question: {message}

Please provide a helpful response based on both the transcripts and company resources content. If the question is specifically about the meeting, focus on the transcripts. If it's about company resources, focus on those. If it's about both, combine information from both sources."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes meeting transcripts and company resources to provide insights."},
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