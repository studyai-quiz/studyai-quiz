from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json
import traceback
from openai import OpenAI
# Note: Removed 'import io' as image compression is deleted.

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
# Corrected print message to reflect the model used in the API call
print("✓ OpenAI API client configured (gpt-4o)")

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__, static_folder="public")
CORS(app)
# Set max file upload size to 50MB (compression notes removed)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# -----------------------------
# File upload configuration
# -----------------------------
# UPDATED: Only PDF and TXT are allowed
ALLOWED_EXTENSIONS = {"pdf", "txt"}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Checks if the file extension is one of the allowed types (pdf, txt)."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# NOTE: The compress_image and chunk_text functions were removed.

def extract_text_from_file(file):
    """Extract text from uploaded TXT or PDF file."""
    filename = file.filename
    extension = filename.rsplit(".", 1)[1].lower()

    if extension == "txt":
        try:
            file.seek(0)
            content = file.read()
            # Limit text file size to 5MB
            if len(content) > 5 * 1024 * 1024:
                content = content[:5 * 1024 * 1024]
            return content.decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"TXT processing error: {e}")
            return None
    
    elif extension == "pdf":
        try:
            # Lazy import PyPDF2 for optional dependency handling
            import PyPDF2 
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            max_pages = 100  # Limit to first 100 pages for large PDFs
            total_pages = len(pdf_reader.pages)
            pages_to_process = min(total_pages, max_pages)
            
            for i in range(pages_to_process):
                try:
                    page = pdf_reader.pages[i]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    # Limit total extracted text to 50,000 characters
                    if len(text) > 50000:
                        text = text[:50000] + "\n[Content truncated due to size]"
                        break
                except Exception as e:
                    # Ignore pages that fail to extract text
                    continue
            
            if pages_to_process < total_pages:
                text += f"\n[Note: PDF has {total_pages} pages, processed first {pages_to_process} pages]"
            
            return text if text.strip() else None
        except ImportError:
            print("PyPDF2 not installed. Install with: pip install pypdf2")
            return "[PDF file uploaded but PyPDF2 library not installed for text extraction.]"
        except Exception as e:
            print(f"PDF processing error: {e}")
            return None
    
    # All other extensions are disallowed
    else:
        return None

# -----------------------------
# OpenAI API call using SDK
# -----------------------------
def call_openai_api(prompt, max_tokens=2000):
    """Call OpenAI API via the Python SDK"""
    try:
        if not prompt.strip():
            print("⚠️ Empty prompt, skipping API call")
            return None
        
        print("➡ Sending prompt to OpenAI API...")
        response = client.chat.completions.create(
            # Using gpt-4o-mini as specified in your call
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are an educational AI assistant. Always respond with valid JSON when requested."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        output_text = response.choices[0].message.content
        print("✅ Received response from OpenAI API")
        return output_text
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return send_from_directory("public", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("public", path)

@app.errorhandler(413)
def request_entity_too_large(error):
    # UPDATED: Removed mention of compression
    return jsonify({"error": "File size exceeds 50MB limit."}), 413

@app.route("/api/process-files", methods=["POST"])
def process_files():
    try:
        if "files" not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            return jsonify({"error": "No files selected"}), 400

        combined_text = ""
        processed_files = []
        errors = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Removed extension check, as allowed_file already ensures it's PDF or TXT
                
                try:
                    # Removed all image compression logic.
                    
                    text = extract_text_from_file(file)
                    if text:
                        combined_text += f"\n\n--- Content from {filename} ---\n\n{text}"
                        processed_files.append(filename)
                    else:
                        errors.append(f"{filename}: Could not extract text")
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")
            else:
                 # Handle files that were not allowed (e.g., a .docx file)
                 if file and file.filename:
                    errors.append(f"{file.filename}: File type not supported. Only PDF and TXT are supported.")


        if not combined_text.strip():
            error_msg = "Could not extract text from files."
            if errors:
                error_msg += " " + " ".join(errors)
            return jsonify({"error": error_msg}), 400

        # -----------------------------
        # Generate explanation
        # -----------------------------
        # Limit text size for API
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000] + "...\n[Content truncated for processing]"
        
        explanation_prompt = f"""You are an educational AI assistant. Analyze the following study material and create a comprehensive, easy-to-understand explanation.

Study Material:
{combined_text}

Please provide:
1. A clear topic/title for this material
2. A detailed explanation broken into 3-5 paragraphs that summarizes the key concepts, main ideas, and important information in an educational and easy-to-understand manner.

Format your response as JSON with this structure:
{{
  "topic": "Topic Title Here",
  "content": [
    "First paragraph of explanation...",
    "Second paragraph...",
    "Third paragraph...",
    "Fourth paragraph...",
    "Fifth paragraph..."
  ]
}}

Only return the JSON, no additional text."""

        explanation_text = call_openai_api(explanation_prompt)
        
        if not explanation_text:
            return jsonify({"error": "Failed to generate explanation. Please try again."}), 500
        
        # Extract JSON from response
        explanation_text = explanation_text.strip()
        if explanation_text.startswith('```json'):
            explanation_text = explanation_text.replace('```json', '').replace('```', '').strip()
        elif explanation_text.startswith('```'):
            explanation_text = explanation_text.replace('```', '').strip()
        
        try:
            explanation_data = json.loads(explanation_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {explanation_text[:500]}")
            # Fallback: create structure from text
            explanation_data = {
                "topic": "Study Material Analysis",
                "content": explanation_text.split('\n\n')[:5] if explanation_text else ["Unable to generate explanation."]
            }

        # -----------------------------
        # Generate quiz questions (3-5 questions)
        # -----------------------------
        quiz_prompt = f"""Based on this study material, create 3-5 multiple-choice questions that test understanding of the most important concepts.

Study Material:
{combined_text}

Create questions with:
- Clear, concise questions
- 4 answer options labeled A, B, C, D for each question
- One correct answer per question

Format your response as JSON array:
[
  {{
    "question": "Question text here?",
    "options": [
      "A) First option",
      "B) Second option",
      "C) Third option",
      "D) Fourth option"
    ],
    "correctAnswer": "B"
  }},
  {{
    "question": "Another question?",
    "options": ["A) Option A", "B) Option B", "C) Option C", "D) Option D"],
    "correctAnswer": "A"
  }}
]

Only return the JSON array, no additional text."""

        quiz_text = call_openai_api(quiz_prompt)
        
        if not quiz_text:
            return jsonify({"error": "Failed to generate quiz. Please try again."}), 500
        
        # Extract JSON from response
        quiz_text = quiz_text.strip()
        if quiz_text.startswith('```json'):
            quiz_text = quiz_text.replace('```json', '').replace('```', '').strip()
        elif quiz_text.startswith('```'):
            quiz_text = quiz_text.replace('```', '').strip()
        
        try:
            quiz_data = json.loads(quiz_text)
            # Ensure it's a list
            if not isinstance(quiz_data, list):
                quiz_data = [quiz_data]
            
            # Validate quiz data structure
            valid_questions = []
            for i, q in enumerate(quiz_data):
                if isinstance(q, dict) and 'question' in q and 'options' in q and 'correctAnswer' in q:
                    valid_questions.append(q)
                else:
                    print(f"Warning: Invalid question structure at index {i}: {q}")
            
            if not valid_questions:
                raise ValueError("No valid questions found in quiz data")
            
            quiz_data = valid_questions
            print(f"✓ Generated {len(quiz_data)} valid quiz questions")
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Quiz generation/parsing error: {e}")
            print(f"Raw response: {quiz_text[:500]}")
            # Fallback quiz
            quiz_data = [
                {
                    "question": "What is the main topic of the study material?",
                    "options": ["A) Option A", "B) Option B", "C) Option C", "D) Option D"],
                    "correctAnswer": "B"
                }
            ]
        except Exception as e:
            print(f"Error processing quiz data: {e}")
            quiz_data = [
                {
                    "question": "What is the main topic of the study material?",
                    "options": ["A) Option A", "B) Option B", "C) Option C", "D) Option D"],
                    "correctAnswer": "B"
                }
            ]

        # Ensure explanation_data is in the right format for frontend
        explanation_for_storage = [explanation_data] if isinstance(explanation_data, dict) else explanation_data
        
        # Debug: Print what we're sending
        print(f"Returning explanation (type: {type(explanation_for_storage)}, length: {len(explanation_for_storage) if isinstance(explanation_for_storage, list) else 'N/A'})")
        print(f"Returning quiz (type: {type(quiz_data)}, length: {len(quiz_data)})")
        
        return jsonify({
            "success": True,
            "explanation": explanation_for_storage,
            "quiz": quiz_data,
            "files_processed": processed_files,
        })

    except Exception as e:
        print(f"Error in process_files: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    print("Starting Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
