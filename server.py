from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import json
import traceback
from openai import OpenAI

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
print("✓ OpenAI API client configured (gpt-4o-mini)")

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__, static_folder="public")
CORS(app)

# -----------------------------
# File upload configuration
# -----------------------------
ALLOWED_EXTENSIONS = {"pdf", "txt"}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file):
    filename = file.filename
    extension = filename.rsplit(".", 1)[1].lower()

    if extension == "txt":
        file.seek(0)
        return file.read().decode("utf-8", errors="ignore")

    elif extension == "pdf":
        try:
            import PyPDF2
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text if text.strip() else None
        except Exception as e:
            print(f"PDF processing error: {e}")
            return None

    return None

def chunk_text(text, max_length=2000):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+max_length])
        start += max_length
    return chunks

# -----------------------------
# OpenAI API call via SDK
# -----------------------------
def call_openai_api(prompt, max_tokens=2000):
    try:
        if not prompt.strip():
            print("⚠️ Empty prompt, skipping API call")
            return None
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an educational AI assistant. Always respond with valid JSON when requested."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
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

        # Extract text
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                try:
                    text = extract_text_from_file(file)
                    if text:
                        combined_text += f"\n\n--- Content from {filename} ---\n\n{text}"
                        processed_files.append(filename)
                    else:
                        errors.append(f"{filename}: Could not extract text")
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")

        if not combined_text.strip():
            return jsonify({"error": "Could not extract text from files."}), 400

        # Limit size for OpenAI
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000] + "...\n[Content truncated]"

        # -----------------------------
        # Explanation generation
        # -----------------------------
        explanation_prompt = f"""You are an educational AI assistant. Analyze the following study material and create a comprehensive explanation.

Study Material:
{combined_text}

Return JSON:
{{
  "topic": "Title",
  "content": [
    "Paragraph 1",
    "Paragraph 2",
    "Paragraph 3",
    "Paragraph 4",
    "Paragraph 5"
  ]
}}"""

        explanation_text = call_openai_api(explanation_prompt)

        explanation_text = explanation_text.strip()
        if explanation_text.startswith("```"):
            explanation_text = explanation_text.replace("```json", "").replace("```", "").strip()

        try:
            explanation_data = json.loads(explanation_text)
        except:
            explanation_data = {
                "topic": "Study Material Analysis",
                "content": explanation_text.split("\n\n")[:5]
            }

        # -----------------------------
        # Quiz generation
        # -----------------------------
        quiz_prompt = f"""Create 3-5 multiple-choice questions from this material.

Study Material:
{combined_text}

Return JSON array:
[
  {{
    "question": "text",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correctAnswer": "B"
  }}
]"""

        quiz_text = call_openai_api(quiz_prompt)

        quiz_text = quiz_text.strip()
        if quiz_text.startswith("```"):
            quiz_text = quiz_text.replace("```json", "").replace("```", "").strip()

        try:
            quiz_data = json.loads(quiz_text)
            if not isinstance(quiz_data, list):
                quiz_data = [quiz_data]
        except:
            quiz_data = [
                {
                    "question": "What is the main idea?",
                    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
                    "correctAnswer": "A"
                }
            ]

        return jsonify({
            "success": True,
            "explanation": [explanation_data],
            "quiz": quiz_data,
            "files_processed": processed_files,
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Render-compatible entry point
# -----------------------------
# IMPORTANT:
# Do NOT use app.run() when deploying to Render.
# Render uses Gunicorn to run the app.
#
# Command to run on Render:
#   gunicorn server:app
# -----------------------------

if __name__ == "__main__":
    # Local development only
    app.run(host="0.0.0.0", port=5000)
