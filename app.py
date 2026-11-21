
import os

# Prevent numerical library memory/thread issues
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import time
import logging
import tempfile

import numpy as np
import faiss
import pdfplumber

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google import genai
from google.genai import types


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Add it to your .env file."
    )

client = genai.Client(api_key=API_KEY)

EMBEDDING_MODEL = "gemini-embedding-001"

GENERATION_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite"
)

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
TOP_K = 4
EMBEDDING_DIMENSION = 768
MAX_RETRIES = 2


# --------------------------------------------------
# 2. EXTRACT TEXT FROM PDF
# --------------------------------------------------

def extract_text_from_pdf(pdf_path):

    extracted_text = []

    try:
        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf.pages, start=1):

                page_text = page.extract_text()

                if page_text:
                    extracted_text.append(page_text)

                logger.info(
                    "Extracted text from page %s",
                    page_number
                )

        return "\n".join(extracted_text)

    except Exception as error:
        logger.exception("PDF extraction failed")
        raise ValueError(
            f"Unable to read the uploaded PDF: {error}"
        )


# --------------------------------------------------
# 3. CLEAN RESUME TEXT
# --------------------------------------------------

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"[^\w\s.,;:()@%+/#&'-]",
        "",
        text
    )

    return text.strip()


# --------------------------------------------------
# 4. SPLIT TEXT INTO CHUNKS
# --------------------------------------------------

def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    logger.info("Created %s chunks", len(chunks))

    return chunks


# --------------------------------------------------
# 5. GENERATE GEMINI EMBEDDINGS
# --------------------------------------------------

def generate_embedding(text, task_type):

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIMENSION
                )
            )

            embedding = response.embeddings[0].values

            return np.array(
                embedding,
                dtype="float32"
            )

        except Exception as error:

            logger.warning(
                "Embedding attempt %s failed: %s",
                attempt + 1,
                error
            )

            if attempt == MAX_RETRIES:
                raise

            time.sleep(2)


def generate_embeddings(chunks):

    embeddings = []

    for chunk in chunks:

        embedding = generate_embedding(
            chunk,
            "RETRIEVAL_DOCUMENT"
        )

        embeddings.append(embedding)

    return np.array(
        embeddings,
        dtype="float32"
    )


# --------------------------------------------------
# 6. CREATE FAISS VECTOR DATABASE
# --------------------------------------------------

def create_vector_database(chunks):

    embeddings = generate_embeddings(chunks)

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    index.add(embeddings)

    logger.info(
        "FAISS vector database created successfully"
    )

    return index


# --------------------------------------------------
# 7. RETRIEVE RELEVANT RESUME CHUNKS
# --------------------------------------------------

def retrieve_relevant_chunks(
    job_description,
    chunks,
    index
):

    query_embedding = generate_embedding(
        job_description,
        "RETRIEVAL_QUERY"
    )

    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    faiss.normalize_L2(query_embedding)

    number_of_results = min(
        TOP_K,
        len(chunks)
    )

    scores, indices = index.search(
        query_embedding,
        number_of_results
    )

    retrieved_chunks = []

    for index_position in indices[0]:

        if index_position >= 0:
            retrieved_chunks.append(
                chunks[index_position]
            )

    logger.info(
        "Retrieved %s relevant chunks",
        len(retrieved_chunks)
    )

    return retrieved_chunks


# --------------------------------------------------
# 8. GENERATE AI RESUME ANALYSIS
# --------------------------------------------------

def generate_resume_analysis(
    retrieved_chunks,
    job_description
):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an expert AI resume evaluator and career assistant.

Analyze the candidate's resume based on the given job description.

Use only the resume information provided in the context.
Do not invent skills, experience, qualifications, or achievements.

RESUME INFORMATION:
{context}

JOB DESCRIPTION:
{job_description}

Prepare a detailed and professional analysis using these sections:

1. RESUME SUMMARY
Give a short summary of the candidate's profile.

2. SKILLS IDENTIFIED
List the technical and relevant skills found in the resume.

3. JOB MATCH SCORE
Give an estimated match score out of 100.
Explain why this score was given.

4. MATCHING SKILLS
Identify skills that match the job description.

5. MISSING SKILLS
Identify important job requirements not found in the resume.
Do not assume the candidate lacks a skill simply because
it is not mentioned.

6. STRENGTHS
Highlight the candidate's relevant strengths.

7. AREAS FOR IMPROVEMENT
Suggest practical improvements to the resume.

8. RECOMMENDATIONS
Give actionable suggestions to improve the candidate's
suitability for this role.

Keep the response structured, honest, and easy to understand.
"""

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1800
                )
            )

            if response.text:

                logger.info(
                    "Analysis generated successfully using %s",
                    GENERATION_MODEL
                )

                return response.text

            raise ValueError(
                "Gemini returned an empty response."
            )

        except Exception as error:

            logger.warning(
                "Generation attempt %s failed: %s",
                attempt + 1,
                error
            )

            if attempt == MAX_RETRIES:
                raise

            time.sleep(3)


# --------------------------------------------------
# 9. PRINT ANALYSIS IN TERMINAL
# --------------------------------------------------

def print_analysis(analysis):

    print("\n")
    print("=" * 75)
    print("             TALENTLENS - RESUME ANALYSIS REPORT")
    print("=" * 75)

    print(analysis)

    print("=" * 75)
    print("             ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 75)
    print("\n")


# --------------------------------------------------
# 10. FLASK HOME ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        pdf_path = None

        try:

            resume = request.files.get("resume")

            job_description = request.form.get(
                "job_description",
                ""
            ).strip()

            if not resume or not resume.filename:

                return render_template(
                    "index.html",
                    error="Please upload your resume PDF."
                )

            if not resume.filename.lower().endswith(".pdf"):

                return render_template(
                    "index.html",
                    error="Only PDF files are supported."
                )

            if not job_description:

                return render_template(
                    "index.html",
                    error="Please enter the job description."
                )

            filename = secure_filename(resume.filename)

            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                dir=UPLOAD_FOLDER
            ) as temporary_file:

                pdf_path = temporary_file.name

                resume.save(pdf_path)

            logger.info("Resume uploaded successfully")

            # Extract resume text
            resume_text = extract_text_from_pdf(
                pdf_path
            )

            if not resume_text.strip():

                raise ValueError(
                    "No readable text found in the PDF. "
                    "Please upload a text-based PDF."
                )

            # Clean text
            resume_text = clean_text(
                resume_text
            )

            # Create chunks
            chunks = create_chunks(
                resume_text
            )

            if not chunks:

                raise ValueError(
                    "Unable to create resume chunks."
                )

            # Create vector database
            vector_index = create_vector_database(
                chunks
            )

            # Retrieve relevant chunks
            relevant_chunks = retrieve_relevant_chunks(
                job_description,
                chunks,
                vector_index
            )

            # Generate AI analysis
            analysis = generate_resume_analysis(
                relevant_chunks,
                job_description
            )

            # Print complete analysis in terminal
            print_analysis(analysis)

            logger.info(
                "TalentLens analysis completed successfully"
            )

            # Display analysis on website
            return render_template(
                "index.html",
                analysis=analysis,
                filename=filename
            )

        except Exception as error:

            logger.exception(
                "Resume analysis failed"
            )

            return render_template(
                "index.html",
                error=f"Analysis failed: {error}"
            )

        finally:

            # Delete temporary PDF
            if pdf_path and os.path.exists(pdf_path):

                try:
                    os.remove(pdf_path)

                except OSError:
                    logger.warning(
                        "Could not delete temporary PDF"
                    )

    return render_template("index.html")


# --------------------------------------------------
# 11. ERROR HANDLER
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        error="File is too large. Maximum allowed size is 10 MB."
    ), 413


# --------------------------------------------------
# 12. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
