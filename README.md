# TalentLens – RAG-Based Resume Intelligence System

TalentLens is an AI-powered resume analysis application that uses **Retrieval-Augmented Generation (RAG)** to evaluate resumes against specific job descriptions.

The application extracts information from PDF resumes, generates embeddings using Google's Gemini Embedding API, stores them in a FAISS vector database, and retrieves relevant information to generate an intelligent resume analysis using Gemini LLM.

It helps candidates understand how well their resumes match job requirements and identify areas for improvement.

## Features

- **PDF Resume Upload:** Upload resumes in PDF format.
- **Text Extraction:** Extract resume content using PDFPlumber.
- **Text Chunking:** Divide extracted resume text into smaller chunks for efficient retrieval.
- **Gemini Embeddings:** Convert resume chunks into numerical vector representations.
- **FAISS Vector Database:** Store and search resume embeddings efficiently.
- **Semantic Search:** Retrieve relevant resume information based on the job description.
- **AI-Powered Analysis:** Generate detailed resume insights using Gemini LLM.
- **Job Match Score:** Estimate how closely the resume matches the job description.
- **Skill Identification:** Identify matching and potentially missing skills.
- **Improvement Suggestions:** Provide recommendations to improve the resume.
- **Terminal Output:** Print the complete analysis report in the terminal.
- **Web Interface:** Display the analysis results through a Flask-based interface.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Generative AI | Google Gemini API |
| Embeddings | Gemini Embedding API |
| Vector Database | FAISS |
| PDF Processing | PDFPlumber |
| Numerical Computing | NumPy |
| Frontend | HTML, CSS |
| Environment Management | Python-dotenv |

## System Architecture

The TalentLens application follows a Retrieval-Augmented Generation (RAG) architecture.

### RAG Pipeline

1. **Resume Upload:** The user uploads a PDF resume.
2. **Text Extraction:** PDFPlumber extracts text from the uploaded resume.
3. **Text Cleaning:** Unnecessary spaces and special characters are removed.
4. **Chunking:** The extracted text is divided into smaller overlapping chunks.
5. **Embedding Generation:** Gemini Embedding API converts each chunk into a vector representation.
6. **Vector Storage:** FAISS stores the generated embeddings.
7. **Query Embedding:** The job description is converted into an embedding.
8. **Semantic Retrieval:** FAISS retrieves the most relevant resume chunks using similarity search.
9. **LLM Processing:** Retrieved context and the job description are sent to Gemini.
10. **Result Generation:** Gemini generates the resume analysis and recommendations.
11. **Output:** Results are displayed on the website and printed in the terminal.

### Architecture Flow

```text
       User
        |
        v
   Upload Resume PDF
        |
        v
   PDF Text Extraction
        |
        v
     Text Cleaning
        |
        v
    Text Chunking
        |
        v
 Gemini Embeddings API
        |
        v
   FAISS Vector Database
        |
        |             Job Description
        |                    |
        |                    v
        |             Query Embedding
        |                    |
        +-----------> Semantic Search
                             |
                             v
                    Relevant Resume Chunks
                             |
                             v
                       Gemini LLM
                             |
                             v
                    Resume Analysis
                             |
                  +----------+----------+
                  |                     |
                  v                     v
             Flask Website        Terminal Output
```

## Project Structure

```text
TalentLens-RAG-Based-Resume-Analyzer/
│
├── app.py
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── .env
│
├── .gitignore
│
├── requirements.txt
│
├── Procfile
│
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/TalentLens-RAG-Based-Resume-Analyzer.git
```

### 2. Navigate to the Project Folder

```bash
cd TalentLens-RAG-Based-Resume-Analyzer
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Gemini API Key

Create a `.env` file in the project root directory.

Add the following configuration:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
```

Replace `your_gemini_api_key` with your actual Gemini API key.

You can obtain an API key through [Google AI Studio](https://aistudio.google.com/).

**Important:** Never upload your API key or `.env` file to GitHub.

### 6. Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## How to Use

1. Open the TalentLens web application.
2. Upload your resume in PDF format.
3. Paste the job description into the provided text area.
4. Click **Analyze My Resume**.
5. Wait for the AI-powered analysis.
6. View the generated report on the website.
7. Check the terminal for the complete analysis output.

## Analysis Results

TalentLens generates the following insights:

- **Resume Summary:** Overview of the candidate's profile.
- **Skills Identified:** Skills extracted from the resume.
- **Job Match Score:** Estimated resume-to-job-description match.
- **Matching Skills:** Skills relevant to the job requirements.
- **Missing Skills:** Important requirements not identified in the resume.
- **Strengths:** Relevant qualifications and capabilities.
- **Areas for Improvement:** Suggestions to improve the resume.
- **Recommendations:** Actionable career and resume improvement suggestions.

## RAG Implementation

Unlike a basic resume analyzer that sends the entire resume directly to an LLM, TalentLens uses a RAG pipeline.

The system first retrieves relevant information from the resume using vector similarity search. The retrieved information is then provided to Gemini as context for generating the final response.

This approach helps ground the analysis in the candidate's actual resume content.

## Future Enhancements

- Support for multiple resume formats.
- Advanced resume ranking and comparison.
- Integration with multiple job descriptions.
- Downloadable PDF analysis reports.
- Improved chunking and retrieval strategies.
- Reranking of retrieved resume chunks.
- User authentication and analysis history.
- Resume improvement and rewriting suggestions.

## Limitations

- Currently supports text-based PDF resumes.
- Scanned PDFs may require OCR integration.
- The match score is an AI-generated estimate, not an official ATS score.
- Analysis quality depends on resume content, job description quality, and API availability.
- FAISS vectors are currently created in memory for each analysis request.

## Author

**Developed as an AI and Generative AI project using Python, RAG, FAISS, and Google Gemini.**

---

⭐ If you find this project useful, consider giving the repository a star on GitHub.
