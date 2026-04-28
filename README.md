RAG Based AI Chatbot - AI Document Assistant

Project Explanation:
This AI-Powered Document Chat Assistant is a sophisticated question-answering system that enables users to upload documents (PDF, DOCX, TXT) and engage in intelligent conversations about their content. The application processes documents by extracting text, cleaning and preprocessing it, then splitting it into semantic chunks for efficient retrieval. It utilizes advanced Retrieval-Augmented Generation (RAG) architecture, combining vector embeddings with traditional keyword search for hybrid document retrieval. When users ask questions, the system searches through the document chunks using both semantic similarity (via sentence transformers) and keyword matching, retrieves the most relevant context, and generates comprehensive answers using Mistral AI's language model. The interface features a clean, single-section design with real-time chat functionality, confidence scoring for responses, and automatic source attribution, making it ideal for document analysis, research assistance, and content exploration.
Technical Stack & Keywords
Frontend & UI

Streamlit - Python web framework for rapid UI development, handles user interactions and real-time updates
HTML/CSS - Custom styling for chat bubbles, animations, and responsive design elements
JavaScript (implicit) - Streamlit's built-in reactivity for dynamic content updates

