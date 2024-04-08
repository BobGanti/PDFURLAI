# Personal Assistant (Chatbot)
Welcome to the repository of our advanced Personal Assistant (Chatbot), designed to enhance your productivity and information retrieval process. This Chatbot leverages artificial intelligence to provide responses based on general knowledge and specific information extracted from user-uploaded PDF documents.

# Features
General Knowledge Conversations: Engage in interactive dialogues on a wide range of topics.
Personalised Responses: Upload PDFs to get answers tailored from your own documents.
Advanced Text Retrieval: Uses the latest advancements in AI and NLP for accurate information extraction.

# Installation
Before you can run the Personal Assistant, ensure you have the following prerequisites installed:
> Python 3.8 or later
> Streamlit
> PyPDF2
> LangChain and related packages

# Follow these steps to install:
> git clone https://github.com/BobGanti/pdfurl.git
> cd yourrepositoryname
> pip install -r requirements.txt

# Setting Up
Environment Variables: Create a .env.local file in your project directory with the following content:
> OPENAI_API_KEY=your_openai_api_key_here
> INSTRUCTIONS=Custom instructions for your assistant
> PA_PROFILE=Your assistant's profile name

# Usage
To start the Personal Assistant, run:
> streamlit run pdfurl.py
> Navigate to the displayed URL to interact with your Chatbot.

# Uploading PDFs
In the sidebar, use the "Upload PDFs" section to add your documents. Once uploaded, click "Process PDFs" to incorporate them into your assistant's knowledge base.

# Chatting
Enter your questions or prompts in the chat input field. The assistant will respond based on its general knowledge and the information extracted from the uploaded PDFs.

# Architecture
Briefly describe the architecture of your Chatbot, including the main components like the PDF text extractor, vector store creation, and conversational chains.

# Contributing
Contributions to improve the Personal Assistant are welcome. Please follow the standard pull request process.


