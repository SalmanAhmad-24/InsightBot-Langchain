# InsightBot-Langchain

### **Description**

InsightBot is an intelligent document retrieval and summarization tool built using LangChain, Pinecone, and OpenAI models. It is designed to assist users in extracting **actionable insights** and **strategies** from large datasets, including books, articles, and other documents. 

Using **Pinecone** for vector-based similarity search and **OpenAI’s GPT-4** for powerful text generation, InsightBot helps you quickly locate and rank key insights, actionable steps, and other important information that are most relevant to your query.

---

### **Key Features**

- **Vector-based Retrieval**: Uses **Pinecone** to search through large document datasets (e.g., books, PDFs) for relevant information based on semantic similarity.
- **Context-Aware Answers**: Combines retrieved documents with **OpenAI's GPT-4** to generate responses that are tailored to your question.
- **Comprehensive Action Items**: Extracts insights and ranks them based on relevance, impact, and source credibility.
- **Structured Responses**: Provides structured output with separated **insights** and **action steps** for easy consumption.

---

### **How It Works**

1. **Document Upload**: Upload documents (e.g., books, research papers) to the Pinecone vector database.
2. **Search for Information**: The user enters a query to search through the database using semantic similarity. The tool retrieves relevant excerpts from the documents.
3. **Actionable Insights Extraction**: The model analyzes the retrieved content to identify key insights, actionable strategies, and tactics.
4. **Ranked Results**: The insights and action steps are ranked based on their relevance to the query, their impact, and their frequency in the documents.
5. **Formatted Output**: The results are presented in a clean, structured format, making it easy to follow and apply the insights.

---

### **System Requirements**

- **Python**: Version 3.7 or higher.
- **Libraries**:
  - `openai`
  - `pinecone-client`
  - `streamlit`
  - `langchain`
  - `langgraph`
  - `python-dotenv`
  
Install dependencies using:

```bash
pip install openai pinecone-client streamlit langchain langgraph python-dotenv
```

---

### **Setup**

1. **Install Required Libraries**: Use `pip` to install the dependencies mentioned above.
   
2. **Set up Environment Variables**:
   - Create a `.env` file with the following keys:
     - `OPENAI_API_KEY`: Your OpenAI API key.
     - `PINECONE_API_KEY`: Your Pinecone API key.
     - `PINECONE_INDEX_NAME`: The name of your Pinecone index.
   
   Example `.env` file:

   ```
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX_NAME=your-pinecone-index-name
   ```

3. **Run the Application**:
   - After configuring the environment variables, you can run the app locally using Streamlit:

   ```bash
   streamlit run your_script_name.py
   ```

4. **Interacting with InsightBot**:
   - Once the Streamlit interface is up and running, you can enter a query (e.g., *"list all action items from Cialdini's Influence book"*) in the provided text box.
   - The bot will then retrieve the relevant content, analyze it, and display a structured list of actionable insights and strategies.
     
---

### **Contribution Guidelines**

If you want to contribute to InsightBot:

1. **Fork the repository** on GitHub.
2. **Submit a pull request** with your changes, enhancements, or bug fixes.
3. Ensure all tests pass and functionality remains intact.

