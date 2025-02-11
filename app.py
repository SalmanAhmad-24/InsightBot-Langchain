import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.graph import END
from langgraph.prebuilt import tools_condition

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize LLM
llm = init_chat_model("gpt-4-turbo-2024-04-09", model_provider="openai", max_tokens=None)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

# Initialize Graph Builder
graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=20)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    conversation_history = state["messages"]
    response = llm_with_tools.invoke(conversation_history)

    # Append the response to the conversation
    conversation_history.append(response)

    # Ensure tool calls are handled and passed on
    if isinstance(response, dict) and "tool_calls" in response:
        tool_response = response["tool_calls"]
        for tool_call in tool_response:
            tool_call_id = tool_call.get("tool_call_id")
            if tool_call_id:
                query = tool_call.get("query")
                # Call the tool and get the response
                tool_response_message = retrieve(query)
                # Now construct the message as per the solution in the community
                tool_message = ToolMessage(
                    tool_call_id=tool_call_id,
                    role="tool",
                    name="retrieve",
                    content=tool_response_message[0],  # The actual tool response
                )
                conversation_history.append(tool_message)

    return {"messages": conversation_history}

tools = ToolNode([retrieve])

def generate_response(state: MessagesState):
    """Generate answer using only the most recent retrieved content."""
    conversation_history = state["messages"]
    
    # Get the most recent tool message
    recent_tool_message = next(
        (msg for msg in reversed(conversation_history) if isinstance(msg, ToolMessage)), None
    )
    docs_content = recent_tool_message.content if recent_tool_message else ""

    system_message_content = (
    "You are an assistant specializing in providing relevant insights and actionable strategies to address a given user query along with the retrieved docs.\n\n"
    f"{docs_content}\n\n"
    "Your goal is to rank the most effective tactics, strategies, and key insights based on their relevance.\n\n"
    "The books are stored in a vector database, which you can access using the retrieve tool.\n\n"
    "Follow these steps to complete the task:\n\n"
    "1. Thoroughly search through all retrieved documents for information relevant to the query.\n\n"
    "2. For each relevant piece of information you find, extract:\n"
    "   a. Insights: Key ideas, concepts, or principles that relate to the query.\n"
    "   b. Action steps: Specific, actionable tactics or strategies that can be implemented to address the query.\n\n"
    "3. Compile a comprehensive list of all insights and action steps you've found, even if there are over 100 items.\n\n"
    "4. Rank the list in order of effectiveness. Consider factors such as:\n"
    "   - How directly the item addresses the query\n"
    "   - The potential impact of implementing the action step or applying the insight\n"
    "   - The frequency with which the item is mentioned across different books (if applicable)\n"
    "   - The credibility of the source or author\n\n"
    "5. Format your output as follows:\n\n"
    "- Use numbered lists for easy readability\n"
    "- Separate insights and action steps into two distinct sections\n"
    "- Include brief explanations or context for each item if necessary\n"
    "- Cite the source book for each item\n\n"
    "Present your findings in the following format:\n\n"
    "<results>\n"
    "<insights>\n"
    "1. [Most effective insight]\n"
    "- Brief explanation (if needed)\n"
    "- Source: [Book title]\n\n"
    "2. [Second most effective insight]\n"
    "- Brief explanation (if needed)\n"
    "- Source: [Book title]\n\n"
    "[Continue listing all insights...]\n"
    "</insights>\n\n"
    "<action_steps>\n"
    "1. [Most effective action step]\n"
    "- Brief explanation (if needed)\n"
    "- Source: [Book title]\n\n"
    "2. [Second most effective action step]\n"
    "- Brief explanation (if needed)\n"
    "- Source: [Book title]\n\n"
    "[Continue listing all action steps...]\n"
    "</action_steps>\n\n"
    "</results>\n\n"
    "Remember to include all relevant items, even if the list becomes very long. The goal is to provide a comprehensive resource that fully addresses the user's query."
    )


    # Prepare the prompt without duplicating conversation history
    prompt = [SystemMessage(system_message_content)] + [
        msg for msg in conversation_history if not isinstance(msg, HumanMessage)
    ]
    
    # Invoke LLM with the cleaned prompt
    response = llm.invoke(prompt)
    conversation_history.append(response)
    
    return {"messages": conversation_history}

# Add nodes to the graph
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate_response)

# Set entry point
graph_builder.set_entry_point("query_or_respond")

# Add conditional logic for retrieval
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)

# Ensure correct edge connections
graph_builder.add_edge("tools", "generate_response")
graph_builder.add_edge("generate_response", END)

# Compile the state graph
graph = graph_builder.compile()

# Streamlit UI
st.title("LangChain-Pinecone Q&A App")
st.write("Ask a question and retrieve contextually relevant answers.")

# Initialize session state if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input
user_input = st.text_input("Enter your query:")
if st.button("Submit") and user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Run through the graph
    for step in graph.stream({"messages": st.session_state["messages"]}, stream_mode="values"):
        response_message = step["messages"][-1]
        st.session_state["messages"].append(response_message)
        
        # Display response
        if isinstance(response_message, AIMessage):
            st.write(f"**AI:** {response_message.content}")
        elif isinstance(response_message, ToolMessage):
            st.write(f"**Retrieved Context:** {response_message.content}")
