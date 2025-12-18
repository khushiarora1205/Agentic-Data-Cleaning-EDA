import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("❌ GOOGLE_API_KEY is missing. Please set it in your .env file.")

# Define AI Model (Gemini)
# Using 'gemini-flash-latest' as confirmed by your diagnostic test
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    google_api_key=google_api_key,
    temperature=0,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
)

class CleaningState(BaseModel):
    input_text: str
    structured_response: str = ""

class AIAgent:
    def __init__(self):
        self.graph = self.create_graph()

    def create_graph(self):
        graph = StateGraph(CleaningState)

        def agent_logic(state: CleaningState) -> CleaningState:
            try:
                # DEBUG PRINT: Check if Agent is receiving input
                print(f"\n🤖 Agent Input (Preview): {state.input_text[:50]}...")
                
                response_msg = llm.invoke(state.input_text)
                
                # --- FIX START: Handle List vs String content ---
                content = response_msg.content
                
                # If content is a list (e.g. [{'type': 'text', ...}]), extract the text
                if isinstance(content, list):
                    # Extract 'text' field if it exists, otherwise convert to string
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            text_parts.append(block.get("text", ""))
                        else:
                            text_parts.append(str(block))
                    content = "".join(text_parts)
                
                # Ensure it is definitively a string
                content = str(content)
                # --- FIX END ---

                # DEBUG PRINT: Check what Gemini actually replied
                print(f"✅ Agent Output: {content[:100]}...")

                return CleaningState(
                    input_text=state.input_text, 
                    structured_response=content
                )
            except Exception as e:
                print(f"❌ Error in Agent: {str(e)}")
                return CleaningState(
                    input_text=state.input_text,
                    structured_response=f"Error: {str(e)}"
                )

        graph.add_node("cleaning_agent", agent_logic)
        graph.add_edge("cleaning_agent", END)
        graph.set_entry_point("cleaning_agent")
        return graph.compile()

    def process_data(self, df, batch_size=20):
        # DEBUG PRINT: Check if DataFrame is valid
        print(f"\n📊 Processing Data... Rows: {len(df)}")
        if len(df) == 0:
            return "Error: DataFrame is empty."

        cleaned_responses = []

        for i in range(0, len(df), batch_size):
            df_batch = df.iloc[i:i + batch_size]
            
            prompt = f"""
            You are an AI Data Cleaning Agent. 
            Input Data (CSV format):
            {df_batch.to_string()}

            Task:
            1. Identify missing values and impute them (mean/median/mode).
            2. Fix inconsistent formatting.
            3. Remove duplicates.

            Output:
            Return ONLY the cleaned dataset in CSV format. 
            NO explanations. NO markdown code blocks (like ```csv).
            """
            
            state = CleaningState(input_text=prompt, structured_response="")
            response = self.graph.invoke(state)
            
            # Extract content properly if response is a dict or state object
            if isinstance(response, dict):
                response = CleaningState(**response)

            cleaned_responses.append(response.structured_response)

        return "\n".join(cleaned_responses)
    def analyze_data(self, df):
        """
        Analyze the given DataFrame and return AI-generated insights.
        """
        # 1. Build context from the DataFrame
        column_info = df.dtypes.to_string()
        basic_stats = df.describe().to_string()
        first_rows = df.head().to_string()

        # 2. Construct the AI prompt
        prompt = f"""
        You are an AI Data Scientist. Analyze the following dataset summary and provide insights:

        Column Information:
        {column_info}

        Basic Statistics:
        {basic_stats}

        First 5 Rows of Data:
        {first_rows}

        Task:
        1. Identify 3 key trends in the data.
        2. Highlight any potential anomalies.
        3. Provide a brief conclusion about the dataset.
        
        Output:
        Return a clean Markdown formatted response.
        """

        try:
            # 3. Pass the prompt to the Gemini model
            response_msg = llm.invoke(prompt)
            
            # --- CRITICAL FIX: Handle List vs String content ---
            content = response_msg.content
            
            # If content is a list (e.g. [{'type': 'text', ...}]), extract the text
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        text_parts.append(block.get("text", ""))
                    else:
                        text_parts.append(str(block))
                content = "".join(text_parts)
            
            # Ensure it is definitively a string
            return str(content)
            # ---------------------------------------------------

        except Exception as e:
            return f"❌ Error in AI Analysis: {str(e)}"