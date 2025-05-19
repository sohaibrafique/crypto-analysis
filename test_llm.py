from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

def test_llm():
    print("Imports successful!")
    
    try:
        # Initialize LLM with new import
        llm = Ollama(
            model="llama3",
            temperature=0.7,
            timeout=60
        )
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You're a crypto trading assistant"),
            ("human", "Analyze this market situation: {input}")
        ])
        
        # Test chain
        chain = prompt | llm
        response = chain.invoke({"input": "BTC showing higher highs with increasing volume"})
        
        print("Ollama connection successful!")
        print("Response:", response)
        return True
    except Exception as e:
        print("Failed:", str(e))
        return False

if test_llm():
    print("All components working!")
else:
    print("Check installation steps")