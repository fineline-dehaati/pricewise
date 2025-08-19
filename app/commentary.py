import os
import toml
from openai import OpenAI

def load_openai_client():
    """Load OpenAI client with API key from secrets.toml"""
    try:
        # Try to load from secrets.toml first
        secrets_path = os.path.join(os.path.dirname(__file__), "..", ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            api_key = secrets.get("openai", {}).get("api_key")
            if api_key and api_key != "sk-REPLACE_ME":
                print(f"🔑 Commentary: OpenAI API key loaded from secrets.toml")
                return OpenAI(api_key=api_key)
        
        # Fallback to environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print(f"🔑 Commentary: OpenAI API key loaded from environment")
            return OpenAI(api_key=api_key)
            
        print("⚠️  Commentary: No OpenAI API key found")
        return None
        
    except Exception as e:
        print(f"❌ Commentary: Error loading OpenAI client: {e}")
        return None

# Initialize OpenAI client
client = load_openai_client()

def narrate_insights(question, result, warnings, plan=None):
    if result is None or len(result)==0:
        return "No results to analyze."
    prompt = f"User question: {question}\nPlan: {plan}\nHead of result: {result.head(5).to_dict()}\nWrite 3 bullet insights and 1 short recommendation."
    if client:
        try:
            print(f"🚀 Commentary: Making OpenAI API call...")
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"You are an analyst who writes crisp bullet insights."},
                          {"role":"user","content":prompt}],
                max_tokens=250
            )
            print(f"✅ Commentary: OpenAI API call successful!")
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Commentary: OpenAI API call failed: {e}")
            pass
    else:
        print("⚠️  Commentary: No OpenAI client available")
    
    return f"Question: {question}\nPlan: {plan}\nNotes: {warnings}\n(Local commentary stub)"
