from dotenv import load_dotenv
load_dotenv()
import os
try:
    import google.generativeai as genai
except Exception as e:
    print("Import failed:", e)
    raise

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
print("API_KEY present:", bool(API_KEY))
print("MODEL:", MODEL)

genai.configure(api_key=API_KEY)

# Try the robust path used above
try:
    if hasattr(genai, "GenerativeModel"):
        m = genai.GenerativeModel(MODEL)
        r = m.generate_content("Say hi in 3 words.")
        text = getattr(r, "text", None) or (r.output[0].content[0].text if hasattr(r, "output") else str(r))
        print("Modern SDK success:", text)
    elif hasattr(genai, "generate"):
        r = genai.generate(model=MODEL, input="Say hi in 3 words.")
        print("Older SDK success:", r)
    else:
        print("No known invocation method on genai module.")
except Exception as e:
    print("Invocation failed:", e)
    raise
