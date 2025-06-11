from groq import Groq

GROQ_API_KEY = ""
client = Groq(api_key=GROQ_API_KEY)

def chatResponse(query):
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        temperature=0.6,

        top_p=0.95,
        stream=True,
        stop=None,
    )
    response = ""
    for chunk in completion:
        if chunk is not None:
            response += chunk.choices[0].delta.content or ""
    return response
