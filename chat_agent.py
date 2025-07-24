# gemini_gradio_chatbot.py

import gradio as gr
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# --- Hardcoded API Key (Use env vars for production!) ---
GOOGLE_API_KEY = "your-gemini-api-key-here"

# --- Set up Gemini Agent ---
provider = GoogleProvider(api_key=GOOGLE_API_KEY)
model = GoogleModel('gemini-2.5-flash',provider=provider)   
agent = Agent(model=model)


# --- Chat state ---
chat_history = []

# --- Chat Function ---
async def chat(user_message, history):
    history = history or []
    history.append({"role": "user", "content": user_message})  # use lowercase 'user'

    full_prompt = "\n".join(f"{item['role']}: {item['content']}" for item in history)

    # Call Gemini LLM using async Agent
    result = await agent.run(full_prompt)

    # Extract plain text from AgentRunResult
    response = result.output if hasattr(result, 'output') else str(result)

    history.append({"role": "assistant", "content": response})
    return history, history


# --- Gradio UI ---
with gr.Blocks(title="Gemini Chatbot") as demo:
    gr.Markdown("## 🤖 Gemini Chatbot with Pydantic-AI + Gradio")
    
    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")
    clear_btn = gr.Button("Clear")

    state = gr.State([])

    msg.submit(chat, inputs=[msg, state], outputs=[chatbot, state])
    clear_btn.click(lambda: ([], []), None, [chatbot, state])

# --- Launch the app ---
demo.queue().launch()
 