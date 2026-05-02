import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Charger le modèle depuis HuggingFace
model_name = "ton_username/studymate-gpt2"  # ← remplace ton_username
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

last_topic = {"topic": "", "last_answer": ""}

def ask_after(question):
    torch.manual_seed(42)
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:")[-1].strip()

def chat(question, history):
    if not question.strip():
        return history, ""
    
    follow_ups = ["yes", "oui", "sure", "ok", "go ahead", "please", "yes please", "why not", "yeah", "tell me more"]
    
    if question.lower().strip() in follow_ups and last_topic["topic"]:
        full_prompt = f"""Question: {last_topic['topic']}
Answer: {last_topic['last_answer']}
Student: Can you explain more with a different example?
Tutor:"""
        torch.manual_seed(42)
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=300).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("Tutor:")[-1].strip()
    else:
        response = ask_after(question)
        last_topic["topic"] = question
    
    last_topic["last_answer"] = response
    history.append((question, response))
    return history, ""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    .gradio-container { max-width: 800px !important; margin: auto !important; }
    .chat-header {
        text-align: center; padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; margin-bottom: 20px; color: white;
    }
    footer {display: none !important}
    """
) as demo:

    gr.HTML("""
    <div class="chat-header">
        <h1>🎓 StudyMate</h1>
        <p>Your patient AI tutor for Machine Learning, Deep Learning & LLMs</p>
    </div>
    """)

    chatbot = gr.Chatbot(
        label="Conversation",
        height=400,
        bubble_full_width=False,
        avatar_images=("👤", "🎓")
    )

    with gr.Row():
        question_input = gr.Textbox(
            placeholder="Ask me anything about AI, ML, Deep Learning...",
            label="", scale=4, lines=1
        )
        send_btn = gr.Button("Send 🚀", scale=1, variant="primary")

    gr.Examples(
        examples=[
            "What is machine learning?",
            "Explain overfitting simply.",
            "What is a neural network?",
            "What is gradient descent?",
            "What is fine-tuning?"
        ],
        inputs=question_input,
        label="Try these examples"
    )

    history_state = gr.State([])

    send_btn.click(fn=chat, inputs=[question_input, history_state], outputs=[chatbot, question_input])
    question_input.submit(fn=chat, inputs=[question_input, history_state], outputs=[chatbot, question_input])

demo.launch()