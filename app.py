import os

import gradio as gr
import requests

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def ask_vyasa(question: str, user_role: str) -> str:
    resp = requests.post(
        f"{API_URL}/query",
        json={
            "question": question,
            "user_role": user_role,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        return f"Error: {resp.text}"
    data = resp.json()
    answer = data["answer"]
    sources = "\n".join(f"- {s}" for s in data.get("sources", []))
    return f"{answer}\n\n**Sources:**\n{sources}"


demo = gr.Interface(
    fn=ask_vyasa,
    inputs=[
        gr.Textbox(label="Your question about the Mahabharata"),
        gr.Dropdown(["public", "scholar", "admin"], label="Role", value="public"),
    ],
    outputs=gr.Markdown(label="Vyasa's Answer"),
    title="Vyasa Intelligence — Mahabharata RAG",
    description="Production-grade hybrid retrieval over 200K Mahabharata verses.",
    examples=[
        ["Who is Karna and what is his relationship with the Pandavas?", "public"],
        ["Explain the dharma dilemma in the Bhagavad Gita", "scholar"],
        ["List all weapons Arjuna received from the gods", "scholar"],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)  # nosec: B104
