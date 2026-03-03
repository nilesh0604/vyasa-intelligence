import json
import random

from locust import HttpUser, between, task


class VyasaUser(HttpUser):
    wait_time = between(1, 3)

    # Sample questions for load testing
    questions = [
        "Who is Arjuna?",
        "What is the Bhagavad Gita?",
        "Explain the dharma dilemma in the Mahabharata",
        "Who are the Pandavas?",
        "What happened at the Battle of Kurukshetra?",
        "Who is Dronacharya?",
        "What are the Vedas?",
        "Explain the concept of karma in the Mahabharata",
        "Who is Kunti?",
        "What is the role of Krishna in the Mahabharata?",
        "Describe the game of dice",
        "Who is Bhishma?",
        "What are the eighteen parvas of the Mahabharata?",
        "Explain the concept of dharma",
        "Who is Draupadi?",
    ]

    user_roles = ["public", "scholar", "admin"]

    def on_start(self):
        """Called when a user starts"""
        # Check health endpoint first
        self.client.get("/health")

    @task(3)
    def ask_question(self):
        """Ask a random question - most common task"""
        question = random.choice(self.questions)  # nosec B311
        role = random.choice(self.user_roles)  # nosec B311

        payload = {"question": question, "user_role": role}

        with self.client.post(
            "/query", json=payload, catch_response=True, timeout=30
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "answer" not in data:
                        response.failure("No answer in response")
                    else:
                        response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def check_health(self):
        """Check health endpoint"""
        self.client.get("/health")

    @task(1)
    def ask_entity_question(self):
        """Ask entity-specific questions (BM25 path)"""
        entity_questions = [
            "Who is Karna?",
            "Tell me about Yudhisthira",
            "What weapons did Arjuna use?",
            "Where is Hastinapura?",
            "Who is the mother of the Pandavas?",
        ]

        question = random.choice(entity_questions)  # nosec B311
        payload = {"question": question, "user_role": "scholar"}

        self.client.post("/query", json=payload, timeout=30)

    @task(1)
    def ask_philosophical_question(self):
        """Ask philosophical questions (Dense + HyDE path)"""
        philo_questions = [
            "What is the meaning of dharma?",
            "Explain the concept of righteous duty",
            "What does the Mahabharata teach about karma?",
            "How should one handle moral dilemmas?",
            "What is the ultimate goal of life according to the Mahabharata?",
        ]

        question = random.choice(philo_questions)  # nosec B311
        payload = {"question": question, "user_role": "scholar"}

        self.client.post("/query", json=payload, timeout=30)
