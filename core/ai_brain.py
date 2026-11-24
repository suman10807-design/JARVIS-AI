
"""
JARVIS AI - Intelligence Engine
Powered by the OpenAI Responses API.
"""

import os
from openai import OpenAI


class JarvisBrain:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("JARVIS_MODEL", "gpt-5.5")
        self.history = []

        self.instructions = """
        You are JARVIS, an intelligent personal AI assistant.

        Your responsibilities:
        - Answer questions clearly and accurately.
        - Solve mathematics step by step.
        - Explain science and academic subjects.
        - Write, explain and debug programming code.
        - Help with research and general knowledge.
        - Have natural and friendly conversations.

        Explain difficult topics in simple language.
        Admit when you are uncertain.
        Never invent facts.
        """

    def ask(self, question):
        self.history.append({
            "role": "user",
            "content": question
        })

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                input=self.history[-12:]
            )

            answer = response.output_text

            self.history.append({
                "role": "assistant",
                "content": answer
            })

            return answer

        except Exception as error:
            self.history.pop()
            return f"JARVIS encountered an error: {error}"


if __name__ == "__main__":
    jarvis = JarvisBrain()

    while True:
        question = input("\nYou: ")

        if question.lower() in ["exit", "quit", "stop"]:
            print("JARVIS: Goodbye!")
            break

        print("\nJARVIS:", jarvis.ask(question))
