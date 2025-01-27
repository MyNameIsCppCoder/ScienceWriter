import asyncio
import dotenv
import google.generativeai as genai
import os

from google.generativeai.types import GenerateContentResponse


def init_project():
    return dotenv.get_key(key_to_get='GEMINI_API_KEY', dotenv_path='../.env')


class Memory:
    def __init__(self):
        self.versions = []
        self.feedback = []

    def store_draft(self, draft: str, version_name: str):
        self.versions.append((version_name, draft))

    def store_feedback(self, feedback_text: str, role: str):
        self.feedback.append((role, feedback_text))

    def get_last_draft(self) -> str:
        return self.versions[-1][1] if self.versions else ""


class StudentAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def produce_draft(self, plan, topic, constraints):
        prompt = f"""
        Ты выступаешь в роли Студента, который пишет научную статью.
        Тема: {topic}
        Ограничения: {constraints}
        План (от преподавателя):
        {plan}
        Напиши черновик научной статьи, учитывая эту структуру:
        1) Введение (значимость и актуальность темы)
        2) Основная часть (теоретические и практические основы)
        3) Заключение (итог, выводы, решение проблемы)
        """
        draft = self.llm_client.generate_content(prompt)
        return draft

    def rewrite_draft(self, plan, topic, constraints, current_text, feedback):
        prompt = f"""
        Ты студент, твоя предыдущая версия статьи:
        {current_text}

        Ниже приведены замечания, которые необходимо учесть:
        {feedback}

        План статьи:
        {plan}
        Тема: {topic}
        Ограничения: {constraints}

        Перепиши статью, исправляя перечисленные проблемы.
        """
        new_draft = self.llm_client.generate_content(prompt)
        return new_draft


class CriticAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def criticize_draft(self, draft_text: str) -> str:
        prompt = f"""
        Выступи в роли Критика. Ниже приводится черновик научной статьи:
        {draft_text}
        Укажи слабые стороны, неточности в фактах, логические несостыковки, методологические ошибки.
        Дай рекомендации, как улучшить статью, чтобы она соответствовала требованиям научного журнала.
        """
        critical_feedback = self.llm_client.generate_content(prompt)
        return critical_feedback


class SupervisorAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def give_plan(self, topic: str, count_word: str) -> str:
        prompt = f"""
        Представь, что ты научный руководитель. Составь план статьи для студента.
        Тема: {topic}
        Объём: {count_word} слов

        Структура плана: аннотация, введение, основная часть (теория + практика), заключение.
        """
        plan = self.llm_client.generate_content(prompt)
        return plan

    def evaluate_draft(self, draft_text: str) -> str:
        prompt = f"""
        Выступи в роли Научного руководителя. Вот черновик статьи:
        {draft_text}
        Дай оценку её структуре, содержанию и качеству. Укажи конкретные рекомендации по улучшению.
        """
        feedback = self.llm_client.generate_content(prompt)
        return feedback


class OrchestratorGemini:
    def __init__(self, gemini_api_key, model_version='gemini-1.5-flash'):
        genai.configure(api_key=gemini_api_key)
        self.llm_client = genai.GenerativeModel(model_version)
        self.supervisor = SupervisorAgent(self.llm_client)
        self.student = StudentAgent(self.llm_client)
        self.critic = CriticAgent(self.llm_client)
        self.memory = Memory()


    def multi_iter_workflow(self, topic: str, count_word: str, constraints: str, max_iterations=2):
        plan = self.supervisor.give_plan(topic, count_word)

        # 1. Первая версия черновика
        draft_v1 = self.student.produce_draft(plan, topic, constraints)
        self.memory.store_draft(draft_v1, "Draft_v1")

        # 2. Критика/оценка
        critique_v1 = self.critic.criticize_draft(draft_v1)
        supervisor_v1 = self.supervisor.evaluate_draft(draft_v1)
        critique_text = critique_v1.text
        supervisor_text = supervisor_v1.text
        combined_feedback_v1 = critique_text + "\n\n" + supervisor_text
        self.memory.store_feedback(combined_feedback_v1, "Critic+Supervisor")

        # Итерационный процесс
        for i in range(1, max_iterations):
            new_draft = self.student.rewrite_draft(
                plan=plan,
                topic=topic,
                constraints=constraints,
                current_text=self.memory.get_last_draft(),
                feedback=combined_feedback_v1
            )
            version_name = f"Draft_v{i + 1}"
            self.memory.store_draft(new_draft, version_name)

            new_critique = self.critic.criticize_draft(new_draft)
            new_supervisor = self.supervisor.evaluate_draft(new_draft)
            new_critique_text = new_critique.text
            new_supervisor_text = new_supervisor.text
            combined_feedback_v1 = new_critique_text + "\n\n" + new_supervisor_text
            self.memory.store_feedback(combined_feedback_v1, "Critic+Supervisor")

        # Возвращаем итог
        return {
            "plan": plan,
            "all_versions": self.memory.versions,
            "final_version": self.memory.get_last_draft(),
            "feedback_history": self.memory.feedback
        }


async def main():
    gemini_api_key = init_project()
    orchestrator = OrchestratorGemini(gemini_api_key)

    topic = "Роль искусственного интеллекта в медицине"
    count_word = "3000"
    constraints = "Строгая формальная лексика, упоминание реальных исследований"

    results = orchestrator.multi_iter_workflow(topic, count_word, constraints, max_iterations=2)

    # 1) Формируем содержимое для файла
    md_content = []
    md_content.append("# Итоги генерации научной статьи\n")
    md_content.append(f"**Тема**: {topic}\n")
    md_content.append(f"**План**:\n\n{results['plan']}\n")

    md_content.append("## Все версии черновиков\n")
    for version_name, draft_text in results["all_versions"]:
        md_content.append(f"### {version_name}\n")
        md_content.append(f"{draft_text}\n")

    md_content.append("## Итоговая версия\n")
    md_content.append(results["final_version"])

    md_content.append("\n## История замечаний\n")
    for role, feedback_text in results["feedback_history"]:
        md_content.append(f"### {role}\n")
        md_content.append(f"{feedback_text}\n")

    # 2) Сохраняем всё в Markdown-файл
    output_filename = "article_results.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    print(f"Результаты записаны в файл: {output_filename}")


if __name__ == '__main__':
    asyncio.run(main())
