import os
import asyncio
import time
import logging

from dotenv import load_dotenv
# LangChain: Chat-модель + базовые утилиты
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Для подсчёта токенов (приблизительно)
from transformers import GPT2TokenizerFast  # <-- Добавлено

# Подгружаем .env, если нужно
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("science_writer.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def log_and_time(func):
    """Декоратор для логирования вызовов функций и измерения времени выполнения."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Начало выполнения функции: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            logging.info(f"Завершено выполнение функции: {func.__name__} за {elapsed:.2f} секунд")
            return result
        except Exception as e:
            end_time = time.time()
            elapsed = end_time - start_time
            logging.error(f"Ошибка в функции {func.__name__} после {elapsed:.2f} секунд: {e}")
            raise
    return wrapper

def init_project():
    # Получаем ключ из переменной окружения GOOGLE_API_KEY
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("GOOGLE_API_KEY не установлен в переменных окружения.")
        raise ValueError("GOOGLE_API_KEY не установлен.")
    return api_key

class Memory:
    def __init__(self):
        self.versions = []  # Список (version_name, draft_text)
        self.feedback = []  # Список (role, feedback_text)

    def store_draft(self, draft: str, version_name: str):
        self.versions.append((version_name, draft))
        logging.info(f"Сохранён черновик: {version_name}")

    def store_feedback(self, feedback_text: str, role: str):
        self.feedback.append((role, feedback_text))
        logging.info(f"Сохранена обратная связь от {role}")

    def get_last_draft(self) -> str:
        if not self.versions:
            logging.warning("Нет сохранённых черновиков.")
            return ""
        last_draft = self.versions[-1][1]
        logging.info("Получен последний черновик.")
        return last_draft

class LangChainClient:
    """
    Обёртка вокруг LangChain, чтобы эмулировать метод generate_content(prompt).
    Внутри создаёт LLMChain для каждого вызова.
    """
    @log_and_time
    def __init__(self):
        # Инициируем чат-модель (можно заменить на Google PaLM, ChatGooglePalm и т.д.)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            max_tokens=2000
        )
        logging.info("Инициализирован LangChainClient с ChatGoogleGenerativeAI.")

        # ### Добавлено для подсчёта токенов ###
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.total_tokens_used = 0
        self.call_count = 0
        # #####################################

    @log_and_time
    def generate_content(self, prompt: str) -> str:
        """
        Синхронно вызываем LLMChain, возвращаем строку с ответом.
        """
        # Создаём шаблон; для простоты используем весь prompt как единственный переменную {user_input}
        prompt_template = PromptTemplate(
            template="{user_input}",
            input_variables=["user_input"],
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template
        )
        # Запускаем
        result = chain.run(user_input=prompt)
        logging.debug(f"Сгенерированный контент: {result}")

        # ### Считаем и логируем расход токенов (приблизительно) ###
        prompt_tokens = len(self.tokenizer.encode(prompt))
        result_tokens = len(self.tokenizer.encode(result))
        step_usage = prompt_tokens + result_tokens

        self.total_tokens_used += step_usage
        self.call_count += 1
        logging.info(
            f"Token usage for call {self.call_count}: "
            f"{step_usage} tokens (prompt: {prompt_tokens}, output: {result_tokens}). "
            f"Total so far: {self.total_tokens_used} tokens."
        )
        # #########################################################

        return result

class StudentAgent:
    def __init__(self, llm_client: LangChainClient):
        self.llm_client = llm_client
        logging.info("Инициализирован StudentAgent.")

    @log_and_time
    def produce_draft(self, plan: str, topic: str, constraints: str) -> str:
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
Ни в коем случае не нарушай ограничения, так как это приведет к ухудшению качества статьи.
"""
        logging.info("Генерация первого черновика статьи.")
        draft = self.llm_client.generate_content(prompt)
        logging.info("Черновик статьи сгенерирован.")
        return draft

    @log_and_time
    def rewrite_draft(self, plan: str, topic: str, constraints: str, current_text: str, feedback: str) -> str:
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
            А также ни в коем случае не нарушай ограничения, так как это приведет к ухудшению качества статьи.
            """
        logging.info("Переписывание черновика статьи с учётом замечаний.")
        new_draft = self.llm_client.generate_content(prompt)
        logging.info("Переписанный черновик статьи сгенерирован.")
        return new_draft

class CriticAgent:
    def __init__(self, llm_client: LangChainClient):
        self.llm_client = llm_client
        logging.info("Инициализирован CriticAgent.")

    @log_and_time
    def criticize_draft(self, draft_text: str) -> str:
        prompt = f"""
Выступи в роли Критика. Ниже приводится черновик научной статьи:
{draft_text}
Укажи слабые стороны, неточности в фактах, логические несостыковки, методологические ошибки.
Дай рекомендации, как улучшить статью, чтобы она соответствовала требованиям научного журнала.
"""
        logging.info("Критика черновика статьи.")
        critical_feedback = self.llm_client.generate_content(prompt)
        logging.info("Критика статьи сгенерирована.")
        return critical_feedback

class SupervisorAgent:
    def __init__(self, llm_client: LangChainClient):
        self.llm_client = llm_client
        logging.info("Инициализирован SupervisorAgent.")

    @log_and_time
    def give_plan(self, topic: str, count_word: str) -> str:
        prompt = f"""
Представь, что ты научный руководитель. Составь план статьи для студента.
Тема: {topic}
Объём: {count_word} слов

Структура плана: аннотация, введение, основная часть (теория + практика), заключение.
"""
        logging.info("Создание плана статьи.")
        plan = self.llm_client.generate_content(prompt)
        logging.info("План статьи сгенерирован.")
        return plan

    @log_and_time
    def evaluate_draft(self, draft_text: str) -> str:
        prompt = f"""
Выступи в роли Научного руководителя. Вот черновик статьи:
{draft_text}
Дай оценку её структуре, содержанию и качеству. Укажи конкретные рекомендации по улучшению.
"""
        logging.info("Оценка черновика статьи.")
        feedback = self.llm_client.generate_content(prompt)
        logging.info("Оценка статьи сгенерирована.")
        return feedback

class OrchestratorGemini:
    def __init__(self, google_api_key: str):
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Создаём клиента LangChain
        self.llm_client = LangChainClient()

        self.supervisor = SupervisorAgent(self.llm_client)
        self.student = StudentAgent(self.llm_client)
        self.critic = CriticAgent(self.llm_client)
        self.memory = Memory()
        logging.info("OrchestratorGemini инициализирован.")

    @log_and_time
    def multi_iter_workflow(self, topic: str, count_word: str, constraints: str, max_iterations=2):
        logging.info("Начало многоитерационного рабочего процесса.")
        plan = self.supervisor.give_plan(topic, count_word)

        # 1. Первая версия черновика
        draft_v1 = self.student.produce_draft(plan, topic, constraints)
        self.memory.store_draft(draft_v1, "Draft_v1")

        # 2. Критика/оценка
        critique_v1 = self.critic.criticize_draft(draft_v1)
        supervisor_v1 = self.supervisor.evaluate_draft(draft_v1)

        combined_feedback_v1 = critique_v1 + "\n\n" + supervisor_v1
        self.memory.store_feedback(combined_feedback_v1, "Critic+Supervisor")

        # Итерационный процесс
        for i in range(1, max_iterations):
            logging.info(f"Итерация {i + 1} начата.")
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

            combined_feedback_v1 = new_critique + "\n\n" + new_supervisor
            self.memory.store_feedback(combined_feedback_v1, "Critic+Supervisor")
            logging.info(f"Итерация {i + 1} завершена.")

        logging.info("Многоитерационный рабочий процесс завершён.")

        # ### Логируем общее количество токенов ###
        total_usage = self.llm_client.total_tokens_used
        logging.info(f"Общее число (приблизительное) использованных токенов: {total_usage}")
        # ##########################################

        return {
            "plan": plan,
            "all_versions": self.memory.versions,
            "final_version": self.memory.get_last_draft(),
            "feedback_history": self.memory.feedback,
            "tokens_used_approx": total_usage  # <-- по желанию возвращаем в ответ
        }

# Асинхронная обёртка
async def main():
    try:
        logging.info("Запуск главной функции.")
        google_api_key = init_project()  # грузим ключ из .env или ещё откуда-то
        orchestrator = OrchestratorGemini(google_api_key)

        topic = "Проблема дифференциации виновного и безвиновного деяния в гражданском праве"
        count_word = "1000"
        constraints = "Строгая формальная лексика, упоминание реальных исследований, максимальное отклонение от заданного количества слов - 50"

        start_time = time.time()
        results = orchestrator.multi_iter_workflow(topic, count_word, constraints, max_iterations=2)
        end_time = time.time()
        elapsed = end_time - start_time
        logging.info(f"Рабочий процесс завершён за {elapsed:.2f} секунд.")

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

        # Добавляем информацию о токенах
        md_content.append("\n## Счётчик использованных токенов (приблизительно)\n")
        md_content.append(f"Итого использовано ~{results['tokens_used_approx']} токенов.\n")

        # 2) Сохраняем всё в Markdown-файл
        output_filename = "article_results.md"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))

        logging.info(f"Результаты записаны в файл: {output_filename}")

    except Exception == 'ValueError' as e:
        logging.exception("Произошла непредвиденная ошибка в главной функции.")

if __name__ == '__main__':
    asyncio.run(main())