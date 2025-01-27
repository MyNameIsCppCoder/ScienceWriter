import os
import asyncio
import time
import logging
import json

from dotenv import load_dotenv
# LangChain: Chat-модель + базовые утилиты
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Для подсчёта токенов (приблизительно)
from transformers import GPT2TokenizerFast

load_dotenv()

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
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("GOOGLE_API_KEY не установлен в переменных окружения.")
        raise ValueError("GOOGLE_API_KEY не установлен.")
    return api_key

class Memory:
    """
    Хранит все версии текстов и обратную связь.
    versions: [(version_name, text), ...]
    feedback: [(role, feedback_text), ...]
    """
    def __init__(self):
        self.versions = []
        self.feedback = []

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
    Обёртка вокруг LangChain, упрощённо: для каждого generate_content делаем LLMChain и вызываем его.
    """
    @log_and_time
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            max_tokens=2000
        )
        logging.info("Инициализирован LangChainClient с ChatGoogleGenerativeAI.")
        # Инициализируем токенизатор (примерно для подсчёта токенов)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.total_tokens_used = 0
        self.call_count = 0

    @log_and_time
    def generate_content(self, prompt: str) -> str:
        prompt_template = PromptTemplate(
            template="{user_input}",
            input_variables=["user_input"],
        )
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template
        )
        result = chain.run(user_input=prompt)
        logging.debug(f"Сгенерированный контент: {result}")

        # Считаем токены (приблизительно)
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
        return result

class StudentAgent:
    """
    Псевдоним для "писателя" (AI-писатель).
    """
    def __init__(self, llm_client: LangChainClient):
        self.llm_client = llm_client
        logging.info("Инициализирован StudentAgent.")

    @log_and_time
    def produce_section_draft(self, section_title: str, topic: str, constraints: str) -> str:
        """
        Генерация черновика конкретной секции статьи.
        """
        prompt = f"""
Ты выступаешь в роли Студента (автора). 
Тема статьи: {topic}
Ограничения: {constraints}

Название секции: {section_title}

Напиши развёрнутый черновик текста для этой секции. 
Помни об ограничениях и научном стиле.
"""
        logging.info(f"Генерация черновика для секции: {section_title}")
        draft = self.llm_client.generate_content(prompt)
        logging.info("Черновик секции сгенерирован.")
        return draft

    @log_and_time
    def rewrite_section_draft(self, section_title: str, topic: str, constraints: str, current_text: str, feedback: str) -> str:
        """
        Переписывает черновик секции с учётом замечаний критика.
        """
        prompt = f"""
Ты - Студент (автор). 
Текущая версия секции "{section_title}":
{current_text}

Полученные замечания (критика):
{feedback}

Тема статьи: {topic}
Ограничения: {constraints}

Перепиши текст секции, устранив указанные проблемы.
Сохрани общий смысл и стиль, не нарушай ограничения.
"""
        logging.info(f"Переписывание секции {section_title} с учётом критики.")
        new_draft = self.llm_client.generate_content(prompt)
        logging.info("Секция переписана с учётом замечаний.")
        return new_draft

class CriticAgent:
    """
    Критик (AI-критик), анализирует предоставленный текст и даёт замечания.
    """
    def __init__(self, llm_client: LangChainClient):
        self.llm_client = llm_client
        logging.info("Инициализирован CriticAgent.")

    @log_and_time
    def criticize_section(self, draft_text: str, section_title: str) -> str:
        prompt = f"""
Выступаешь в роли Критика. 
Секция статьи: {section_title}

Ниже черновик текста секции:
{draft_text}

Укажи все слабые места, неточности, логические ошибки, стилистические проблемы.
Дай конкретные рекомендации по улучшению, но без переписывания всей секции.
"""
        logging.info(f"Критика секции: {section_title}")
        critical_feedback = self.llm_client.generate_content(prompt)
        logging.info("Сгенерирована критика текущего черновика.")
        return critical_feedback

class SupervisorAgent:
    """
    Научный руководитель (AI), даёт общий план статьи в формате JSON.
    """
    def __init__(self, llm_client: LangChainClient):
        self.llm_client = llm_client
        logging.info("Инициализирован SupervisorAgent.")

    @log_and_time
    def give_plan(self, topic: str, count_word: str) -> dict:
        """
        Генерация высокого уровня плана статьи в формате JSON.
        Пример ожидаемого JSON:
        {
          "plan_title": "Название плана",
          "sections": [
            {
              "title": "Введение",
              "summary": "2-3 предложения с описанием",
              "recommended_word_count": 200
            },
            ...
          ]
        }
        """
        prompt = f"""
Ты - Научный руководитель. 
Тема статьи: {topic}
Примерный общий объём статьи: {count_word} слов.

Составь подробный план статьи. 
Сначала определи, как распределить слова по секциям, чтобы суммарное количество слов примерно соответствовало {count_word}.

Верни результат СТРОГО в формате JSON без дополнительных пояснений. 
Формат должен быть таким (данные после ключа - для примера):

{{
  "plan_title": "Статья о ...",
  "sections": [
    {{
      "title": "Введение",
      "summary": "Краткое описание, 1-2 предложения",
      "recommended_word_count": 200
    }},
    {{
      "title": "Основная часть",
      "summary": "Основные теоретические и практические аспекты ...",
      "recommended_word_count": 600
    }},
    {{
      "title": "Заключение",
      "summary": "Выводы ...",
      "recommended_word_count": 200
    }}
  ]
}}
Важно: Не добавляй никаких полей, кроме указанных выше. Не пиши текст вокруг JSON. Только сам JSON.
"""
        logging.info("Генерация плана статьи в формате JSON.")
        plan_json_str_unformatted = self.llm_client.generate_content(prompt)
        plan_json_str = plan_json_str_unformatted.replace('```', '').replace('json', '')
        logging.info('--'*20)
        logging.info(f"Получен сырой JSON-план: {plan_json_str}")
        logging.info('--'*20)

        # Парсим JSON в Python-словарь
        try:
            plan_dict = json.loads(plan_json_str)
            return plan_dict
        except json.JSONDecodeError as e:
            logging.error(f"Ошибка парсинга JSON-плана: {e}")
            raise ValueError("LLM вернул невалидный JSON. Попробуйте повторить запрос.")

############################################################
# Orchestrator: основной класс, который управляет процессом
############################################################
class OrchestratorGemini:
    def __init__(self, google_api_key: str):
        os.environ["GOOGLE_API_KEY"] = google_api_key
        self.llm_client = LangChainClient()

        self.supervisor = SupervisorAgent(self.llm_client)
        self.student = StudentAgent(self.llm_client)
        self.critic = CriticAgent(self.llm_client)
        self.memory = Memory()

        logging.info("OrchestratorGemini инициализирован.")

    @log_and_time
    def run_workflow(self, topic: str, count_word: str, constraints: str, max_section_iterations=2):
        """
        Полный процесс:
        1. Получение плана у SupervisorAgent (JSON)
        2. Итерация по sections (название + summary + recommended_word_count)
        3. Для каждой секции:
           - Создать черновик
           - Критика
           - Переписать с учётом критики
        4. Объединить финальные тексты секций в один текст
        """
        # 1) Генерируем общий план (JSON -> dict)
        plan_dict = self.supervisor.give_plan(topic, count_word)
        plan_title = plan_dict.get("plan_title", "Без названия")
        sections = plan_dict.get("sections", [])

        # Сохраняем исходный JSON-план в память (как строку, при желании)
        plan_json_str = json.dumps(plan_dict, ensure_ascii=False, indent=2)
        self.memory.store_draft(plan_json_str, "Plan_JSON")

        # 2) Проходимся по секциям
        final_sections_texts = []
        for idx, sec in enumerate(sections, start=1):
            section_title = sec["title"]
            # Можно при желании использовать sec["summary"], sec["recommended_word_count"] и т.д.

            logging.info(f"Начинаем работу над секцией {idx}: {section_title}")

            # Генерируем черновик
            draft = self.student.produce_section_draft(
                section_title=section_title,
                topic=topic,
                constraints=constraints
            )
            version_name = f"Section_{idx}_Draft_v1"
            self.memory.store_draft(draft, version_name)

            # Итерационный цикл для секции
            current_draft = draft
            for it in range(1, max_section_iterations+1):
                criticism = self.critic.criticize_section(current_draft, section_title)
                self.memory.store_feedback(criticism, f"Critic_Section_{idx}_Iteration_{it}")

                new_draft = self.student.rewrite_section_draft(
                    section_title=section_title,
                    topic=topic,
                    constraints=constraints,
                    current_text=current_draft,
                    feedback=criticism
                )
                version_name = f"Section_{idx}_Draft_v{it+1}"
                self.memory.store_draft(new_draft, version_name)
                current_draft = new_draft

            # Сохраняем итоговый вариант секции
            final_sections_texts.append(f"## {section_title}\n{current_draft}")

        # 4) Склеиваем секции
        final_text = f"# {plan_title}\n\n" + "\n\n".join(final_sections_texts)

        # Подсчёт токенов
        total_usage = self.llm_client.total_tokens_used
        logging.info(f"Общее число использованных токенов (приблизительно): {total_usage}")

        return {
            "plan_dict": plan_dict,
            "all_versions": self.memory.versions,
            "final_text": final_text,
            "feedback_history": self.memory.feedback,
            "tokens_used_approx": total_usage
        }

# Асинхронная обёртка
async def main():
    try:
        logging.info("Запуск главной функции.")
        google_api_key = init_project()
        orchestrator = OrchestratorGemini(google_api_key)

        topic = "Проблема дифференциации виновного и безвиновного деяния в гражданском праве"
        count_word = "1000"
        constraints = "Строгая формальная лексика, упоминание реальных исследований, отклонение от заданного объёма не более 50 слов"

        start_time = time.time()
        results = orchestrator.run_workflow(
            topic=topic,
            count_word=count_word,
            constraints=constraints,
            max_section_iterations=2
        )
        end_time = time.time()
        elapsed = end_time - start_time
        logging.info(f"Рабочий процесс завершён за {elapsed:.2f} секунд.")

        # Формируем содержимое Markdown-файла
        md_content = []
        md_content.append("# Итоги генерации научной статьи\n")

        # Сам JSON-план
        md_content.append("## Исходный JSON-план\n")
        plan_json_str = json.dumps(results["plan_dict"], ensure_ascii=False, indent=2)
        md_content.append(f"```json\n{plan_json_str}\n```\n")

        # Все версии черновиков
        md_content.append("## Все версии черновиков\n")
        for version_name, draft_text in results["all_versions"]:
            md_content.append(f"### {version_name}\n")
            md_content.append(f"{draft_text}\n")

        # Итоговый текст
        md_content.append("## Итоговый текст\n")
        md_content.append(results["final_text"])

        # История замечаний
        md_content.append("\n## История замечаний\n")
        for role, feedback_text in results["feedback_history"]:
            md_content.append(f"### {role}\n")
            md_content.append(f"{feedback_text}\n")

        # Подсчёт токенов
        md_content.append("\n## Счётчик использованных токенов (приблизительно)\n")
        md_content.append(f"Итого использовано ~{results['tokens_used_approx']} токенов.\n")

        # Сохраняем
        output_filename = "article_results.md"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))

        logging.info(f"Результаты записаны в файл: {output_filename}")

    except Exception as e:
        logging.exception("Произошла ошибка в main().")

if __name__ == '__main__':
    asyncio.run(main())