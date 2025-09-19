import argparse
import os

import torch
from transformers import logging
from transformers import AutoTokenizer
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration


logging.set_verbosity_error()

class TagGenerator:
    def __init__(self, model_name: str = "trl-algo/summary_tags_qwen2_v2"):
        """
        Инициализация модели и токенизатора.

        Args:
            model_name (str, optional): Название модели на Hugging Face Hub.
                По умолчанию "trl-algo/summary_tags_qwen2_v2".

        Raises:
            RuntimeError: Если не удалось загрузить модель или токенизатор.
        """

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке модели {model_name}: {e}")

    def generate_tags(self, text: str, max_new_tokens: int = 128) -> str:
        """
        Генерация тегов для текста.

        Args:
            text (str): Исходный текст, для которого нужно сгенерировать теги.
            max_new_tokens (int, optional): Максимальное количество новых токенов
                в ответе. По умолчанию 128.

        Returns:
            str: Сырые теги в виде строки (например, "кот, собака, питомцы").

        Raises:
            RuntimeError: Если произошла ошибка при генерации тегов.
        """

        prompt = f"""<|im_start|>user
Сгенерируй список из минимум 3 тегов для следующего текста.
Теги должны быть короткими (одно слово или максимум два), без повторов.
Формат ответа: только список через запятую.
Текст: {text}<|im_end|>
<|im_start|>assistant
"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return generated_text
        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации тегов: {e}")

    @staticmethod
    def clean_tags(generated_text: str):
        """
        Приведение списка тегов к нормальному виду.

        Операции:
        - удаление дубликатов,
        - удаление пробелов,
        - удаление символа `#` в начале тегов.

        Args:
            generated_text (str): Сырые теги в виде строки, полученные от модели.

        Returns:
            list[str]: Список уникальных тегов в алфавитном порядке.
        """
        tags = [t.strip().lstrip("#") for t in generated_text.split(",") if t.strip()]
        tags = sorted(set(tags))
        return tags


def run(text: str = None, file: str = None):
    """
    Основная функция запуска генерации тегов.

    Принимает текст напрямую или читает его из файла.

    Args:
        text (str, optional): Исходный текст для генерации тегов.
        file (str, optional): Путь к файлу с текстом.

    Raises:
        FileNotFoundError: Если указанный файл не найден.
        ValueError: Если не передан ни текст, ни файл.
    """

    if file:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Файл {file} не найден")
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

    if not text:
        raise ValueError("Нужно передать либо текст (--text), либо путь к файлу (--file)")

    generator = TagGenerator()
    raw_tags = generator.generate_tags(text)
    clean = generator.clean_tags(raw_tags)

    print("Сырые теги:\n", raw_tags, "\n")
    print("Очищенные теги:\n", clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Генерация тегов для текста при помощи Qwen2 модели.",
        epilog="Примеры использования:\n"
               "  python tagger.py --text 'Кот играет с собакой'\n"
               "  python tagger.py --file story.txt",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--text", type=str,
        help="Текст для генерации тегов"
    )
    parser.add_argument(
        "--file", type=str,
        help="Файл с текстом для генерации тегов"
    )
    args = parser.parse_args()

    run(text=args.text, file=args.file)