import os
import time
import ssl
import glob
from datetime import datetime
from pathlib import Path
from bing_image_downloader import downloader
import concurrent.futures

# --- КОНФИГУРАЦИЯ ---

# Список целевых запросов для скачивания (НОУТБУКИ, КРОМЕ MAC)
TARGET_QUERIES = [
    "Dell XPS 13 silver on desk",
    "HP Spectre x360 convertible mode",
    "Lenovo ThinkPad X1 Carbon business",
    "ASUS ROG Zephyrus G14 gaming setup",
    "Acer Swift 5 thin and light",
    "Microsoft Surface Laptop Studio open",
    "Razer Blade 15 gaming black",
    "Windows laptop modern setup",
    "Chromebook for student",
    "Laptop 2-in-1 in tent mode",
    "High performance laptop workstation",
    "Laptop keyboard close up professional",
    "Laptop with external monitor and keyboard",
    "Dell Latitude rugged laptop outside",
    "ASUS Zenbook OLED display",
    "HP EliteBook on coffee table",
    "Lenovo Legion 5 gaming laptop",
    "Silver ultrabook top view",
    "Thin and light laptop side view",
    "Open laptop on a wooden table interior",
]

N_IMAGES_PER_QUERY = 200
MAX_WORKERS = 10


def ensure_dir(path: Path):
    """Создает директорию, если она не существует."""
    path.mkdir(parents=True, exist_ok=True)


def download_images_for_query(query: str, root_dir: Path, n_images: int) -> dict:
    """
    Загружает изображения для одного запроса. Работает как целевая функция для потока.

    Возвращает словарь с результатом: {'query': str, 'status': str, 'count': int}
    """
    start_time = time.time()
    downloaded_count = 0
    # Создаем имя для потока, заменяя пробелы
    clean_query = query.replace(" ", "_")

    print(f"[Поток: {clean_query}] Начат запрос...")

    try:
        # Библиотека bing_image_downloader создаст подпапку с именем `query`
        # внутри переданного `output_dir` (root_dir).
        downloader.download(
            query=query,
            limit=n_images,
            output_dir=str(root_dir),
            adult_filter_off=True,
            force_replace=False,
            timeout=10,
            verbose=False  # Отключаем подробный вывод в потоке для чистоты консоли
        )

        # Фактический путь сохранения (созданный библиотекой)
        final_save_path = root_dir / query
        if final_save_path.exists():
            # Подсчитываем скачанные файлы по факту
            downloaded_count = len(list(final_save_path.glob('*')))

        end_time = time.time()

        print(f"[Поток: {clean_query}] Успех. Скачано: {downloaded_count} ({end_time - start_time:.2f} сек)")
        return {
            'query': query,
            'status': 'SUCCESS',
            'count': downloaded_count,
            'duration': end_time - start_time
        }

    except Exception as e:
        end_time = time.time()
        print(f"[Поток: {clean_query}] ОШИБКА: {e} ({end_time - start_time:.2f} сек)")
        return {
            'query': query,
            'status': f'ERROR: {e.__class__.__name__}',
            'count': 0,
            'duration': end_time - start_time
        }


def main():
    """Основная функция, реализующая многопоточную логику загрузки."""

    # --- 1. Инициализация и настройка директории ---
    project_name = "multi_query_non_mac_laptops"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Корневая директория для всех результатов
    unique_root_dir = Path(f"data/laptops/{project_name}-{timestamp}")
    ensure_dir(unique_root_dir)

    # --- Временный SSL-фикс для macOS/неполных сертификатов ---
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        print("Применен временный SSL-фикс для предотвращения ошибок CERTIFICATE_VERIFY_FAILED.")
    except AttributeError:
        pass
    # --- Конец SSL-фикса ---

    print("\n" + "=" * 60)
    print(f"СТАРТ МНОГОПОТОЧНОГО СКАЧИВАНИЯ")
    print(f"Запросов в очереди: {len(TARGET_QUERIES)}")
    print(f"Изображений на запрос: {N_IMAGES_PER_QUERY}")
    print(f"Одновременных потоков: {MAX_WORKERS}")
    print(f"Корневая папка сохранения: {unique_root_dir.resolve()}")
    print("=" * 60 + "\n")

    overall_start_time = time.time()
    results = []

    # --- 2. Многопоточная загрузка ---
    # ThreadPoolExecutor используется для управления пулом потоков
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Словарь для хранения Future-объектов, которые будут возвращать результаты
        future_to_query = {
            executor.submit(
                download_images_for_query, query, unique_root_dir, N_IMAGES_PER_QUERY
            ): query
            for query in TARGET_QUERIES
        }

        # Отслеживание завершения потоков по мере их выполнения
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # В случае критической ошибки пула потоков
                print(f"Критическая ошибка при обработке запроса {query}: {e}")
                results.append({'query': query, 'status': 'CRITICAL_ERROR', 'count': 0, 'duration': 0})

    overall_end_time = time.time()

    # --- 3. Итоговый отчет ---
    total_downloaded = sum(r['count'] for r in results)
    total_queries = len(TARGET_QUERIES)
    success_queries = sum(1 for r in results if r['status'] == 'SUCCESS')
    error_queries = total_queries - success_queries

    print("\n" + "=" * 60)
    print("ОБЩИЙ ОТЧЕТ О СКАЧИВАНИИ")
    print("=" * 60)
    print(f"Общее время выполнения: {overall_end_time - overall_start_time:.2f} секунд")
    print(f"Всего обработано запросов: {total_queries}")
    print(f"Успешно завершено: {success_queries}")
    print(f"Завершено с ошибками: {error_queries}")
    print(f"ВСЕГО СКАЧАНО ИЗОБРАЖЕНИЙ: {total_downloaded}")
    print("=" * 60)

    # Детальный вывод результатов
    print("\nДЕТАЛЬНЫЙ ОТЧЕТ ПО ЗАПРОСАМ:")
    for result in results:
        print(f"  [{result['status']}] {result['query']}: {result['count']} файлов ({result['duration']:.2f} сек)")


if __name__ == "__main__":
    main()
