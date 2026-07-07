# AI Agent — Autonomous Agent with Plugin Architecture

Модульный AI-агент с автономными возможностями, изолированной архитектурой и системой плагинов.

## Архитектура

```
ai-agent/
├── pyproject.toml              # Управление зависимостями по группам
├── core/                       # Ядро системы (без внешних зависимостей)
│   ├── interfaces.py           # Абстрактные интерфейсы
│   ├── event_bus.py            # Шина событий (изоляция модулей)
│   ├── plugin_manager.py       # Менеджер плагинов (hot-loading)
│   ├── registry.py             # Реестр инструментов
│   ├── container.py            # DI-контейнер
│   ├── agent.py                # AgentCore (координатор)
│   └── cli.py                  # CLI-интерфейс
├── adapters/                   # Адаптеры внешних сервисов
│   └── ollama_adapter.py       # Адаптер Ollama (LLM)
├── modules/                    # Опциональные модули
│   ├── telegram_bot.py         # Telegram-транспорт
│   └── memory.py               # Система памяти (SQLite/in-memory)
├── plugins/                    # Плагины
│   └── example/                # Пример плагина
├── tests/                      # Тесты изоляции
└── docs/                       # Документация
```

## Принципы архитектуры

### 1. Полная изоляция модулей
- Каждый модуль — независимая единица
- Модули общаются только через **EventBus** (шаблон Pub/Sub)
- Ошибка в одном модуле **не останавливает** всю систему
- Циклические зависимости **запрещены**

### 2. Dependency Injection
- Все зависимости передаются через конструктор
- DI-контейнер на базе `injector`
- Модули не создают зависимости сами

### 3. Адаптеры для внешних сервисов
- Ollama изолирован в `adapters/ollama_adapter.py`
- Другие модули используют только интерфейс `LLMProvider`
- Добавление нового LLM — создание нового адаптера без изменения кода

### 4. Опциональные зависимости
- `core` — минимальный набор (pydantic, httpx, structlog)
- `telegram` — python-telegram-bot
- `web` — FastAPI + uvicorn
- `vision` — Pillow
- `voice` — SpeechRecognition + pydub
- `test` — pytest + фикстуры
- `dev` — ruff + mypy + pre-commit

### 5. Плагинная система
- Плагины загружаются из директорий
- Каждый плагин получает **изолированный контекст**:
  - Собственное пространство имён для событий
  - Изолированный логгер
  - Собственный конфиг
- Плагины регистрируют инструменты через `ToolRegistry`
- **Hot-reload** без перезапуска системы

## Установка

### Базовая установка (только ядро)
```bash
pip install -e .
```

### С Telegram
```bash
pip install -e ".[telegram]"
```

### Полная установка (все модули)
```bash
pip install -e ".[all]"
```

### Разработка
```bash
pip install -e ".[full]"
```

## Запуск

### Предварительные требования
- Python 3.11+
- Ollama с запущенной моделью (например, `llama3.2`)

### Базовый запуск
```bash
# Запуск с настройками по умолчанию
ai-agent

# С указанием модели
ai-agent --model llama3.2 --ollama-url http://localhost:11434

# С Telegram
ai-agent --telegram-token YOUR_BOT_TOKEN

# Полная конфигурация
ai-agent \
    --model llama3.2 \
    --ollama-url http://localhost:11434 \
    --memory sqlite \
    --telegram-token YOUR_TOKEN \
    --log-level INFO
```

### Запуск тестов
```bash
# Все тесты
pytest

# С изоляцией процессов
pytest -n auto

# С покрытием
pytest --cov=core --cov=adapters --cov=modules
```

## Создание плагина

### Минимальный плагин
```python
# plugins/my_plugin/__init__.py
from core.interfaces import PluginContext, Tool, ToolDefinition, ToolResult

METADATA = {
    "name": "my_plugin",
    "version": "1.0.0",
    "description": "My awesome plugin",
}

class MyTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="my_tool",
            description="Does something useful",
        )

    async def execute(self, **params) -> ToolResult:
        return ToolResult(success=True, content="Done!")

class MyPlugin:
    @property
    def metadata(self):
        from core.interfaces import PluginMetadata
        return PluginMetadata(**METADATA)

    async def initialize(self, context: PluginContext) -> None:
        context.register_tool(MyTool())

    async def shutdown(self) -> None:
        pass
```

### События в плагине
```python
async def initialize(self, context: PluginContext) -> None:
    # Подписка на события (автоматически в namespace плагина)
    context.event_bus.on("startup")(self._on_startup)

    # Публикация событий
    await context.event_bus.emit("ready", {"status": "ok"})
```

## Добавление нового LLM-провайдера

Создайте адаптер, реализующий интерфейс:

```python
# adapters/openai_adapter.py
from core.interfaces import LLMProvider, Conversation, LLMResponse

class OpenAIProvider:
    @property
    def name(self) -> str:
        return "openai:gpt-4"

    @property
    def is_available(self) -> bool:
        # Проверка доступности
        return True

    async def generate(self, conversation: Conversation, **kwargs) -> LLMResponse:
        # Реализация генерации
        return LLMResponse(content="Hello!", model="gpt-4")

    async def stream(self, conversation: Conversation, **kwargs):
        # Реализация стриминга
        yield StreamChunk(content="Hello!")
```

## Проверка архитектуры

Тесты проверяют:
- ✅ Изоляцию ошибок (падающий обработчик не ломает другие)
- ✅ Изоляцию плагинов (падающий плагин не ломает систему)
- ✅ Изоляцию инструментов (падающий инструмент не ломает другие)
- ✅ Изоляцию событий (события плагинов не пересекаются)
- ✅ Отсутствие циклических импортов
- ✅ Независимость модулей
- ✅ Конкурентное выполнение

## Лицензия

MIT
