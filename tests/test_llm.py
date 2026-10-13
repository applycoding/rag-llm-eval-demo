import pytest

from ragqa.llm import (
    ABSTENTION,
    AnthropicChatLLM,
    ExtractiveLLM,
    OpenAIChatLLM,
    resolve_llm_name,
)
from ragqa.prompts import render_prompt

TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n"


def _prompt(context: str, question: str) -> str:
    return render_prompt(TEMPLATE, context=context, question=question)


def test_extractive_reader_copies_the_supporting_sentence() -> None:
    context = (
        "The pilot skiff waits at the buoy. "
        "The Meridian lighthouse lantern was first lit on March 12, 1886."
    )
    answer = ExtractiveLLM().generate(
        _prompt(context, "When was the Meridian lighthouse lantern first lit?")
    )
    assert answer == "The Meridian lighthouse lantern was first lit on March 12, 1886."


def test_extractive_reader_ignores_citation_labels() -> None:
    context = "[1] (added).\nThe night clerk keeps a brass key stamped with the number 77."
    answer = ExtractiveLLM().generate(_prompt(context, "What number is stamped on the brass key?"))
    assert answer == "The night clerk keeps a brass key stamped with the number 77."


def test_extractive_reader_abstains_without_overlap() -> None:
    answer = ExtractiveLLM().generate(
        _prompt("Pilots board at the outer buoy.", "What is the serial number of the diaphone?")
    )
    assert answer == ABSTENTION


def test_llm_auto_uses_extractive_without_keys() -> None:
    assert resolve_llm_name("auto", openai_key=None, anthropic_key=None) == "extractive"
    assert resolve_llm_name("auto", openai_key="sk-test", anthropic_key=None) == "openai"
    assert resolve_llm_name("auto", openai_key=None, anthropic_key="sk-ant") == "anthropic"


def test_hosted_llm_requires_a_key() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        resolve_llm_name("openai", openai_key=None, anthropic_key=None)


def test_openai_chat_sends_the_prompt() -> None:
    captured: dict[str, object] = {}

    class Completions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            message = type("Message", (), {"content": "from openai"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    assert OpenAIChatLLM(client, model="gpt-4o-mini").generate("hello") == "from openai"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_chat_joins_text_blocks() -> None:
    captured: dict[str, object] = {}

    class Messages:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            blocks = [type("Block", (), {"text": "from "})(), type("Block", (), {"text": "anthropic"})()]
            return type("Response", (), {"content": blocks})()

    client = type("Client", (), {"messages": Messages()})()
    assert AnthropicChatLLM(client, model="claude-3-5-haiku-latest").generate("hello") == "from anthropic"
    assert captured["model"] == "claude-3-5-haiku-latest"
    assert captured["max_tokens"] == 800
