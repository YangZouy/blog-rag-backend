from api.models import ConversationTurn
from core.conversation import is_referential_follow_up, prepare_query


def test_independent_question_skips_rewrite(monkeypatch):
    def fail_if_called():
        raise AssertionError("independent questions must not call an LLM")

    monkeypatch.setattr("core.conversation.get_classify_llm", fail_if_called)
    query, rewritten = prepare_query("什么是混合检索？", [ConversationTurn(content="你好")])
    assert query == "什么是混合检索？"
    assert rewritten is False


def test_referential_follow_up_is_detected():
    history = [ConversationTurn(content="你的 RAG 为什么使用混合检索？")]
    assert is_referential_follow_up("那为什么还要加重排？", history)


def test_referential_follow_up_uses_recent_history(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            assert "你的 RAG 为什么使用混合检索？" in messages[1]["content"]
            return type("Response", (), {"content": "博客 RAG 在混合检索之后为什么还要加入重排？"})()

    monkeypatch.setattr("core.conversation.get_classify_llm", lambda: FakeLLM())
    query, rewritten = prepare_query(
        "那为什么还要加重排？",
        [ConversationTurn(content="更早的问题"), ConversationTurn(content="你的 RAG 为什么使用混合检索？")],
    )
    assert rewritten is True
    assert query == "博客 RAG 在混合检索之后为什么还要加入重排？"
