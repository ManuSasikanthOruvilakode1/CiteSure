import pytest
from conftest import ScriptedLLM

from citesure.config import EvaluationConfig
from citesure.evaluation.metrics import (
    AnswerQuestions,
    Claims,
    ClaimVerdict,
    ContextRelevanceJudgement,
    FaithfulnessJudgement,
    PassageGrade,
    TriadEvaluator,
    retrieval_hit_metrics,
)
from citesure.evaluation.runner import ModeReport, check_gate
from citesure.generation.generator import Generator, extract_citations
from citesure.generation.prompts import REFUSAL_MESSAGE
from citesure.types import ScoredChunk


def _contexts(chunks, n=2):
    return [ScoredChunk(c, 1.0) for c in chunks[:n]]


def test_faithfulness_is_supported_fraction(chunks, embedder):
    judge = ScriptedLLM(structured={
        "Claims": Claims(claims=["a", "b", "c", "d"]),
        "FaithfulnessJudgement": FaithfulnessJudgement(verdicts=[
            ClaimVerdict(claim="a", supported=True),
            ClaimVerdict(claim="b", supported=True),
            ClaimVerdict(claim="c", supported=True),
            ClaimVerdict(claim="d", supported=False),
        ]),
    })
    score, n, unsupported = TriadEvaluator(judge, embedder).faithfulness("ans", _contexts(chunks))
    assert score == 0.75 and n == 4 and unsupported == ["d"]


def test_faithfulness_with_no_claims_is_perfect(chunks, embedder):
    judge = ScriptedLLM(structured={"Claims": Claims(claims=[])})
    assert TriadEvaluator(judge, embedder).faithfulness(REFUSAL_MESSAGE, _contexts(chunks))[0] == 1.0


def test_context_relevance_counts_missing_grades_as_zero(chunks, embedder):
    judge = ScriptedLLM(structured={
        "ContextRelevanceJudgement": ContextRelevanceJudgement(grades=[PassageGrade(passage=1, score=3)])
    })
    score = TriadEvaluator(judge, embedder).context_relevance("q", _contexts(chunks, 2))
    assert score == pytest.approx(0.5)


def test_answer_relevance_high_for_matching_questions(embedder):
    judge = ScriptedLLM(structured={
        "AnswerQuestions": AnswerQuestions(noncommittal=False, questions=["how do I rotate api keys"] * 3)
    })
    assert TriadEvaluator(judge, embedder).answer_relevance("how do I rotate api keys", "x") > 0.99


def test_answer_relevance_zero_when_noncommittal(embedder):
    judge = ScriptedLLM(structured={"AnswerQuestions": AnswerQuestions(noncommittal=True, questions=["q"])})
    assert TriadEvaluator(judge, embedder).answer_relevance("q", REFUSAL_MESSAGE) == 0.0


def test_retrieval_hit_metrics():
    m = retrieval_hit_metrics([["a", "b"], ["c", "d"], ["e"]], ["b", "c", "z"])
    assert m["hit_rate"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_gate_fails_on_low_faithfulness():
    report = ModeReport("hybrid_rerank", 10, 0.9, 0.80, 0.95, 1.0, 2.0)
    failures = check_gate(report, EvaluationConfig())
    assert len(failures) == 1 and failures[0].startswith("faithfulness")


def test_gate_passes():
    assert check_gate(ModeReport("hybrid_rerank", 10, 0.9, 0.95, 0.95, 1.0, 2.0), EvaluationConfig()) == []


def test_generator_refuses_without_context():
    llm = ScriptedLLM(text="should not be called")
    assert Generator(llm).generate("q", []) == REFUSAL_MESSAGE
    assert llm.calls == []


def test_generator_prompt_contains_numbered_passages(chunks):
    llm = ScriptedLLM(text="Rotate keys in the dashboard [1].")
    answer = Generator(llm).generate("how to rotate?", _contexts(chunks))
    assert answer == "Rotate keys in the dashboard [1]."
    prompt = llm.calls[0][1]
    assert "[1] source: auth.md" in prompt and "[2] source: rate.md" in prompt


def test_extract_citations_ignores_out_of_range():
    assert extract_citations("a [1] b [3][1] c [9]", num_contexts=3) == [1, 3]
