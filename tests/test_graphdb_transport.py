import pytest

from kgr4vrd.adapters.graphdb import GraphDB, ReasonerError, Response, inconsistency_reason

CONFIG = {
    "endpoint": "http://localhost:7200",
    "repository": "dedicated-test",
    "ruleset": "owl2-rl",
    "expected_version": "test",
}


@pytest.mark.parametrize(
    "status,attempts", [(503, 3), (502, 3), (401, 1), (403, 1), (404, 1), (500, 1)]
)
def test_http_failure_never_becomes_a_logical_decision(status, attempts):
    calls, delays = [], []

    def transport(request, timeout):
        calls.append(request)
        assert timeout == 30
        return Response(status, b"unrelated server or authentication error")

    client = GraphDB(CONFIG, transport=transport, sleep=delays.append)
    with pytest.raises(ReasonerError):
        client._request("GET", client.url, allow_inconsistent=True)
    assert len(calls) == attempts
    assert delays == ([1, 2] if attempts == 3 else [])


def test_timeout_and_retry_after_are_bounded():
    calls = []

    def timeout(request, seconds):
        calls.append(seconds)
        raise ReasonerError("timeout", retryable=True)

    client = GraphDB(CONFIG, transport=timeout, sleep=lambda _: None)
    with pytest.raises(ReasonerError):
        client.ask("ASK {}")
    assert len(calls) == 3
    responses = iter([Response(429, headers={"Retry-After": "60"})])
    client = GraphDB(
        CONFIG,
        transport=lambda *_: next(responses),
        sleep=lambda _: pytest.fail("retried too soon"),
    )
    with pytest.raises(ReasonerError, match="Retry-After"):
        client.ask("ASK {}")


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"boolean":"false"}'])
def test_malformed_response_is_not_compatibility(body):
    client = GraphDB(CONFIG, transport=lambda *_: Response(200, body))
    with pytest.raises(ReasonerError, match="Malformed"):
        client.ask("ASK {}")


def test_verified_signature_only_and_cleanup_before_mutation_retry():
    unrelated = Response(500, b"Failed to load something")
    assert inconsistency_reason(unrelated) is None
    logical = Response(
        500, b"com.ontotext.trree.sdk.ConsistencyException: consistency rule cax-dw failed"
    )
    assert "cax-dw" in inconsistency_reason(logical)
    calls = []
    responses = iter([Response(503), Response(204)])

    def transport(*_):
        calls.append("request")
        return next(responses)

    client = GraphDB(CONFIG, transport=transport, sleep=lambda _: calls.append("delay"))
    assert (
        client._request("POST", client.url, cleanup=lambda: calls.append("cleanup")).status == 204
    )
    assert calls == ["request", "cleanup", "delay", "request"]
    client = GraphDB(CONFIG, transport=lambda *_: Response(503), sleep=lambda _: None)
    with pytest.raises(ReasonerError, match="cleanup"):
        client._request("POST", client.url)
    with pytest.raises(ReasonerError, match="not owned"):
        client.cleanup("urn:someone-elses-graph")
