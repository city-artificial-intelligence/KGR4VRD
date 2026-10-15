"""Bounded GraphDB preparation queries; HTTP failure is never a logical result.

Requires an exclusive, dedicated repository using owl2-rl consistency checking.
Only contexts created by this handle are removed. No repository is cleared.
"""

import base64
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen

from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef

from ..types import Decision


class ReasonerError(RuntimeError):
    """Operational failure; deliberately distinct from incompatibility."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes = b""
    headers: dict | None = None


def http_request(request: Request, timeout: float) -> Response:
    try:
        with urlopen(request, timeout=timeout) as response:
            return Response(response.status, response.read(), dict(response.headers))
    except HTTPError as error:
        return Response(error.code, error.read(), dict(error.headers))
    except (TimeoutError, URLError, OSError) as error:
        raise ReasonerError("GraphDB connection failed or timed out", retryable=True) from error


def inconsistency_reason(response: Response) -> str | None:
    """Accept the GraphDB consistency exception signature, never HTTP 500 alone."""
    text = response.body.decode("utf-8", errors="replace")
    if response.status != 500 or not re.search(
        r"(?:com\.ontotext\.[\w.]+\.)?ConsistencyException\s*:", text
    ):
        return None
    # The real fixture verifies this signature before any production query.
    reason = text.split("ConsistencyException", 1)[1].lstrip(": ").strip()
    return reason[:2000] if reason and re.search(r"consisten|rule|disjoint", reason, re.I) else None


class GraphDB:
    def __init__(
        self, config: dict, *, transport: Callable = http_request, sleep: Callable = time.sleep
    ):
        self.config = dict(config)
        self.endpoint = config["endpoint"].rstrip("/")
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.username
            or parsed.password
            or parsed.query
        ):
            raise ValueError(
                "GraphDB endpoint must be HTTP(S), with credentials only in environment variables"
            )
        self.repository = config["repository"]
        self.url = self.endpoint + "/repositories/" + quote(self.repository, safe="")
        self.transport, self.sleep = transport, sleep
        self.timeout = float(config.get("timeout_seconds", 30))
        self.attempts = int(config.get("max_attempts", 3))
        if not 0 < self.timeout <= 30 or not 1 <= self.attempts <= 3:
            raise ValueError("GraphDB permits at most 3 attempts and 30 seconds per attempt")
        self.headers = {"Accept": "application/json"}
        if config.get("username_env") or config.get("password_env"):
            username = os.environ[config["username_env"]]
            password = os.environ[config["password_env"]]
            encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
            self.headers["Authorization"] = "Basic " + encoded
        self.contexts: set[str] = set()
        self.verified = False
        self.metadata: dict = {}

    def _request(
        self,
        method: str,
        url: str,
        data: bytes | None = None,
        *,
        content_type: str = "application/json",
        cleanup: Callable | None = None,
        allow_inconsistent: bool = False,
        retry: bool = True,
    ) -> Response:
        for attempt in range(self.attempts if retry else 1):
            response = None
            try:
                response = self.transport(
                    Request(
                        url,
                        data=data,
                        method=method,
                        headers={**self.headers, "Content-Type": content_type},
                    ),
                    self.timeout,
                )
                if 200 <= response.status < 300:
                    return response
                if allow_inconsistent and inconsistency_reason(response):
                    return response
                transient = response.status in {429, 502, 503, 504}
                raise ReasonerError(
                    f"GraphDB HTTP {response.status}; no logical decision recorded",
                    retryable=transient,
                )
            except ReasonerError as error:
                if not error.retryable or attempt + 1 >= (self.attempts if retry else 1):
                    raise
                delay = attempt + 1
                retry_after = (response.headers or {}).get("Retry-After") if response else None
                if retry_after is not None:
                    try:
                        too_long = float(retry_after) > delay
                    except ValueError:
                        too_long = True
                    if too_long:
                        raise ReasonerError(
                            "Server Retry-After exceeds bounded retry delay; resume later",
                            retryable=True,
                        ) from error
                if method != "GET":
                    if cleanup is None:
                        raise ReasonerError(
                            "Mutation outcome uncertain; cleanup is required before retry"
                        ) from error
                    cleanup()
                self.sleep(delay)
        raise AssertionError("unreachable")

    def _json(self, url: str) -> dict:
        try:
            value = json.loads(self._request("GET", url).body)
            if not isinstance(value, dict):
                raise ValueError("Expected object")
            return value
        except (ValueError, UnicodeError) as error:
            raise ReasonerError("Malformed GraphDB JSON response") from error

    def ask(self, query: str) -> bool:
        result = self._json(self.url + "?" + urlencode({"query": query, "infer": "false"}))
        if type(result.get("boolean")) is not bool:
            raise ReasonerError("Malformed GraphDB ASK response")
        return result["boolean"]

    def _context_url(self, context: str) -> str:
        return self.url + "/statements?" + urlencode({"context": f"<{context}>"})

    def cleanup(self, context: str) -> None:
        if context not in self.contexts:
            raise ReasonerError("Refusing to clear a context not owned by this handle")
        self._request("DELETE", self._context_url(context), retry=False)
        if self.ask(f"ASK {{ GRAPH <{context}> {{ ?s ?p ?o }} }}"):
            raise ReasonerError("GraphDB context cleanup verification failed")

    def _insert(self, graph: Graph, context: str) -> Response:
        return self._request(
            "POST",
            self._context_url(context),
            graph.serialize(format="nt").encode(),
            content_type="application/n-triples",
            cleanup=lambda: self.cleanup(context),
            allow_inconsistent=True,
        )

    def _temporary(self, graph: Graph) -> tuple[Decision, dict]:
        context = "urn:kgr4vrd:query:" + uuid.uuid4().hex
        self.contexts.add(context)
        try:
            response = self._insert(graph, context)
            reason = inconsistency_reason(response)
            return (Decision.INCOMPATIBLE if reason else Decision.COMPATIBLE), {
                "reason": reason or "consistent_under_verified_ruleset",
                "response_digest": hashlib.sha256(response.body).hexdigest(),
            }
        finally:
            # Even an ambiguous timeout is followed by cleanup; cleanup failure aborts the build.
            self.cleanup(context)
            self.contexts.remove(context)

    def verify(self, ontology: Graph) -> dict:
        """Check actual configuration and positive/negative fixtures before loading schema."""
        version = self._json(self.endpoint + "/rest/info/version")
        if version.get("productVersion") != self.config["expected_version"]:
            raise ReasonerError("GraphDB productVersion does not match expected_version")
        repository = self._json(
            self.endpoint + "/rest/repositories/" + quote(self.repository, safe="")
        )
        params = repository.get("params", {})

        def parameter(name: str):
            entry = params.get(name, {})
            return entry.get("value") if isinstance(entry, dict) else entry

        if (
            parameter("ruleset") != self.config["ruleset"]
            or str(parameter("checkForInconsistencies")).lower() != "true"
        ):
            raise ReasonerError("GraphDB ruleset mismatch or consistency checking disabled")
        if self.config["ruleset"] not in {"owl2-rl", "owl2-rl-optimized"}:
            raise ReasonerError("Only the declared OWL2-RL rulesets are supported")
        if self.ask("ASK { GRAPH ?g { ?s ?p ?o } }"):
            raise ReasonerError(
                "Dedicated preparation repository has existing named graphs; nothing was cleared"
            )
        # Fresh GraphDB repositories contain W3C axiomatic triples; reject user data.
        allowed = (
            "|".join(re.escape(str(ns)) for ns in (RDF, RDFS, OWL))
            + "|http://www.w3.org/2001/XMLSchema#"
        )
        pattern = json.dumps("^(" + allowed + ")")
        if self.ask(f"ASK {{ ?s ?p ?o FILTER(!isIRI(?s) || !regex(str(?s), {pattern})) }}"):
            raise ReasonerError("Dedicated preparation repository contains user assertions")
        ns = Namespace("urn:kgr4vrd:verification:")
        good = Graph()
        good.add((ns.Person, RDFS.subClassOf, ns.Animal))
        good.add((ns.Car, RDFS.subClassOf, ns.Vehicle))
        good.add((ns.Animal, OWL.disjointWith, ns.Vehicle))
        good.add((ns.rel, RDFS.range, ns.Animal))
        good.add((ns.a, RDF.type, ns.Person))
        good.add((ns.b, RDF.type, ns.Person))
        good.add((ns.a, ns.rel, ns.b))
        if self._temporary(good)[0] != Decision.COMPATIBLE:
            raise ReasonerError("Known compatible GraphDB fixture failed")
        bad = Graph()
        for triple in good:
            bad.add(triple)
        bad.add((ns.b, RDF.type, ns.Car))
        outcome, evidence = self._temporary(bad)
        if outcome != Decision.INCOMPATIBLE:
            raise ReasonerError(
                "Known inconsistent fixture accepted: consistency checker is not verified"
            )
        context = "urn:kgr4vrd:schema:" + uuid.uuid4().hex
        self.contexts.add(context)
        response = self._insert(ontology, context)
        if inconsistency_reason(response):
            self.cleanup(context)
            self.contexts.remove(context)
            raise ReasonerError("Background ontology is inconsistent under the configured checker")
        self.verified = True
        self.metadata = {
            "backend": "GraphDB",
            "version": version["productVersion"],
            "ruleset": self.config["ruleset"],
            "consistency_fixture": evidence,
            "query_scope": "independent_typed_relation",
        }
        return self.metadata

    def __call__(
        self, subject_class: str, predicate: str, object_class: str
    ) -> tuple[Decision, dict]:
        if not self.verified:
            raise ReasonerError("GraphDB must pass verify before classifying relations")
        graph = Graph()
        fresh = Namespace("urn:kgr4vrd:individual:" + uuid.uuid4().hex + ":")
        graph.add((fresh.a, RDF.type, URIRef(subject_class)))
        graph.add((fresh.b, RDF.type, URIRef(object_class)))
        graph.add((fresh.a, URIRef(predicate), fresh.b))
        return self._temporary(graph)

    def close(self) -> None:
        for context in tuple(self.contexts):
            self.cleanup(context)
            self.contexts.remove(context)
        self.verified = False
