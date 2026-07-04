import json
import re
import time
import unicodedata
from typing import Any

from app import llm_service, logging_service
from app.agents.central import extract_json_object
from app.agents.central.agent import shorten_for_history, strip_answer_options
from app.rag import ontology_retriever

import sqlite3
from pathlib import Path
from rdflib.plugins.sparql import prepareQuery
from app.rag.ontology_retriever import fetch_document, lookup_db_path, documents_path

def validate_sparql(query: str, context: str = "") -> str | None:
    if not query.strip(): return None
    # 1. Syntax check
    try:
        prepareQuery(query)
    except Exception as e:
        return f"SYNTAX_ERROR: {e}. Please fix your SPARQL syntax."
    
    # 2. Extract URIs
    uris = set(re.findall(r'<http://dbpedia.org/resource/([^>]+)>', query))
    curies = set(re.findall(r'(?<![\w:/#])dbr:([A-Za-z_][\w.-]*)', query))
    uris.update(curies)
    subject_uris = set(re.findall(r'VALUES\s+\?\w*subject\w*\s*\{[^}]*?dbr:([A-Za-z_][\w.-]*)', query, re.IGNORECASE))
    subject_uris.update(re.findall(r'(?m)^\s*dbr:([A-Za-z_][\w.-]*)\s+\?\w+', query))
    
    if not uris: return None
        
    # 3. Check SQLite
    db_path = lookup_db_path()
    if not db_path.exists():
        db_path = Path(__file__).resolve().parents[4] / "Ontology" / "normalized" / "embedding_lookup.sqlite"
    source_path = documents_path()
    if not source_path.exists():
        source_path = Path(__file__).resolve().parents[4] / "Ontology" / "normalized" / "embedding_documents.jsonl"
    if not db_path.exists(): return None
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    missing = []
    type_mismatch = []
    normalized_context = unicodedata.normalize("NFKD", context)
    normalized_context = "".join(ch for ch in normalized_context if not unicodedata.combining(ch)).casefold()
    optional_label_vars = set(re.findall(r"OPTIONAL\s*\{[^{}]*rdfs:label\s+\?(\w+)[^{}]*\}\s*FILTER\s*\([^)]*LANG\s*\(\s*\?\1\s*\)", query, re.IGNORECASE | re.DOTALL))
    if optional_label_vars:
        return "OPTIONAL_LABEL_FILTER_OUTSIDE: A LANG filter for an OPTIONAL label variable is outside the OPTIONAL block, which drops rows when the label is absent. Move the LANG filter inside the OPTIONAL block or remove it."

    leadership_context = any(term in normalized_context for term in ["rector", "president", "principal", "head", "leader", "hieu truong"])
    if leadership_context and any(term in query for term in ["dbo:rector", "dbo:head"]):
        missing_leadership_fallbacks = [term for term in ["dbo:president", "dbo:firstPopularVote", "sameSettingAs"] if term not in query]
        if len(missing_leadership_fallbacks) >= 2:
            return "LEADERSHIP_PREDICATE_INCOMPLETE: Educational-institution leadership queries must include broader DBpedia predicates such as dbo:president and non-standard fallback predicates like dbo:firstPopularVote or DUL sameSettingAs, because local DBpedia may not use dbo:rector/dbo:head. Return a corrected focused query."

    expected_type_patterns = []
    if any(term in normalized_context for term in ["aircraft carrier", "tau san bay", "ship", "tau", "vessel"]):
        expected_type_patterns = ["ship", "aircraftcarrier", "meanoftransportation"]
    elif any(term in normalized_context for term in ["aircraft", "may bay"]):
        expected_type_patterns = ["aircraft", "meanoftransportation"]
    elif any(term in normalized_context for term in ["city", "thanh pho"]):
        expected_type_patterns = ["city", "settlement", "populatedplace"]
    elif any(term in normalized_context for term in ["university", "truong dai hoc"]):
        expected_type_patterns = ["university", "educationalinstitution"]
    elif any(term in normalized_context for term in ["person", "nguoi"]):
        expected_type_patterns = ["person"]

    for uri in uris:
        full_uri = f"http://dbpedia.org/resource/{uri}"
        row = conn.execute("SELECT * FROM documents WHERE curie = ? LIMIT 1", (f"dbr:{uri}",)).fetchone()
        if not row:
            missing.append(uri)
            continue
        if uri in subject_uris and expected_type_patterns and source_path.exists():
            with source_path.open("rb") as handle:
                document = fetch_document(handle, int(row["byte_offset"]))
            types_text = str(document.get("types_json") or document.get("types") or "").casefold()
            if types_text and not any(pattern in types_text.replace("_", "") for pattern in expected_type_patterns):
                type_mismatch.append((uri, types_text[:200]))
    conn.close()
    
    if missing:
        msg = f"ENTITY_NOT_FOUND: The following resources do not exist in the database: "
        msg += ", ".join([f"dbr:{u}" for u in missing])
        msg += ". Please rewrite the query using a valid resource or broad label search."
        return msg
    if type_mismatch:
        msg = "SUBJECT_TYPE_MISMATCH: The query anchors the requested typed entity to incompatible resources: "
        msg += ", ".join([f"dbr:{u} has types {t}" for u, t in type_mismatch])
        msg += ". Use a more specific valid resource candidate or search by label with an rdf:type/rdfs:subClassOf* constraint matching the requested entity type. For aircraft carriers/naval vessels, use dbo:Ship rather than dbo:Aircraft."
        return msg
        
    return None


READ_ONLY_SPARQL_PATTERN = re.compile(
    r"\b(INSERT|DELETE|LOAD|CLEAR|CREATE|DROP|MOVE|COPY|ADD|SERVICE)\b",
    re.IGNORECASE,
)

ONTOLOGY_HINTS = """
Local GraphDB schema hints from Ontology/ontology--DEV_type=parsed_sorted.nt:
- The graph uses DBpedia ontology IRIs. Main namespace: http://dbpedia.org/ontology/ as dbo:.
- Resource namespace is usually http://dbpedia.org/resource/ as dbr:.
- Labels are stored with rdfs:label, often with @en language tags.
- Classes are declared as owl:Class. Examples:
  dbo:Academic, dbo:AcademicConference, dbo:AcademicJournal, dbo:AcademicSubject,
  dbo:Activity, dbo:Actor, dbo:AdministrativeRegion, dbo:Agent, dbo:Aircraft,
  dbo:Airline, dbo:Airport, dbo:Album, dbo:Ambassador, dbo:Animal, dbo:Architect,
  dbo:ArchitecturalStructure, dbo:Artist, dbo:Artwork, dbo:Astronaut, dbo:Athlete,
  dbo:Automobile, dbo:Award, dbo:Band, dbo:Bank, dbo:BaseballPlayer,
  dbo:BasketballPlayer, dbo:BasketballTeam, dbo:Bay, dbo:Beach.
- Object properties are declared as owl:ObjectProperty and connect resources to resources. Examples:
  dbo:academicAdvisor, dbo:academicDiscipline, dbo:academyAward, dbo:achievement,
  dbo:activity, dbo:adjacentSettlement, dbo:administrativeCenter,
  dbo:administrator, dbo:affiliation, dbo:agency, dbo:airline, dbo:album,
  dbo:almaMater, dbo:architect, dbo:architecturalStyle, dbo:artist,
  dbo:author, dbo:award, dbo:bandMember, dbo:basedOn, dbo:basinCountry.
- Some numeric/literal datatype properties are class-scoped IRIs, not normal dbo:localName CURIEs.
  Use full IRIs for these, for example:
  <http://dbpedia.org/ontology/Person/height>, <http://dbpedia.org/ontology/Person/weight>,
  <http://dbpedia.org/ontology/Building/floorArea>,
  <http://dbpedia.org/ontology/Automobile/fuelCapacity>,
  <http://dbpedia.org/ontology/Automobile/wheelbase>,
  <http://dbpedia.org/ontology/Engine/topSpeed>,
  <http://dbpedia.org/ontology/Engine/powerOutput>,
  <http://dbpedia.org/ontology/PopulatedPlace/areaTotal>,
  <http://dbpedia.org/ontology/PopulatedPlace/populationDensity>,
  <http://dbpedia.org/ontology/Lake/volume>.
- Datatype units use http://dbpedia.org/datatype/, for example metre, kilometre, kilogram, kelvin,
  squareKilometre, inhabitantsPerSquareKilometre.
- If an exact property is uncertain, first prefer anchored predicate discovery using a known subject,
  a constrained rdf:type class, a VALUES ?p candidate list, or schema-only property typing.
  Return candidate ?p ?pLabel ?value only from anchored patterns. Do not invent properties outside dbo:, dbp:, rdf:, rdfs:, foaf:.
""".strip()

COMMON_PREFIXES = {
    "dbo": "http://dbpedia.org/ontology/",
    "dbr": "http://dbpedia.org/resource/",
    "dbp": "http://dbpedia.org/property/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "owl": "http://www.w3.org/2002/07/owl#",
}

PREFIX_DECLARATION_PATTERN = re.compile(r"(?im)^\s*PREFIX\s+([A-Za-z][\w-]*):")
PREFIX_USAGE_PATTERN = re.compile(r"(?<![\w:/#])([A-Za-z][\w-]*):[A-Za-z_][\w.-]*")
PREFIX_LINE_PATTERN = re.compile(r"(?im)^\s*PREFIX\s+[^\n]+\n?")
LEADING_VALUES_PATTERN = re.compile(r"(?is)^\s*((?:VALUES\s+\?\w+\s*\{[^{}]*\}\s*)+)((?:SELECT|ASK)\b.*)$")
WORD_PATTERN = re.compile(r"[^\W\d_][\w.-]*", re.UNICODE)
STOP_RESOURCE_PHRASES = {"GraphDB", "DBpedia", "SPARQL", "JSON", "Entity", "Length", "Resource", "Question", "Answer"}
OPERATIONAL_LOOKUP_TERM_PATTERN = re.compile(
    r"(?is)\b(sparql|query|count|count\s*distinct|total\s+number|single\s+integer|"
    r"aggregate|perform|execute|return|select|ask|where|limit|evidence)\b"
)


def add_missing_common_prefixes(query: str) -> str:
    declared_prefixes = set(PREFIX_DECLARATION_PATTERN.findall(query))
    used_prefixes = set(PREFIX_USAGE_PATTERN.findall(query))
    missing_prefixes = [
        prefix for prefix in COMMON_PREFIXES
        if prefix in used_prefixes and prefix not in declared_prefixes
    ]
    if not missing_prefixes:
        return query

    declarations = "\n".join(
        f"PREFIX {prefix}: <{COMMON_PREFIXES[prefix]}>" for prefix in missing_prefixes
    )
    return f"{declarations}\n{query}"


def move_leading_values_into_where(query: str) -> str:
    prefix_lines = "".join(PREFIX_LINE_PATTERN.findall(query))
    body = PREFIX_LINE_PATTERN.sub("", query).strip()
    match = LEADING_VALUES_PATTERN.match(body)
    if not match:
        return query

    leading_values = match.group(1).strip()
    query_body = match.group(2).strip()
    where_start = query_body.find("{")
    if where_start < 0:
        return query

    indented_values = "\n".join(f"  {line}" for line in leading_values.splitlines())
    return f"{prefix_lines}{query_body[:where_start + 1]}\n{indented_values}{query_body[where_start + 1:]}"




def expand_parenthesized_dbr_resources(query: str) -> str:
    return re.sub(
        r"(?<![\w:/#])dbr:([^\s{};,]+\([^\s{};,]+\)[^\s{};,]*)",
        lambda match: f"<http://dbpedia.org/resource/{match.group(1)}>",
        query,
    )


def normalize_filter_logical_or(query: str) -> str:
    return re.sub(r"(?i)(\s+)OR(\s+)", r"\1||\2", query)

def normalize_escaped_sparql_whitespace(query: str) -> str:
    return query.replace(r"\n", "\n").replace(r"\t", "\t")

def ensure_select_limit(query: str, *, default_limit: int = 200) -> str:
    body = PREFIX_LINE_PATTERN.sub("", query).strip()
    if not body.upper().startswith("SELECT"):
        return query
    if re.search(r"(?is)\bCOUNT\s*\(", body):
        return query
    if re.search(r"(?is)\bLIMIT\s+\d+\s*$", query):
        return query
    return f"{query.rstrip()}\nLIMIT {default_limit}"

def is_read_only_sparql(query: str) -> bool:
    if not query or READ_ONLY_SPARQL_PATTERN.search(query):
        return False
    without_prefixes = PREFIX_LINE_PATTERN.sub("", query).strip()
    return without_prefixes.upper().startswith(("SELECT", "ASK"))


def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(character for character in normalized if unicodedata.category(character) != "Mn")
    return stripped.replace("\u0110", "D").replace("\u0111", "d")


def resource_iri_name(text: str) -> str:
    return re.sub(r"\s+", "_", text.strip())




def _local_lookup_db_path() -> Path | None:
    db_path = lookup_db_path()
    if not db_path.exists():
        db_path = Path(__file__).resolve().parents[4] / "Ontology" / "normalized" / "embedding_lookup.sqlite"
    return db_path if db_path.exists() else None


def _clean_resource_name(name: str) -> str:
    prefixes = {"entity", "resource", "ship", "vessel", "aircraft", "carrier", "tau", "may", "bay", "san", "city"}
    words = name.split()
    while words and strip_diacritics(words[0]).casefold() in prefixes:
        words.pop(0)
    return " ".join(words).strip()


def lookup_resource_candidates(names: list[str], context: str, *, limit: int = 8) -> list[str]:
    db_path = _local_lookup_db_path()
    if not db_path:
        return []

    normalized_context = unicodedata.normalize("NFKD", context)
    normalized_context = "".join(ch for ch in normalized_context if not unicodedata.combining(ch)).casefold()
    year_tokens = re.findall(r"\b(?:18|19|20)\d{2}\b", context)
    candidates: list[str] = []

    def add(curie: str) -> None:
        if curie and curie.startswith("dbr:") and curie not in candidates:
            candidates.append(curie)

    query_names = []
    for name in names:
        for candidate in [name, _clean_resource_name(name), strip_diacritics(_clean_resource_name(name))]:
            candidate = candidate.strip()
            if candidate and candidate not in query_names:
                query_names.append(candidate)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    for name in query_names:
        iri_name = resource_iri_name(name)
        exact_curie = f"dbr:{iri_name}"
        row = conn.execute("SELECT curie FROM documents WHERE curie = ? AND kind = 'entity' LIMIT 1", (exact_curie,)).fetchone()
        if row:
            add(str(row["curie"]))

        rows = conn.execute(
            """
            SELECT d.curie, d.uri FROM terms t
            JOIN documents d ON d.doc_id = t.doc_id
            WHERE t.term = ? AND d.kind = 'entity'
            ORDER BY d.line_number
            LIMIT 20
            """,
            (ontology_retriever.normalize_term(name),),
        ).fetchall()
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                0 if any(year in str(row["uri"]) for year in year_tokens) else 1,
                0 if "aircraft_carrier" in normalized_context and "aircraft_carrier" in str(row["uri"]).casefold() else 1,
                len(str(row["uri"])),
            ),
        )
        for row in sorted_rows[:3]:
            add(str(row["curie"]))
        if len(candidates) >= limit:
            break
    conn.close()
    return candidates[:limit]


def possible_resource_names(user_prompt: str, query_description: str) -> list[str]:
    text = f"{query_description}\n{strip_answer_options(user_prompt)}"
    names: list[str] = []
    for match in re.findall(r"[A-Z][\w.-]*(?:\s+[A-Z][\w.-]*)+\s*\([^)]+\)", text):
        cleaned = match.strip()
        if cleaned and cleaned not in names:
            names.append(cleaned)
    for segment in re.split(r"[\n\r\t,;:!?()\[\]{}]+", text):
        words = WORD_PATTERN.findall(segment)
        index = 0
        while index < len(words):
            word = words[index]
            if not word[:1].isupper():
                index += 1
                continue

            phrase_words = [word]
            cursor = index + 1
            while cursor < len(words) and len(phrase_words) < 6 and words[cursor][:1].isupper():
                phrase_words.append(words[cursor])
                cursor += 1

            index = cursor
            if len(phrase_words) < 2:
                continue

            name = " ".join(phrase_words).strip(" .,:;!?()[]{}\"'")
            if not name or name in STOP_RESOURCE_PHRASES:
                continue
            if name not in names:
                names.append(name)

            ascii_name = strip_diacritics(name)
            if ascii_name != name and ascii_name not in names:
                names.append(ascii_name)
    return names[:8]


def possible_resource_block(user_prompt: str, query_description: str) -> str:
    names = possible_resource_names(user_prompt, query_description)
    candidates = lookup_resource_candidates(names, f"{user_prompt}\n{query_description}")
    if not candidates:
        return ""

    lines = ["Verified DBpedia resource IRIs from local ontology lookup:"]
    lines.extend(f"- {candidate}" for candidate in candidates)
    return "\n".join(lines)

def generate_ontology_lookup_terms(
    user_prompt: str,
    query_description: str,
    *,
    subquery_id: str | None = None,
    round_context: dict[str, Any] | None = None,
) -> list[str]:
    subquery = round_context.get("subquery") if isinstance(round_context, dict) else None
    purpose = str(subquery.get("purpose", "")) if isinstance(subquery, dict) else ""
    expected_evidence = str(subquery.get("expected_evidence", "")) if isinstance(subquery, dict) else ""
    prompt = (
        "Create ontology lookup terms for a DBpedia/GraphDB SPARQL coder.\n"
        "Return only terms that could be ontology classes, properties, resource labels, aliases, or domain concepts.\n"
        "Do not include operational/query-planning words such as SPARQL, query, COUNT, COUNT DISTINCT, total number, single integer, aggregate, perform, execute, return, SELECT, WHERE, LIMIT, or evidence.\n"
        "Prefer short terms, CURIEs, or labels, for example: scientist, researcher, scholar, occupation, field, dbo:Scientist, dbo:occupation, dbp:occupation.\n"
        "Return only valid JSON with this schema: {\"terms\":[\"...\"]}. Use 3 to 8 terms.\n\n"
        f"Original user prompt:\n{user_prompt}\n\n"
        f"Subquery id: {subquery_id or ''}\n"
        f"Central subquery description:\n{query_description}\n\n"
        f"Purpose:\n{purpose}\n\n"
        f"Expected evidence:\n{expected_evidence}"
    )
    logging_service.trace_step(
        "sparql_agent.ontology_lookup_terms_input",
        {"subquery_id": subquery_id, "round_context": shorten_for_history(round_context), "prompt": prompt},
        limit=12000,
    )
    raw_text = llm_service.complete_text(
        [
            llm_service.system_message("You choose ontology lookup terms. Return only lookup-term JSON."),
            llm_service.user_message(prompt),
        ]
    )
    logging_service.trace_text("sparql_agent.raw_ontology_lookup_terms", raw_text, limit=12000)
    data = extract_json_object(raw_text)
    raw_terms = data.get("terms") if isinstance(data, dict) else None
    if not isinstance(raw_terms, list):
        return []

    terms: list[str] = []
    seen_terms: set[str] = set()
    for raw_term in raw_terms:
        term = str(raw_term).strip(" \t\r\n.,;!?()[]{}<>`~|/@#$%^&*+=\\\"")
        if not term or OPERATIONAL_LOOKUP_TERM_PATTERN.search(term):
            continue
        normalized = term.casefold()
        if normalized in seen_terms:
            continue
        seen_terms.add(normalized)
        terms.append(term)
        if len(terms) >= 8:
            break

    logging_service.agent_step("sparql_agent.ontology_lookup_terms", {"terms": terms}, limit=2000)
    return terms



def previous_attempts_block(history: dict[str, Any] | None) -> str:
    if not history:
        return ""

    rounds = history.get("rounds", [])
    if isinstance(rounds, list) and rounds:
        recent_rounds = rounds[-2:]
        attempts: list[dict[str, Any]] = []
        for round_data in recent_rounds:
            if not isinstance(round_data, dict):
                continue
            for execution in round_data.get("executions", []):
                if not isinstance(execution, dict):
                    continue
                attempts.append(
                    {
                        "round": round_data.get("round"),
                        "subquery_id": execution.get("subquery_id"),
                        "query_description": shorten_for_history(execution.get("query_description", "")),
                        "sparql": shorten_for_history(execution.get("sparql", "")),
                        "result_summary": shorten_for_history(execution.get("result_summary", "")),
                        "error": shorten_for_history(execution.get("error", "")),
                    }
                )
        summary = history.get("accumulated_summary")
        return (
            "Previous GraphDB context in this same user request:\n"
            f"Accumulated summary: {json.dumps(shorten_for_history(summary), ensure_ascii=False, indent=2)}\n"
            f"Recent executions: {json.dumps(attempts[-8:], ensure_ascii=False, indent=2)}\n\n"
            "Use this context directly: do not repeat failed query shapes. If exact predicates returned no rows, use a discovery query over non-rdf:type predicates with language-filtered labels.\n\n"
        )

    steps = history.get("steps", [])
    if not isinstance(steps, list):
        return ""

    attempts: list[dict[str, Any]] = []
    for step in steps:
        if isinstance(step, dict) and step.get("type") == "sparql_execution":
            attempts.append(
                {
                    "attempt": step.get("attempt"),
                    "query_description": shorten_for_history(step.get("query_description", "")),
                    "sparql": shorten_for_history(step.get("sparql", "")),
                    "result_summary": shorten_for_history(step.get("result_summary", "")),
                    "error": shorten_for_history(step.get("error", "")),
                }
            )
    if not attempts:
        return ""

    return (
        "Previous SPARQL attempts in this same user request:\n"
        f"{json.dumps(attempts[-4:], ensure_ascii=False, indent=2)}\n\n"
        "Use this history directly: do not repeat failed query shapes, malformed VALUES blocks, broad DISTINCT label scans, "
        "or queries that caused GraphDB memory errors. Prefer a narrower VALUES-based query when a retrieved resource candidate exists.\n\n"
    )
def format_ontology_candidates_block(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return ""

    lines = ["Retrieved ontology/resource URI candidates:"]
    lines.extend(f"- {ontology_retriever.format_candidate(candidate)}" for candidate in candidates)
    return "\n".join(lines)


def generate_sparql(
    user_prompt: str,
    query_description: str,
    history: dict[str, Any] | None = None,
    *,
    subquery_id: str | None = None,
    round_context: dict[str, Any] | None = None,
) -> str:
    started = time.monotonic()
    ontology_lookup_terms = generate_ontology_lookup_terms(
        user_prompt,
        query_description,
        subquery_id=subquery_id,
        round_context=round_context,
    )
    ontology_candidates = ontology_retriever.retrieve_candidates_for_terms(
        ontology_lookup_terms,
        source_query=query_description,
    )
    logging_service.agent_step(
        "sparql_agent.ontology_lookup_done",
        {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "candidate_count": len(ontology_candidates),
            "candidates": [ontology_retriever.format_candidate(candidate) for candidate in ontology_candidates[:8]],
        },
        limit=4000,
    )
    previous_attempts_prompt = previous_attempts_block(history)
    ontology_candidates_block = format_ontology_candidates_block(ontology_candidates)
    resource_candidates_block = possible_resource_block(user_prompt, query_description)
    ontology_candidates_prompt = (
        f"{ontology_candidates_block}\n\n"
        "Use retrieved candidates as hints, not as final evidence. Candidates may include subject resources, "
        "answer resources, classes, or predicates. First identify the likely subject resource from the user prompt. "
        "Then choose a predicate candidate only when its CURIE, label, kind, domain/range, or surrounding context clearly matches the requested relation/attribute. "
        "When a retrieved schema/property candidate is plausible, test it directly before inventing a hand-written list of common predicates. "
        "Do not prefer a hand-written list such as parent/child/spouse/date/type/etc. over a retrieved candidate that semantically matches the request. "
        "A dbo: schema candidate can represent a class or a predicate; verify its role by using it in a focused query when its label fits the requested relation/attribute. "
        "Broad or generic predicate labels can still be valid in DBpedia; do not discard them only because they are not named like parent/spouse/date/etc. "
        "For non-count questions, a focused verification query should bind or use the candidate predicate and return distinct objects plus labels. "
        "For count questions, bind or use the candidate predicate inside COUNT(DISTINCT ...) and return only the aggregate count. "
        "When predicate confidence is low for non-count questions, use an anchored predicate scan from a known subject, VALUES subject list, or constrained rdf:type class and return predicate labels plus objects; for count questions, count distinct matching subjects or objects from the anchored pattern.\n\n"
        if ontology_candidates_block
        else ""
    )
    resource_candidates_prompt = f"{resource_candidates_block}\n\n" if resource_candidates_block else ""
    prompt = (
        "You are a SPARQL coder for the local GraphDB.\n"
        "Create one read-only SPARQL query from the central agent description.\n"
        "Only create SELECT or ASK. Do not use INSERT, DELETE, UPDATE, or SERVICE.\n"
        "After PREFIX declarations, the query must start with SELECT or ASK. Put VALUES blocks inside WHERE { ... }, never before SELECT/ASK.\n"
        "If Round context includes current_subquery_attempts, repair the newest failed SPARQL directly. Use the GraphDB error body as authoritative syntax feedback and do not repeat the same query.\n"
        "VALUES syntax matters: separate values with whitespace only. Never use |, commas, or semicolons inside VALUES. Correct: VALUES ?p { dbo:port dbo:homePort dbo:location }. Wrong: VALUES ?p { dbo:port | dbo:homePort | dbo:location }. Use | only in property paths such as ?ship (dbo:port|dbo:homePort) ?location .\n"
        "Query-cost safety rules, highest priority after read-only safety:\n"
        "- Never generate unconstrained global triple scans such as ?s ?p ?o, ?x ?p ?y, ?subject ?predicate ?object, or SELECT DISTINCT ?p over the whole graph.\n"
        "- Never UNION a full triple scan with schema label lookup, for example { ?s ?p ?o . } UNION { ?p rdfs:label ?pLabel . }.\n"
        "- Never search all predicates with FILTER(CONTAINS(STR(?p), ...)) unless ?p is constrained by VALUES, a schema type, or a small candidate set.\n"
        "- Predicate discovery must be anchored by at least one of: a known resource IRI/VALUES ?subject, a narrow rdf:type/rdfs:subClassOf* class constraint, a VALUES ?p candidate list, or schema-only property typing such as ?p a owl:ObjectProperty, ?p a owl:DatatypeProperty, or ?p a rdf:Property.\n"
        "- If no safe anchor exists, return a schema-only property discovery query or {\"sparql\":\"\"}; do not scan data triples globally.\n"
        "- Prefer ontology candidates and VALUES lists over CONTAINS scans. If using CONTAINS for property discovery, apply it only to schema/property resources, not to every data triple.\n"
        "Highest-priority count rule: when the original prompt or central description asks 'how many', 'count', 'number of', 'bao nhiêu', 'mấy', or 'số lượng', the query must return one numeric scalar using COUNT(DISTINCT ...). Do not list matching rows first.\n"
        "For count questions, even if the central description says 'find all', 'search for entities', 'discover entities', 'list URIs', or 'return labels', interpret that as defining the set to count, not as permission to SELECT the rows.\n"
        "For count questions likely to match more than 100 rows, never use SELECT ?entity ... LIMIT 100/200 as the primary answer; use SELECT (COUNT(DISTINCT ?entity) AS ?count) WHERE { ... }.\n"
        "Example for a count over scientists: SELECT (COUNT(DISTINCT ?scientist) AS ?count) WHERE { ?scientist rdf:type dbo:Scientist . }.\n"
        "Main priority for non-count questions: identify the correct subject and object resources. Predicate choice may be imperfect.\n"
        "Prefer returning neutral factual evidence: entities, relationships, labels, dates, counts, and literal values needed by the core question.\n"
        "For entity lookup, prefer exact dbr:Entity_Name candidates from the question before label search. Preserve Vietnamese diacritics in dbr: IRIs and also try ASCII-folded alternatives when provided.\n"
        "If exact resources are plausible, use VALUES for them and then query their predicates directly; do not make rdfs:label matching the only way to find the entity.\n"
        "If retrieved candidates include both a likely subject resource and a likely schema/property for the requested relation or attribute, first create a focused query using those candidates; use anchored scans only after that is unsuitable.\n"
        "For relationship or attribute questions, a retrieved property candidate whose label literally matches the requested concept is stronger evidence than generic property names you remember from DBpedia.\n"
        "For count questions such as 'how many', 'bao nhiêu', 'mấy', or 'số lượng', return an aggregate count query using COUNT(DISTINCT ?entity) AS ?count. Do not return candidate rows for the primary count. Do not add LIMIT to aggregate count queries that return one row.\n"
        "If the requested count may involve more than 100 matching entities, use COUNT(DISTINCT ...) instead of SELECT rows with LIMIT 100; limited rows are samples, not totals.\n"
        "If the original user prompt is a count question, this rule overrides central descriptions that say 'find all', 'list', or ask for URI rows; produce the numeric aggregate count first.\n"
        "If a count question also needs verification, the central agent can ask for a separate sample query later; the first query should produce the numeric count.\n"
        "When selecting resources for non-count questions, include labels/names when available using OPTIONAL rdfs:label or foaf:name. Do not require labels to exist.\n"
        "When the correct predicate is uncertain for non-count questions, use an anchored pattern like VALUES ?subject { dbr:Known_Resource } ?subject ?predicate ?object, or ?subject rdf:type/rdfs:subClassOf* dbo:KnownClass . ?subject ?predicate ?object, with OPTIONAL predicate/object labels. For count questions with uncertain predicates, count distinct subjects or objects from an anchored pattern instead of returning rows.\n"
        "For broad predicate discovery over an entity, filter out rdf:type unless type is the requested answer; prefer non-type predicates and language-filter labels to en, vi, or empty language.\n"
        "If exact predicates such as deathPlace return no rows, try semantically adjacent DBpedia predicates such as restingPlace, placeOfBurial, location, subdivision, country, or other location hierarchy evidence when relevant.\n"
        "Avoid SELECT DISTINCT for broad label scans; use SELECT with LIMIT 100 instead to keep GraphDB memory usage low.\n"
        "For fallback lookup, search across rdfs:label, foaf:name, and dbo:alias, and avoid strict language filters unless the variable is optional evidence only.\n"
        "Respect entity-type constraints from the wording. If the question asks for ships/tàu, constrain candidates to ship-like resources when possible; for aircraft/máy bay use aircraft-like classes; for cities/thành phố use city/settlement classes and avoid provinces/regions; for universities use university/educational-institution classes; for people/người use person classes. Prefer rdf:type/rdfs:subClassOf* constraints or anchored type verification over label/common-sense matching.\n"
        "For list, count, comparison, and superlative questions, do not return or compare entities outside the requested class just because their labels look plausible.\n"
        "Do not include answer choices, option IDs, VALUES blocks for choices, or BINDs mapping choices to options. The central agent handles choices later.\n"
        "SPARQL function syntax matters: use CONTAINS(LCASE(STR(?label)), \"text\"), never LCASE(STR(?label)) CONTAINS(\"text\"). Use || for logical OR; never use keyword OR inside FILTER expressions. If combining EXISTS with another boolean condition, put the whole expression inside one FILTER(...); never place || between graph patterns.\n\n"
        "Common prefixes:\n"
        "PREFIX dbo: <http://dbpedia.org/ontology/>\n"
        "PREFIX dbr: <http://dbpedia.org/resource/>\n"
        "PREFIX dbp: <http://dbpedia.org/property/>\n"
        "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n"
        "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n\n"
        f"{resource_candidates_prompt}"
        f"{ontology_candidates_prompt}"
        f"Schema hints:\n{ONTOLOGY_HINTS}\n\n"
        "Return only valid JSON with this schema: {\"sparql\":\"...\"}.\n"
        "If a useful query cannot be created, return {\"sparql\":\"\"}.\n\n"
        f"Original user prompt, for context only; do not extract answer options from it:\n{user_prompt}\n\n"
        f"Subquery id: {subquery_id or ''}\n"
        f"Round context:\n{json.dumps(shorten_for_history(round_context), ensure_ascii=False, indent=2) if round_context else '{}'}\n\n"
        f"Central agent query description:\n{query_description}"
    )
    logging_service.trace_step(
        "sparql_agent.generate_input",
        {
            "subquery_id": subquery_id,
            "query_description": query_description,
            "ontology_lookup_terms": ontology_lookup_terms,
            "ontology_candidates": ontology_candidates,
            "prompt": prompt,
        },
        limit=30000,
    )
    
    messages = [
        llm_service.system_message("You are a SPARQL coder. Return only SPARQL JSON."),
        llm_service.user_message(prompt),
    ]
    
    for attempt in range(3):
        attempt_started = time.monotonic()
        logging_service.agent_step(
            "sparql_agent.llm_generate_start",
            {"attempt": attempt + 1, "message_chars": sum(len(message.get("content", "")) for message in messages)},
            limit=1000,
        )
        raw_text = llm_service.complete_text(messages)
        logging_service.trace_text(f'sparql_agent.raw_response_attempt_{attempt+1}', raw_text, limit=30000)
        data = extract_json_object(raw_text)
        sparql = str(data.get('sparql', '')).strip() if data else ''
        if not sparql: return ''
        sparql = normalize_escaped_sparql_whitespace(sparql)
        sparql = add_missing_common_prefixes(sparql)
        sparql = expand_parenthesized_dbr_resources(sparql)
        sparql = move_leading_values_into_where(sparql)
        sparql = normalize_filter_logical_or(sparql)
        sparql = ensure_select_limit(sparql)
        if not is_read_only_sparql(sparql):
            logging_service.trace_step('sparql_agent.rejected_sparql', {'sparql': sparql})
            return ''
        validation_error = validate_sparql(sparql, context=user_prompt + "\n" + query_description)
        logging_service.agent_step(
            "sparql_agent.llm_generate_done",
            {
                "attempt": attempt + 1,
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                "sparql": sparql,
                "validation": validation_error or "ok",
            },
            limit=8000,
        )
        if validation_error:
            logging_service.agent_step('sparql_agent.validation_error', {'attempt': attempt+1, 'error': validation_error})
            messages.append({"role": "assistant", "content": raw_text})
            err_msg = 'Validation failed:\n' + validation_error + '\nReturn a corrected JSON object with {"sparql": "..."}.'
            messages.append(llm_service.user_message(err_msg))
            continue
        logging_service.agent_text('sparql_agent.final_sparql', sparql, limit=12000)
        return sparql
    logging_service.agent_text('sparql_agent.validation_failed_max_attempts', sparql, limit=12000)
    return sparql
