import re
import unicodedata
from typing import Any

from app import llm_service, logging_service
from app.agents.central import extract_json_object
from app.agents.central.agent import strip_answer_options
from app.rag import ontology_retriever

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
- If an exact property is uncertain, first prefer broad predicate discovery queries using rdfs:label filters
  and return candidate ?p ?pLabel ?value. Do not invent properties outside dbo:, dbp:, rdf:, rdfs:, foaf:.
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
WORD_PATTERN = re.compile(r"[\wÀ-ỹĐđ.-]+", re.UNICODE)
STOP_RESOURCE_PHRASES = {"GraphDB", "DBpedia", "SPARQL", "JSON"}


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


def is_read_only_sparql(query: str) -> bool:
    if not query or READ_ONLY_SPARQL_PATTERN.search(query):
        return False
    without_prefixes = re.sub(r"(?im)^\s*PREFIX\s+[^\n]+\n?", "", query).strip()
    return without_prefixes.upper().startswith(("SELECT", "ASK"))


def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(character for character in normalized if unicodedata.category(character) != "Mn")
    return stripped.replace("Đ", "D").replace("đ", "d")


def resource_iri_name(text: str) -> str:
    return re.sub(r"\s+", "_", text.strip())


def possible_resource_names(user_prompt: str, query_description: str) -> list[str]:
    text = f"{query_description}\n{strip_answer_options(user_prompt)}"
    names: list[str] = []
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
    if not names:
        return ""

    lines = ["Possible DBpedia resource IRIs from the question:"]
    lines.extend(f"- dbr:{resource_iri_name(name)}" for name in names)
    return "\n".join(lines)


def generate_sparql(user_prompt: str, query_description: str) -> str:
    ontology_candidates_block = ontology_retriever.format_candidates_block(query_description)
    resource_candidates_block = possible_resource_block(user_prompt, query_description)
    ontology_candidates_prompt = (
        f"{ontology_candidates_block}\n\n"
        "Use these ontology URI candidates as schema hints only. "
        "They are not guaranteed to be populated predicates in the local graph. "
        "If predicate confidence is low, prefer a broad predicate scan from the correct subject instead of forcing a narrow candidate.\n\n"
        if ontology_candidates_block
        else ""
    )
    resource_candidates_prompt = f"{resource_candidates_block}\n\n" if resource_candidates_block else ""
    prompt = (
        "You are a SPARQL coder for the local GraphDB.\n"
        "Create one read-only SPARQL query from the central agent description.\n"
        "Only create SELECT or ASK. Do not use INSERT, DELETE, UPDATE, or SERVICE.\n"
        "Main priority for this phase: identify the correct subject and object resources. Predicate choice may be imperfect.\n"
        "Prefer returning neutral factual evidence: entities, relationships, labels, dates, counts, and literal values needed by the core question.\n"
        "For entity lookup, prefer exact dbr:Entity_Name candidates from the question before label search. Preserve Vietnamese diacritics in dbr: IRIs and also try ASCII-folded alternatives when provided.\n"
        "If exact resources are plausible, use VALUES for them and then query their predicates directly; do not make rdfs:label matching the only way to find the entity.\n"
        "When selecting resources, include labels/names when available using OPTIONAL rdfs:label or foaf:name. Do not require labels to exist.\n"
        "When the correct predicate is uncertain, use a broad pattern like ?subject ?predicate ?object with OPTIONAL predicate/object labels, and return enough rows for the central/answer agents to choose or count distinct objects.\n"
        "For count questions, return candidate objects/resources and labels; do not over-filter with a hand-picked predicate list unless the predicate is highly certain.\n"
        "For fallback lookup, search across rdfs:label, foaf:name, and dbo:alias, and avoid strict language filters unless the variable is optional evidence only.\n"
        "Do not include answer choices, option IDs, VALUES blocks for choices, or BINDs mapping choices to options. The central agent handles choices later.\n"
        "SPARQL function syntax matters: use CONTAINS(LCASE(STR(?label)), \"text\"), never LCASE(STR(?label)) CONTAINS(\"text\").\n\n"
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
        f"Central agent query description:\n{query_description}"
    )
    raw_text = llm_service.complete_text(
        [
            llm_service.system_message("You are a SPARQL coder. Return only SPARQL JSON."),
            llm_service.user_message(prompt),
        ]
    )
    logging_service.verbose_text("sparql_agent.raw_response", raw_text)
    data: dict[str, Any] = extract_json_object(raw_text)
    sparql = str(data.get("sparql", "")).strip() if data else ""
    sparql = add_missing_common_prefixes(sparql)
    if not is_read_only_sparql(sparql):
        logging_service.agent_step("sparql_agent.rejected_sparql", {"sparql": sparql})
        return ""
    logging_service.agent_text("sparql_agent.final_sparql", sparql)
    return sparql

