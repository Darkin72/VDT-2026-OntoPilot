import os
import re
from dataclasses import dataclass, field
from pathlib import Path

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"
RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"

OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"
OWL_DATATYPE_PROPERTY = "http://www.w3.org/2002/07/owl#DatatypeProperty"
RDFS_DATATYPE = "http://www.w3.org/2000/01/rdf-schema#Datatype"

DBO = "http://dbpedia.org/ontology/"
DBR = "http://dbpedia.org/resource/"
DBP = "http://dbpedia.org/property/"
DBDT = "http://dbpedia.org/datatype/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
XSD = "http://www.w3.org/2001/XMLSchema#"

TRIPLE_PATTERN = re.compile(r'^<([^>]+)>\s+<([^>]+)>\s+(.+)\s+\.\s*$')
URI_OBJECT_PATTERN = re.compile(r'^<([^>]+)>$')
LITERAL_OBJECT_PATTERN = re.compile(r'^"((?:[^"\\]|\\.)*)"(?:@([A-Za-z-]+)|\^\^<[^>]+>)?$')


@dataclass
class OntologyDocument:
    uri: str
    types: set[str] = field(default_factory=set)
    labels: list[tuple[str, str]] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    ranges: list[str] = field(default_factory=list)
    sub_class_of: list[str] = field(default_factory=list)

    @property
    def curie(self) -> str:
        return to_curie(self.uri)

    @property
    def local_name(self) -> str:
        tail = self.uri.rstrip("/#").rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        return tail or self.uri

    @property
    def kind(self) -> str:
        if OWL_OBJECT_PROPERTY in self.types:
            return "ObjectProperty"
        if OWL_DATATYPE_PROPERTY in self.types:
            return "DatatypeProperty"
        if OWL_CLASS in self.types:
            return "Class"
        if RDFS_DATATYPE in self.types:
            return "Datatype"
        if self.domains or self.ranges:
            return "Property"
        return "OntologyURI"

    @property
    def label(self) -> str:
        for value, language in self.labels:
            if language.lower() == "en":
                return value
        return self.labels[0][0] if self.labels else self.local_name

    def payload(self) -> dict[str, str]:
        return {
            "uri": self.uri,
            "curie": self.curie,
            "local_name": self.local_name,
            "kind": self.kind,
            "label": self.label,
            "domain": join_curies(self.domains),
            "range": join_curies(self.ranges),
            "sub_class_of": join_curies(self.sub_class_of),
        }

    def text(self) -> str:
        lines = [
            f"URI: {self.uri}",
            f"CURIE: {self.curie}",
            f"local name: {self.local_name}",
            f"kind: {self.kind}",
            f"label: {self.label}",
        ]
        if self.domains:
            lines.append(f"domain: {join_curies(self.domains)}")
        if self.ranges:
            lines.append(f"range: {join_curies(self.ranges)}")
        if self.sub_class_of:
            lines.append(f"subclass of: {join_curies(self.sub_class_of)}")
        return "\n".join(lines)


def ontology_schema_path() -> Path:
    configured_path = os.getenv("ONTOLOGY_SCHEMA_PATH", "").strip()
    if configured_path:
        return Path(configured_path)

    container_path = Path("/app/Ontology/ontology--DEV_type=parsed_sorted.nt")
    if container_path.exists():
        return container_path

    return Path(__file__).resolve().parents[3] / "Ontology" / "ontology--DEV_type=parsed_sorted.nt"


def to_curie(uri: str) -> str:
    prefixes = (
        (DBO, "dbo"),
        (DBR, "dbr"),
        (DBP, "dbp"),
        (DBDT, "dbdt"),
        (RDF, "rdf"),
        (RDFS, "rdfs"),
        (OWL, "owl"),
        (XSD, "xsd"),
    )
    for namespace, prefix in prefixes:
        if uri.startswith(namespace):
            return f"{prefix}:{uri[len(namespace):]}"
    return f"<{uri}>"


def join_curies(uris: list[str]) -> str:
    return ", ".join(to_curie(uri) for uri in uris)


def parse_object(raw_object: str) -> tuple[str, str, str] | None:
    uri_match = URI_OBJECT_PATTERN.match(raw_object)
    if uri_match:
        return "uri", uri_match.group(1), ""

    literal_match = LITERAL_OBJECT_PATTERN.match(raw_object)
    if literal_match:
        return "literal", literal_match.group(1).replace(r'\"', '"'), literal_match.group(2) or ""

    return None


def parse_ontology_documents(path: Path) -> list[OntologyDocument]:
    documents: dict[str, OntologyDocument] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = TRIPLE_PATTERN.match(line)
            if not match:
                continue

            subject, predicate, raw_object = match.groups()
            parsed_object = parse_object(raw_object)
            if not parsed_object:
                continue

            object_kind, object_value, object_language = parsed_object
            document = documents.setdefault(subject, OntologyDocument(uri=subject))

            if predicate == RDF_TYPE and object_kind == "uri":
                document.types.add(object_value)
            elif predicate == RDFS_LABEL and object_kind == "literal":
                document.labels.append((object_value, object_language))
            elif predicate == RDFS_DOMAIN and object_kind == "uri":
                document.domains.append(object_value)
            elif predicate == RDFS_RANGE and object_kind == "uri":
                document.ranges.append(object_value)
            elif predicate == RDFS_SUBCLASS_OF and object_kind == "uri":
                document.sub_class_of.append(object_value)

    return sorted(documents.values(), key=lambda item: item.uri)
