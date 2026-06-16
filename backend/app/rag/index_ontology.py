import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterator
from pathlib import Path

from app.rag import embedding_service, qdrant_store
from app.rag.ontology_documents import OntologyDocument, ontology_schema_path, parse_ontology_documents


def batched(items: list[OntologyDocument], batch_size: int) -> Iterator[list[OntologyDocument]]:
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def print_samples(documents: list[OntologyDocument], *, sample_count: int) -> None:
    for document in documents[:sample_count]:
        print("---")
        print(document.text())
        print(f"payload: {document.payload()}")


def document_text(document: OntologyDocument, *, max_text_length: int) -> str:
    text = document.text()
    if max_text_length > 0 and len(text) > max_text_length:
        return text[:max_text_length]
    return text


def embed_and_upsert_batch(batch: list[OntologyDocument], *, max_text_length: int) -> int:
    vectors = embedding_service.embed_texts([document_text(document, max_text_length=max_text_length) for document in batch])
    qdrant_store.upsert_points(
        (document.uri, vector, document.payload())
        for document, vector in zip(batch, vectors, strict=True)
    )
    return len(batch)


def index_documents(
    documents: list[OntologyDocument],
    *,
    batch_size: int,
    concurrency: int,
    max_text_length: int,
) -> None:
    first_batch = documents[:batch_size]
    first_vectors = embedding_service.embed_texts(
        [document_text(document, max_text_length=max_text_length) for document in first_batch]
    )
    if not first_vectors:
        raise ValueError("No vectors returned for the first ontology batch")

    qdrant_store.recreate_collection(vector_size=len(first_vectors[0]))
    qdrant_store.upsert_points(
        (document.uri, vector, document.payload())
        for document, vector in zip(first_batch, first_vectors, strict=True)
    )
    print(f"Indexed {len(first_batch)} documents into {qdrant_store.qdrant_collection()}")

    remaining_batches = list(batched(documents[batch_size:], batch_size))
    indexed_count = len(first_batch)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [
            executor.submit(embed_and_upsert_batch, batch, max_text_length=max_text_length)
            for batch in remaining_batches
        ]
        for future in as_completed(futures):
            indexed_count += future.result()
            print(f"Indexed {indexed_count}/{len(documents)} documents")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index DBpedia ontology URIs into Qdrant.")
    parser.add_argument("--path", type=Path, default=ontology_schema_path(), help="Path to ontology .nt file")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print samples without embedding/upserting")
    parser.add_argument("--sample-count", type=int, default=5, help="Number of sample documents to print")
    parser.add_argument("--limit", type=int, default=0, help="Limit documents for a partial indexing run")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding/upsert batch size")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent embedding/upsert batches")
    parser.add_argument("--max-text-length", type=int, default=1024, help="Maximum characters per embedded document text")
    args = parser.parse_args()

    documents = parse_ontology_documents(args.path)
    if args.limit > 0:
        documents = documents[:args.limit]

    print(f"Ontology path: {args.path}")
    print(f"Documents: {len(documents)}")
    print_samples(documents, sample_count=max(0, args.sample_count))

    if args.dry_run:
        return

    if not documents:
        raise ValueError("No ontology documents found")
    index_documents(
        documents,
        batch_size=max(1, args.batch_size),
        concurrency=max(1, args.concurrency),
        max_text_length=max(0, args.max_text_length),
    )


if __name__ == "__main__":
    main()
