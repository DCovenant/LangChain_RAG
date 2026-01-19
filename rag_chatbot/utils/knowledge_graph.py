"""
Knowledge Graph for RAG Enhancement.

Hierarchical structure: Document → Section → Chunk
Rich entity extraction: Standards, Abbreviations, Technical Terms
"""
import re
import pickle
import logging
from typing import List, Dict, Set, Optional
import networkx as nx

logger = logging.getLogger(__name__)

# Entity patterns - more comprehensive
PATTERNS = {
    "standard": re.compile(r'\b(BS\s*EN\s*\d+[-\d]*|IEC[/\s]*\d+[-\d]*|ISO\s*\d+[-\d]*)\b', re.I),
    "spec_ref": re.compile(r'\b([SPF][RPOA][-_]?NET[-_]?[A-Z]{3}[-_]?\d{3})\b', re.I),
    "part_number": re.compile(r'\b([A-Z]{2,4}\d{2,}[-_]?\d*)\b'),
    "abbreviation": re.compile(r'\b([A-Z]{2,5})\b'),  # Will filter against known list
}

# Known abbreviations from technical documents
KNOWN_ABBREVIATIONS = {
    "HTM", "USCD", "SCD", "SPS", "ALS", "FRP", "STL", "RIV", "HTV", "RTV", "LSR",
    "CIGRE", "EATS", "ESI", "CDM", "EAWR", "ESQC", "HASAWA", "HSE", "PSSR"
}


class KnowledgeGraph:
    """Hierarchical knowledge graph: Document → Section → Chunk → Entity"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_to_chunks: Dict[str, Set[str]] = {}
        self.chunk_to_entities: Dict[str, Set[str]] = {}
        self.section_to_chunks: Dict[str, Set[str]] = {}

    def build_from_chunks(self, chunks: List[Dict]) -> None:
        """Build hierarchical graph from chunks."""
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            doc = chunk["file_name"]
            page = chunk.get("page_number", 0)
            content_type = chunk.get("content_type", "body")
            text = chunk.get("chunk_text", "")

            # Document node
            if not self.graph.has_node(doc):
                self.graph.add_node(doc, type="document")

            # Chunk node
            self.graph.add_node(chunk_id, type="chunk", page=page, content_type=content_type)

            # Section hierarchy: doc → section → chunk
            sections = chunk.get("parent_sections", [])
            if sections:
                # Build section chain
                parent = doc
                for i, sec in enumerate(sections):
                    section_id = f"{doc}::s{sec}"
                    if not self.graph.has_node(section_id):
                        self.graph.add_node(section_id, type="section", number=sec)
                        self.graph.add_edge(parent, section_id, relation="has_section")
                    parent = section_id

                    # Track section → chunks
                    if section_id not in self.section_to_chunks:
                        self.section_to_chunks[section_id] = set()
                    self.section_to_chunks[section_id].add(chunk_id)

                self.graph.add_edge(parent, chunk_id, relation="contains")
            else:
                self.graph.add_edge(doc, chunk_id, relation="contains")

            # Extract entities
            entities = self._extract_all_entities(text)
            if chunk.get("has_table") and chunk.get("table_data"):
                entities.update(self._extract_table_entities(chunk["table_data"]))

            # Store entity mappings
            self.chunk_to_entities[chunk_id] = entities
            for entity in entities:
                if entity not in self.entity_to_chunks:
                    self.entity_to_chunks[entity] = set()
                self.entity_to_chunks[entity].add(chunk_id)

                # Only add important entities as nodes (standards, specs)
                if self._is_important_entity(entity):
                    if not self.graph.has_node(entity):
                        self.graph.add_node(entity, type="entity")
                    self.graph.add_edge(chunk_id, entity, relation="references")

        logger.info(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def _extract_all_entities(self, text: str) -> Set[str]:
        """Extract all entity types from text."""
        entities = set()

        # Standards (BS EN 60060-1, IEC 60815, etc.)
        for m in PATTERNS["standard"].findall(text):
            entities.add(self._normalize_standard(m))

        # Spec references (SP-NET-SST-501, PR-NET-ENG-505)
        for m in PATTERNS["spec_ref"].findall(text):
            entities.add(m.upper().replace(" ", "-"))

        # Part numbers
        for m in PATTERNS["part_number"].findall(text):
            if len(m) >= 4:  # Filter noise
                entities.add(m.upper())

        # Known abbreviations only
        for m in PATTERNS["abbreviation"].findall(text):
            if m in KNOWN_ABBREVIATIONS:
                entities.add(m)

        return entities

    def _normalize_standard(self, s: str) -> str:
        """Normalize standard references: 'BS EN 60060-1' → 'BS-EN-60060-1'"""
        return re.sub(r'\s+', '-', s.upper().strip())

    def _extract_table_entities(self, table_data: Dict) -> Set[str]:
        """Extract entities from table cells."""
        entities = set()

        # From headers
        for h in table_data.get("headers", []):
            if h and not h.startswith("col_"):
                clean = re.sub(r'<[^>]+>', '', h)
                entities.update(self._extract_all_entities(clean))

        # From cell values
        for row in table_data.get("rows", []):
            for cell in row.values():
                if isinstance(cell, dict):
                    val = cell.get("value", "")
                    if val:
                        clean = re.sub(r'<[^>]+>', '', val)
                        entities.update(self._extract_all_entities(clean))

        return entities

    def _is_important_entity(self, entity: str) -> bool:
        """Check if entity should be a graph node (not just indexed)."""
        # Standards and spec references are important
        if re.match(r'^(BS|IEC|ISO|[SPF][RPOA]-NET)', entity, re.I):
            return True
        # Known abbreviations
        if entity in KNOWN_ABBREVIATIONS:
            return True
        return False

    def find_chunks_by_entity(self, entity: str) -> List[str]:
        """Find chunks mentioning an entity."""
        normalized = entity.upper().replace(" ", "-")
        return list(self.entity_to_chunks.get(normalized, []))

    def find_chunks_by_section(self, doc: str, section: str) -> List[str]:
        """Find all chunks in a section."""
        section_id = f"{doc}::s{section}"
        return list(self.section_to_chunks.get(section_id, []))

    def find_related_chunks(self, chunk_id: str) -> List[str]:
        """Find related chunks via shared entities or same section."""
        related = set()

        # Via shared entities
        for entity in self.chunk_to_entities.get(chunk_id, []):
            if self._is_important_entity(entity):
                related.update(self.entity_to_chunks.get(entity, []))

        # Via same section
        for section_id, chunks in self.section_to_chunks.items():
            if chunk_id in chunks:
                related.update(chunks)

        related.discard(chunk_id)
        return list(related)[:20]  # Limit

    def extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from user query."""
        return list(self._extract_all_entities(query))

    def find_documents_by_entity(self, entity: str) -> List[str]:
        """Find documents containing an entity."""
        chunks = self.find_chunks_by_entity(entity)
        docs = set()
        for cid in chunks:
            # Walk up to document
            for pred in self.graph.predecessors(cid):
                node_type = self.graph.nodes[pred].get("type")
                if node_type == "document":
                    docs.add(pred)
                elif node_type == "section":
                    # Go up from section
                    for ppred in self.graph.predecessors(pred):
                        if self.graph.nodes[ppred].get("type") == "document":
                            docs.add(ppred)
        return list(docs)

    def save(self, path: str) -> None:
        """Save graph to disk."""
        data = {
            "graph": self.graph,
            "entity_to_chunks": self.entity_to_chunks,
            "chunk_to_entities": self.chunk_to_entities,
            "section_to_chunks": self.section_to_chunks,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved to {path}")

    def load(self, path: str) -> None:
        """Load graph from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.graph = data["graph"]
        self.entity_to_chunks = data["entity_to_chunks"]
        self.chunk_to_entities = data["chunk_to_entities"]
        self.section_to_chunks = data.get("section_to_chunks", {})
        logger.info(f"Loaded: {self.graph.number_of_nodes()} nodes")

    def stats(self) -> Dict:
        """Graph statistics."""
        types = {}
        for _, d in self.graph.nodes(data=True):
            t = d.get("type", "unknown")
            types[t] = types.get(t, 0) + 1

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "types": types,
            "entities": len(self.entity_to_chunks),
            "sections": len(self.section_to_chunks),
        }


def build_graph_from_json(json_path: str, output_path: Optional[str] = None) -> KnowledgeGraph:
    """Build graph from extraction JSON."""
    import ijson

    kg = KnowledgeGraph()
    chunks = []

    logger.info(f"Loading {json_path}")
    with open(json_path, "rb") as f:
        for chunk in ijson.items(f, "chunks.item"):
            chunks.append(chunk)

    logger.info(f"Building graph from {len(chunks)} chunks")
    kg.build_from_chunks(chunks)

    if output_path:
        kg.save(output_path)

    return kg


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python knowledge_graph.py <extraction.json> [output.pkl]")
        sys.exit(1)

    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else json_path.replace(".json", "_graph.pkl")

    kg = build_graph_from_json(json_path, output_path)
    print(kg.stats())

    # Show top entities
    top = sorted(kg.entity_to_chunks.items(), key=lambda x: len(x[1]), reverse=True)[:15]
    print("\nTop entities:")
    for e, chunks in top:
        print(f"  {e}: {len(chunks)} chunks")
