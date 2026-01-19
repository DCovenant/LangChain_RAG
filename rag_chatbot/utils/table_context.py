"""Context builder for RAG pipeline.
Includes table-title-to-content linking for better answer generation.
"""

import re
from typing import List, Dict, Optional, Set

MAX_CONTEXT_CHARS = 10000  # LongT5 supports 4k tokens (~12k chars)
MIN_SCORE_RATIO = 0.1

SYNONYMS = {
    'legislation': {'regulations', 'act', 'law', 'statute'},
    'regulations': {'legislation', 'rules', 'requirements'},
    'documents': {'papers', 'files', 'specs', 'specifications'},
    'standards': {'specifications', 'requirements', 'documents'},
}


def _filter_by_score(chunks: List[Dict], min_ratio: float = MIN_SCORE_RATIO) -> List[Dict]:
    """Keep only chunks with score >= min_ratio * top_score."""
    if not chunks:
        return chunks

    top_score = max(c.get("final_score", c.get("score", 0)) for c in chunks)
    if top_score <= 0:
        return chunks

    threshold = top_score * min_ratio
    return [c for c in chunks if c.get("final_score", c.get("score", 0)) >= threshold]


def _get_adjacent_chunks(top_chunks: List[Dict], all_results: List[Dict]) -> List[Dict]:
    """Add chunks from adjacent pages of top-scoring documents."""
    if not all_results:
        return top_chunks

    included = set()
    expanded = []

    for chunk in top_chunks:
        doc = chunk.get("file_name", "")
        page = chunk.get("page_number", 0)
        key = (doc, page)
        if key not in included:
            expanded.append(chunk)
            included.add(key)

    for chunk in top_chunks:
        doc = chunk.get("file_name", "")
        page = chunk.get("page_number", 0)

        for adj_page in [page - 1, page + 1]:
            if adj_page < 1:
                continue
            key = (doc, adj_page)
            if key in included:
                continue

            for candidate in all_results:
                if candidate.get("file_name") == doc and candidate.get("page_number") == adj_page:
                    expanded.append(candidate)
                    included.add(key)
                    break

    return expanded


def _extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keywords from text."""
    stopwords = {'the', 'and', 'for', 'with', 'from', 'are', 'was', 'were', 'been', 'table', 'col'}
    text_clean = re.sub(r'<[^>]+>', '', text.lower())
    words = re.findall(r'\b[a-z]{3,}\b', text_clean)
    return {w for w in words if w not in stopwords}


def _expand_with_synonyms(keywords: Set[str]) -> Set[str]:
    """Expand keywords with synonyms."""
    expanded = set(keywords)
    for kw in keywords:
        if kw in SYNONYMS:
            expanded.update(SYNONYMS[kw])
    return expanded


def _score_table_match(table_chunk: Dict, title_keywords: Set[str]) -> int:
    """Score how well a table matches title keywords."""
    if not title_keywords:
        return 0
    
    table_text = table_chunk.get('chunk_text', '')
    table_keywords = _extract_keywords(table_text)
    
    expanded_title = _expand_with_synonyms(title_keywords)
    expanded_table = _expand_with_synonyms(table_keywords)
    
    direct_matches = len(title_keywords & table_keywords)
    synonym_matches = len(expanded_title & expanded_table) - direct_matches
    
    return direct_matches * 2 + synonym_matches


def _find_best_matching_table(
    title: str, 
    page: int, 
    file_name: str, 
    all_results: List[Dict],
    exclude_ids: Set[str]
) -> Optional[Dict]:
    """Find best matching table not already in exclude_ids."""
    title_keywords = _extract_keywords(title)
    
    page_tables = [
        c for c in all_results
        if c.get("page_number") == page
        and c.get("file_name") == file_name
        and c.get("has_table") and c.get("table_data")
        and c.get("chunk_id") not in exclude_ids
    ]
    
    if not page_tables:
        return None
    
    scored = [(tc, _score_table_match(tc, title_keywords)) for tc in page_tables]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    best_table, best_score = scored[0]
    if best_score > 0:
        return best_table
    return None


def _link_table_titles_to_content(chunks: List[Dict], all_results: List[Dict]) -> List[Dict]:
    """
    When a body chunk references a table, find the best matching table.
    """
    if not all_results:
        return chunks

    table_title_pattern = re.compile(r'Table\s+\d+\.?\d*\s*[-–]\s*([^\n|]+)', re.IGNORECASE)
    included_ids = {c.get("chunk_id") for c in chunks}
    additional = []

    for chunk in chunks:
        if chunk.get("has_table"):
            continue

        text = chunk.get("chunk_text", "")
        matches = table_title_pattern.findall(text)
        
        for title in matches:
            title = title.strip()
            page = chunk.get("page_number", 0)
            doc = chunk.get("file_name", "")
            
            matched_table = _find_best_matching_table(title, page, doc, all_results, included_ids)
            if matched_table:
                additional.append(matched_table)
                included_ids.add(matched_table.get("chunk_id"))

    return chunks + additional


def build_context(chunks: List[Dict], all_results: Optional[List[Dict]] = None) -> str:
    """Build context with source attribution."""
    chunks = _filter_by_score(chunks)

    if all_results:
        chunks = _get_adjacent_chunks(chunks, all_results)

    chunks = sorted(chunks, key=lambda c: c.get("final_score", c.get("score", 0)), reverse=True)

    parts = []
    chars = 0

    for c in chunks:
        content = c.get('chunk_text', '')
        if not content:
            continue

        doc = c.get("file_name", "unknown")
        page = c.get("page_number", 0)
        section = c.get("section", "")
        section_str = f" | {section}" if section else ""
        header = f"[{doc} | Page {page}{section_str}]"
        entry = f"{header}\n{content}\n"

        remaining = MAX_CONTEXT_CHARS - chars
        if remaining <= 0:
            break

        if len(entry) > remaining:
            entry = entry[:remaining]

        parts.append(entry)
        chars += len(entry)

    return "\n".join(parts)


def _table_to_text(table_data) -> str:
    """Convert table to plain text enumeration."""
    if not table_data:
        return ""

    if isinstance(table_data, dict) and 'headers' in table_data:
        return _semantic_to_text(table_data)

    if isinstance(table_data, dict) and 'table_data' in table_data:
        return _coordinate_to_text(table_data)

    if isinstance(table_data, list) and table_data:
        return _list_to_text(table_data)

    return str(table_data)


def _semantic_to_text(table_data: Dict) -> str:
    """Convert semantic table to text lines."""
    headers = table_data.get('headers', [])
    rows = table_data.get('rows', [])

    if not rows:
        return ""

    lines = []
    for row in rows:
        parts = []
        for h in headers:
            cell = row.get(h, {})
            val = cell.get('value', '') if isinstance(cell, dict) else str(cell)
            val = re.sub(r'<[^>]+>', '', str(val)) if val else ''
            if val:
                clean_h = re.sub(r'<[^>]+>', '', str(h)) if h else ''
                if clean_h and not clean_h.startswith('col_'):
                    parts.append(f"{clean_h}: {val}")
                else:
                    parts.append(val)
        if parts:
            lines.append(", ".join(parts))

    return "\n".join(lines)


def _coordinate_to_text(table_info: Dict) -> str:
    """Convert coordinate table to text."""
    data = table_info.get("table_data", {})
    num_rows = table_info.get("rows", 0)
    num_cols = table_info.get("cols", 0)

    if not data:
        return ""

    headers = [str(data.get(f"{c},0", "")) for c in range(num_cols)]
    lines = []

    for r in range(1, num_rows):
        parts = []
        for c in range(num_cols):
            val = str(data.get(f"{c},{r}", ""))
            if val and headers[c]:
                parts.append(f"{headers[c]}: {val}")
        if parts:
            lines.append(", ".join(parts))

    return "\n".join(lines)


def _list_to_text(table_data: List[Dict]) -> str:
    """Convert list of dicts to text."""
    lines = []
    for row in table_data:
        parts = [f"{k}: {v}" for k, v in row.items() if v and k not in ('source', 'role')]
        if parts:
            lines.append(", ".join(parts))
    return "\n".join(lines)