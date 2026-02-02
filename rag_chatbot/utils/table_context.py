"""Context builder for RAG pipeline.
Includes table-title-to-content linking for better answer generation.
"""

import re
from typing import List, Dict, Optional, Set
from config import MAX_CONTEXT_CHARS, MIN_SCORE_RATIO

def _filter_by_score(chunks: List[Dict]) -> List[Dict]:
    """Keep only chunks with score >= min_ratio * top_score."""
    if not chunks:
        return chunks

    top_score = max(c.get("final_score", c.get("score", 0)) for c in chunks)
    if top_score <= 0:
        return chunks

    threshold = top_score * MIN_SCORE_RATIO
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


def _score_table_match(table_chunk: Dict, title_keywords: Set[str]) -> int:
    """Score how well a table matches title keywords."""
    if not title_keywords:
        return 0
    table_text = table_chunk.get('chunk_text', '')
    table_keywords = _extract_keywords(table_text)
    return len(title_keywords & table_keywords)


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
    """Build context with source attribution and section summaries."""
    chunks = _filter_by_score(chunks)
    if not chunks:
        return ""

    if all_results:
        chunks = _get_adjacent_chunks(chunks, all_results)

    chunks = sorted(chunks, key=lambda c: c.get("final_score", c.get("score", 0)), reverse=True)

    parts = []
    chars = 0
    seen_summaries: Set[str] = set()

    for c in chunks:
        content = c.get('chunk_text', '')
        if not content:
            continue

        doc = c.get("file_name", "unknown")
        page = c.get("page_number", 0)
        section_path = c.get("section_path", "") or c.get("section", "")
        section_str = f" | {section_path}" if section_path else ""
        header = f"[{doc} | Page {page}{section_str}]"

        # Include section summary and context once per section
        section_key = section_path or c.get("section", "")
        summary_key = f"{doc}:{section_key}"
        summary_line = ""
        if summary_key not in seen_summaries:
            summary = c.get("section_summary", "")
            section_context = c.get("section_context", "")
            if summary:
                summary_line = f"Section summary: {summary}\n"
            if section_context:
                summary_line += f"Context: {section_context}\n"
            if summary_line:
                seen_summaries.add(summary_key)

        entry = f"{header}\n{summary_line}{content}\n"

        remaining = MAX_CONTEXT_CHARS - chars
        if remaining <= 0:
            break

        if len(entry) > remaining:
            entry = entry[:remaining]

        parts.append(entry)
        chars += len(entry)

    return "\n".join(parts)
