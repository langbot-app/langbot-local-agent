"""Knowledge retrieval utilities for local-agent runner."""

from __future__ import annotations

import typing

from langbot_plugin.api.proxies.agent_run_api import AgentRunAPIProxy, PermissionDeniedError

from .messages import format_rag_results


async def retrieve_from_knowledge_bases(
    api: AgentRunAPIProxy,
    kb_ids: list[str],
    query_text: str,
    top_k: int = 5,
    rerank_model_id: str | None = None,
    rerank_top_k: int = 5,
) -> str:
    """Retrieve context from authorized knowledge bases with optional reranking.

    Only retrieves from KBs that are in ctx.resources.knowledge_bases.

    Args:
        api: AgentRunAPIProxy for authorized access
        kb_ids: Knowledge base IDs to query (must be in allowed set)
        query_text: Query text for retrieval
        top_k: Number of results per KB
        rerank_model_id: Optional rerank model UUID for re-scoring results
        rerank_top_k: Number of top results to keep after reranking

    Returns:
        Formatted context string for RAG prompt
    """
    if not kb_ids or not query_text:
        return ""

    all_results: list[dict[str, typing.Any]] = []

    for kb_id in kb_ids:
        try:
            results = await api.retrieve_knowledge(
                kb_id=kb_id,
                query_text=query_text,
                top_k=top_k,
            )
            if results:
                all_results.extend(results)
        except PermissionDeniedError:
            # KB not authorized - skip
            continue
        except Exception:
            # KB retrieval failed - skip and continue
            continue

    if not all_results:
        return ""

    # Rerank step: re-score results using a rerank model if configured
    if rerank_model_id and all_results:
        try:
            # Extract text content from results for reranking
            doc_texts = []
            for entry in all_results:
                # Handle different result formats
                if isinstance(entry, dict):
                    content = entry.get('content', '')
                    if isinstance(content, str):
                        doc_texts.append(content)
                    elif isinstance(content, list):
                        # Multiple content parts - join text parts
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                text_parts.append(part.get('text', ''))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        doc_texts.append(' '.join(text_parts))
                elif hasattr(entry, 'content'):
                    # Object with content attribute
                    content = entry.content
                    if isinstance(content, str):
                        doc_texts.append(content)
                    elif isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if hasattr(part, 'type') and part.type == 'text' and hasattr(part, 'text'):
                                text_parts.append(part.text)
                        doc_texts.append(' '.join(text_parts))

            if doc_texts:
                # Invoke rerank model
                scores = await api.invoke_rerank(
                    rerank_model_uuid=rerank_model_id,
                    query=query_text,
                    documents=doc_texts,
                    top_k=rerank_top_k,
                )

                # Sort results by rerank scores
                if scores:
                    # Get indices sorted by relevance score (already sorted by invoke_rerank)
                    top_indices = [s['index'] for s in scores if s['index'] < len(all_results)]
                    all_results = [all_results[i] for i in top_indices]
        except PermissionDeniedError:
            # Rerank model not authorized - use original order
            pass
        except Exception:
            # Rerank failed - use original order
            pass

    return format_rag_results(all_results)
