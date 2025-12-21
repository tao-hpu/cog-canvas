"""Chat API endpoints with real LLM streaming and CogCanvas integration."""

import json
import os
import asyncio
from typing import AsyncGenerator, Optional
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from models import ChatRequest, CanvasObjectResponse
from routes.canvas import get_canvas

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Lazy-initialized OpenAI client
_client = None

# Chat model for response generation (use MODEL_DEFAULT for faster responses)
CHAT_MODEL = os.environ.get("MODEL_DEFAULT", "gpt-4o-mini")


def get_openai_client():
    """Get or create OpenAI client lazily."""
    global _client
    if _client is None:
        from openai import OpenAI

        # Use API_KEY from .env (compatible with OpenAI API format)
        api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("API_BASE") or os.environ.get("OPENAI_API_BASE")

        if not api_key:
            raise ValueError(
                "API_KEY environment variable is required. "
                "Please set it in .env file."
            )

        _client = OpenAI(api_key=api_key, base_url=api_base)
    return _client


async def generate_chat_stream(
    user_message: str, session_id: str, cogcanvas_enabled: bool = True
) -> AsyncGenerator[str, None]:
    """
    Generate streaming chat response with SSE events using real LLM.

    Flow:
    1. Retrieve relevant context from canvas
    2. Inject context into system prompt
    3. Stream response from LLM
    4. Extract objects from conversation
    5. Return extracted objects

    Yields events:
    - type: "token" - LLM generated token
    - type: "done" - Response complete
    - type: "extraction" - Extracted canvas objects
    """
    try:
        canvas = get_canvas(session_id)

        # Step 1: Retrieve relevant context from canvas
        context_prompt = ""
        if cogcanvas_enabled and canvas.size > 0:
            retrieval_result = canvas.retrieve(
                query=user_message,
                top_k=5,
                method="semantic",
                include_related=False,  # Strict top-k limit to prevent result explosion
            )

            if retrieval_result.objects:
                # Step 2: Inject context into prompt
                context_prompt = canvas.inject(
                    result=retrieval_result,
                    format="markdown",
                    max_tokens=500,  # Token budget for context
                    strategy="relevance",
                )

        # Build messages for LLM
        system_message = """You are a helpful AI assistant. Be concise and informative.
When making decisions or commitments, be explicit about them.
When noting important facts or TODOs, state them clearly."""

        if context_prompt:
            system_message += f"\n\n{context_prompt}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        # If context was injected, send it to frontend as a separate event
        if context_prompt and retrieval_result and retrieval_result.objects:
            retrieved_objs = [
                CanvasObjectResponse.from_canvas_object(obj).model_dump()
                for obj in retrieval_result.objects
            ]
            yield json.dumps(
                {
                    "type": "retrieval",
                    "objects": retrieved_objs,
                    "count": len(retrieved_objs),
                }
            ) + "\n"

        # Step 3: Stream response from LLM
        full_response = ""
        client = get_openai_client()

        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1024,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                yield json.dumps({"type": "token", "content": token}) + "\n"
                await asyncio.sleep(0.02)  # Small delay for smoother UI streaming

        # Step 4: Extract canvas objects from this turn
        if cogcanvas_enabled:
            # Signal extraction start - yield and flush immediately
            yield json.dumps(
                {
                    "type": "extracting",
                }
            ) + "\n"
            await asyncio.sleep(0)  # Force flush to client

            extraction_result = canvas.extract(
                user=user_message, assistant=full_response
            )

            if extraction_result.objects:
                # Step 5: Auto-link objects based on semantic similarity
                try:
                    canvas.auto_link(reference_threshold=0.7, causal_threshold=0.6)
                except Exception as link_error:
                    print(f"Auto-link warning: {link_error}")

                # Step 6: Return extracted objects (with updated relations)
                # Re-fetch objects to get updated relations
                updated_objects = [
                    canvas.get(obj.id) for obj in extraction_result.objects
                ]
                extracted_objs = [
                    CanvasObjectResponse.from_canvas_object(obj).model_dump()
                    for obj in updated_objects
                    if obj
                ]
                yield json.dumps(
                    {
                        "type": "extraction",
                        "objects": extracted_objs,
                        "count": len(extracted_objs),
                    }
                ) + "\n"

        # Signal complete (after extraction if enabled)
        yield json.dumps({"type": "done", "content": full_response}) + "\n"

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        print(f"Chat error: {error_detail}")
        yield json.dumps(
            {"type": "error", "error": str(e), "detail": error_detail}
        ) + "\n"


@router.post("")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response with Server-Sent Events (SSE).

    Uses real LLM (GPT-4o-mini by default) with CogCanvas integration:
    1. Retrieves relevant context from canvas
    2. Injects context into LLM prompt
    3. Streams LLM response
    4. Extracts cognitive objects from conversation
    """
    return EventSourceResponse(
        generate_chat_stream(
            user_message=request.message,
            session_id=request.session_id or "default",
            cogcanvas_enabled=(
                request.cogcanvas_enabled
                if request.cogcanvas_enabled is not None
                else True
            ),
        )
    )


@router.post("/simple")
async def chat_simple(request: ChatRequest):
    """
    Simple non-streaming chat endpoint for testing.

    Returns complete response at once.
    """
    canvas = get_canvas(request.session_id or "default")

    # Retrieve context
    context_prompt = ""
    if canvas.size > 0:
        retrieval_result = canvas.retrieve(
            query=request.message, top_k=5, method="semantic"
        )
        if retrieval_result.objects:
            context_prompt = canvas.inject(
                result=retrieval_result, format="markdown", max_tokens=500
            )

    # Build messages
    system_message = "You are a helpful AI assistant."
    if context_prompt:
        system_message += f"\n\n{context_prompt}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": request.message},
    ]

    # Get response
    client = get_openai_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    assistant_response = response.choices[0].message.content

    # Extract objects
    extraction_result = canvas.extract(
        user=request.message, assistant=assistant_response
    )

    extracted_objs = None
    if extraction_result.objects:
        extracted_objs = [
            CanvasObjectResponse.from_canvas_object(obj)
            for obj in extraction_result.objects
        ]

    return {
        "role": "assistant",
        "content": assistant_response,
        "extracted_objects": extracted_objs,
    }
