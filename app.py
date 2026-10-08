import json
import uuid
from pathlib import Path
from datetime import date

import litellm
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tools import TOOLS, run_tool


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a research-gap discovery assistant. "

    "When the user asks for research gaps on a topic, complete the full workflow automatically: "
    "first call search_papers, then call extract_paper_profiles, then call find_research_gaps. "

    "After search_papers, do not list every retrieved paper unless the user explicitly asks for the paper list. "
    "Do not ask whether to continue; proceed automatically through the workflow. "

    "Do not infer methods, findings, limitations, or relevance from paper titles alone. "
    "Use extract_paper_profiles before making claims about paper content. "

    "Never invent papers, methods, findings, limitations, or research gaps. "
    "Treat all gaps as candidate gaps inferred from the retrieved sample, "
    "not as confirmed absence of prior research. "
    "Candidate gaps may be classified as method coverage gaps, recurring limitations, "
    "or conflicting findings."
)

MAX_TOOL_ROUNDS = 6

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def format_gap_result(result: dict) -> str:
    gaps = result.get("gaps", [])

    if not gaps:
        return (
            "No clear candidate gaps were identified "
            "from the retrieved sample."
        )

    sections = []

    for gap in gaps:
        gap_type = (
            gap.get("gap_type", "candidate_gap")
            .replace("_", " ")
            .title()
        )

        description = gap.get(
            "description",
            "No description available."
        )

        sections.append(
            f"{gap_type}\n{description}"
        )

    sections.append(
        "These candidate gaps are inferred from the retrieved sample "
        "and should be validated against a broader literature review."
    )

    return "\n\n".join(sections)

# ---------------------------------------------------------
# Harness
# ---------------------------------------------------------

def run_agent(session: dict) -> tuple[str, list[dict]]:
    """
    Run the agent until Gemini returns a normal response without requesting another tool.

    Full research data is stored in session state. Only compact tool results are added to model history.

    Returns:
        final response text
        list of tool calls made
    """

    messages = session["messages"]
    tool_calls = []

    for _ in range(MAX_TOOL_ROUNDS):

        reply = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=messages,
            tools=TOOLS,
        ).choices[0].message

        # Save Gemini's reply into the conversation history.
        messages.append(reply.model_dump())

        # If Gemini does not request a tool,
        # the agent has finished.
        if not reply.tool_calls:
            return reply.content, tool_calls

        # Execute requested tools

        for call in reply.tool_calls:

            args = json.loads(call.function.arguments)

            raw_result = run_tool(
                call.function.name,
                args,
                session=session,
            )

            # Default:
            # tool output shown to Gemini is identical
            # to the tool's raw return value.
            model_result = raw_result

            # -----------------------------------------
            # Special handling for search_papers
            # -----------------------------------------

            if call.function.name == "search_papers":

                try:
                    parsed = json.loads(raw_result)

                    # If the tool itself returned an error,
                    # send the error directly to Gemini.
                    if "error" not in parsed:

                        session["research_query"] = parsed.get("query")

                        # Store full papers in backend session state.
                        for paper in parsed.get("papers", []):
                            paper_id = paper.get("paper_id")

                            if paper_id:
                                session["papers"][paper_id] = paper

                        # Build lightweight result for Gemini.
                        compact_result = {
                            "count": parsed.get("count"),
                                "year_range": [
                                    parsed.get("start_year"),
                                    date.today().year,
                                ],
                            "warning": parsed.get("warning"),
                        }
                        model_result = json.dumps(
                            compact_result
                        )

                except (
                    json.JSONDecodeError,
                    TypeError,
                ):
                    # If parsing somehow fails,
                    # fall back to the original result.
                    model_result = raw_result

            # -----------------------------------------
            # Record tool call for UI / grader
            # -----------------------------------------

            tool_calls.append({
                "name": call.function.name,
                "args": args,
                "result": model_result,
            })

            # Only compact tool output enters
            # Gemini conversation history.
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": model_result,
            })

            if call.function.name == "find_research_gaps":
                try:
                    gap_data = json.loads(raw_result)

                    if "error" not in gap_data:
                        final_response = format_gap_result(gap_data)

                        messages.append({
                            "role": "assistant",
                            "content": final_response,
                        })

                        return final_response, tool_calls

                except json.JSONDecodeError:
                    pass

    return (
        "Sorry, I hit my tool-call limit before finishing.",
        tool_calls,
    )


# ---------------------------------------------------------
# Session Store
# ---------------------------------------------------------

# Each session stores:
#
# messages:
#     Conversation history visible to Gemini.
#
# papers:
#     Full OpenAlex paper records including abstracts.
#
# profiles:
#     Structured paper profiles produced by Tool 2 later.
#
sessions: dict[str, dict] = {}


def create_session() -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ],
        "research_query": None,
        "papers": {},
        "profiles": {},
        "gaps": [],
    }


# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/")
def index():
    return FileResponse(
        Path(__file__).parent / "index.html"
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    # ---------------------------------------------
    # Get or create session
    # ---------------------------------------------

    session_id = (
        request.session_id
        or str(uuid.uuid4())
    )

    if session_id not in sessions:
        sessions[session_id] = create_session()

    session = sessions[session_id]

    # ---------------------------------------------
    # Add user message
    # ---------------------------------------------

    session["messages"].append({
        "role": "user",
        "content": request.message,
    })

    # ---------------------------------------------
    # Run agent
    # ---------------------------------------------

    try:
        response, tool_calls = run_agent(
            session
        )

    except Exception as e:

        response = (
            "Model call failed: "
            f"{type(e).__name__}: "
            f"{str(e)[:300]}"
        )

        tool_calls = []

    return ChatResponse(
        response=response,
        session_id=session_id,
        tool_calls=tool_calls,
    )


@app.post("/clear")
def clear(
    session_id: str | None = None
):

    sessions.pop(
        session_id,
        None,
    )

    return {
        "status": "ok"
    }


# ---------------------------------------------------------
# Run locally
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )