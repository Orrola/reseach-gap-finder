import json
import os
from dotenv import load_dotenv
from datetime import date

import litellm
import requests

from collections import Counter, defaultdict

load_dotenv()

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

PROFILE_MODEL = "vertex_ai/gemini-3.5-flash-lite"
PROFILE_BATCH_SIZE = 5

def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """
    Reconstruct a normal abstract string from OpenAlex's
    abstract_inverted_index format.

    Example OpenAlex format:
    {
        "Generative": [0],
        "AI": [1, 10],
        ...
    }

    Returns:
        Reconstructed abstract text, or None if no abstract is available.
    """
    if not inverted_index:
        return None

    positions = []

    for word, word_positions in inverted_index.items():
        for position in word_positions:
            positions.append((position, word))

    positions.sort(key=lambda x: x[0])

    return " ".join(word for _, word in positions)


def search_papers(
    query: str,
    years: int = 3,
    max_results: int = 15,
) -> str:
    """
    Search OpenAlex for recent journal articles with available abstracts.

    By default, the search covers the current calendar year and the
    previous two calendar years.

    Example:
        If the current year is 2026 and years=3,
        papers from 2024 onward are searched.

    Args:
        query:
            Research topic or research question.

        years:
            Number of calendar years to search.
            Defaults to 3.

        max_results:
            Number of papers to retrieve.
            Must be between 10 and 25.
            Defaults to 15.

    Returns:
        A JSON string containing:

        - model_results:
            Lightweight metadata intended for the main Gemini conversation.

        - papers:
            Full paper records including abstracts.
            TEMPORARY during development.
            These will later be stored in backend/session state instead
            of being returned to the main model.

        - api_cost_usd:
            OpenAlex-reported API request cost when available.
    """

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not isinstance(query, str) or not query.strip():
        return json.dumps({
            "error": "Query cannot be empty."
        })

    if not isinstance(years, int) or years < 1:
        return json.dumps({
            "error": "years must be an integer of at least 1."
        })

    if not isinstance(max_results, int):
        return json.dumps({
            "error": "max_results must be an integer."
        })

    if max_results < 10 or max_results > 25:
        return json.dumps({
            "error": "max_results must be between 10 and 25."
        })

    # -----------------------------------------------------
    # Read OpenAlex API key from environment
    # -----------------------------------------------------

    api_key = os.getenv("OPENALEX_API_KEY")

    if not api_key:
        return json.dumps({
            "error": (
                "OPENALEX_API_KEY is not set. "
                "Set it as an environment variable before running the app."
            )
        })

    # -----------------------------------------------------
    # Build publication-year window
    # -----------------------------------------------------

    today = date.today()

    # Example:
    # current year = 2026
    # years = 3
    # start year = 2024
    start_year = today.year - years + 1

    from_date = f"{start_year}-01-01"

    # -----------------------------------------------------
    # Build OpenAlex filters
    # -----------------------------------------------------

    filters = ",".join([
        f"from_publication_date:{from_date}",
        "type:article",
        "primary_location.source.type:journal",
        "has_abstract:true",
    ])

    params = {
        "search": query,
        "filter": filters,
        "per_page": max_results,
        "api_key": api_key,

        # Only request the fields needed by this project.
        "select": ",".join([
            "id",
            "title",
            "publication_year",
            "primary_location",
            "doi",
            "cited_by_count",
            "abstract_inverted_index",
        ]),
    }

    # -----------------------------------------------------
    # Send request to OpenAlex
    # -----------------------------------------------------

    try:
        response = requests.get(
            OPENALEX_WORKS_URL,
            params=params,
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        api_cost_usd = (
            data
            .get("meta", {})
            .get("cost_usd")
        )

        papers = []
        model_results = []

        # -------------------------------------------------
        # Clean and normalize paper records
        # -------------------------------------------------

        for item in results:

            # -----------------------------
            # OpenAlex work ID
            # -----------------------------

            paper_id = item.get("id")

            if paper_id:
                # Convert:
                # https://openalex.org/W123456
                # into:
                # W123456
                paper_id = paper_id.rsplit("/", 1)[-1]

            # -----------------------------
            # Venue / journal
            # -----------------------------

            primary_location = (
                item.get("primary_location")
                or {}
            )

            source = (
                primary_location.get("source")
                or {}
            )

            venue = source.get("display_name")

            # -----------------------------
            # Reconstruct abstract
            # -----------------------------

            abstract = _reconstruct_abstract(
                item.get("abstract_inverted_index")
            )

            # -----------------------------
            # Full backend record
            # -----------------------------

            paper = {
                "paper_id": paper_id,
                "title": item.get("title"),
                "year": item.get("publication_year"),
                "venue": venue,
                "doi": item.get("doi"),
                "cited_by_count": item.get(
                    "cited_by_count",
                    0,
                ),
                "abstract": abstract,
            }

            papers.append(paper)

            # -----------------------------
            # Lightweight model-facing record
            # -----------------------------

            model_results.append({
                "paper_id": paper_id,
                "title": item.get("title"),
                "year": item.get("publication_year"),
                "venue": venue,
            })

        # -------------------------------------------------
        # Handle small result sets gracefully
        # -------------------------------------------------

        warning = None

        if len(papers) < 10:
            warning = (
                f"Only {len(papers)} eligible journal articles "
                f"were found for '{query}'. "
                "Consider broader keywords or a longer year range."
            )

        # -------------------------------------------------
        # Development-stage return
        # -------------------------------------------------

        return json.dumps({
            "query": query,
            "years": years,
            "start_year": start_year,
            "max_results": max_results,
            "count": len(papers),
            "api_cost_usd": api_cost_usd,

            # This lightweight version is what the main
            # Gemini conversation should eventually receive.
            "model_results": model_results,

            # TODO:
            # DEVELOPMENT ONLY.
            #
            # Before integrating this tool into the final agent,
            # store full paper records in session/backend state
            # and DO NOT return them to the main Gemini conversation.
            #
            # Tool 2 should later retrieve papers from session state
            # by paper_id.
            "papers": papers,

            "warning": warning,
        })

    # -----------------------------------------------------
    # Error handling
    # -----------------------------------------------------

    except requests.Timeout:
        return json.dumps({
            "error": (
                "OpenAlex request timed out. "
                "Please try again."
            )
        })

    except requests.HTTPError as e:

        if (
            e.response is not None
            and e.response.status_code == 429
        ):
            return json.dumps({
                "error": (
                    "OpenAlex rate limit reached. "
                    "Please retry later."
                )
            })

        if (
            e.response is not None
            and e.response.status_code == 401
        ):
            return json.dumps({
                "error": (
                    "OpenAlex authentication failed. "
                    "Check the OPENALEX_API_KEY environment variable."
                )
            })

        return json.dumps({
            "error": (
                "OpenAlex request failed: "
                f"{str(e)}"
            )
        })

    except requests.RequestException as e:
        return json.dumps({
            "error": (
                "OpenAlex request failed: "
                f"{str(e)}"
            )
        })

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as e:
        return json.dumps({
            "error": (
                "Could not parse OpenAlex response: "
                f"{str(e)}"
            )
        })

# ---------------------------------------------------------
# Tool 2: Extract structured paper profiles
# ---------------------------------------------------------

PROFILE_MODEL = "vertex_ai/gemini-3.5-flash-lite"
PROFILE_BATCH_SIZE = 5


METHOD_CATEGORIES = [
    "experimental",
    "observational",
    "survey",
    "qualitative",
    "simulation_modeling",
    "review_theoretical",
    "mixed_methods",
    "other",
]


SETTING_CATEGORIES = [
    "workplace_organization",
    "education",
    "healthcare",
    "technology_software",
    "public_sector",
    "finance_business",
    "consumer_general_population",
    "other",
]


FINDING_DIRECTIONS = [
    "positive",
    "negative",
    "mixed",
    "null",
    "descriptive",
    "unclear",
]


def _strip_json_fence(text: str) -> str:
    """
    Remove Markdown JSON fences if the model returns them.
    """

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def _validate_profile(
    profile: dict,
    valid_paper_ids: set[str],
) -> dict | None:
    """
    Validate and normalize one model-generated paper profile.
    """

    paper_id = profile.get("paper_id")

    if paper_id not in valid_paper_ids:
        return None

    method = profile.get(
        "method_category",
        "other",
    )

    if method not in METHOD_CATEGORIES:
        method = "other"

    setting = profile.get(
        "setting_category",
        "other",
    )

    if setting not in SETTING_CATEGORIES:
        setting = "other"

    direction = profile.get(
        "finding_direction",
        "unclear",
    )

    if direction not in FINDING_DIRECTIONS:
        direction = "unclear"

    setting_detail = (
        profile.get("setting_detail")
        or "not stated"
    )

    main_finding = (
        profile.get("main_finding")
        or "not stated"
    )

    explicit_limitation = (
        profile.get("explicit_limitation")
        or "not stated"
    )

    return {
        "paper_id": paper_id,
        "method_category": method,
        "setting_category": setting,
        "setting_detail": setting_detail,
        "main_finding": main_finding,
        "finding_direction": direction,
        "explicit_limitation": explicit_limitation,
    }


def _extract_batch_profiles(
    research_query: str,
    papers: list[dict],
) -> list[dict]:
    """
    Use Gemini to extract structured profiles
    from one batch of paper abstracts.
    """

    paper_payload = []

    for paper in papers:
        paper_payload.append({
            "paper_id": paper["paper_id"],
            "title": paper.get("title"),
            "year": paper.get("year"),
            "abstract": paper.get("abstract"),
        })

    system_prompt = """
You extract structured research-paper information from abstracts.

You must use ONLY information explicitly supported by the title and abstract.
Do not use outside knowledge.
Do not guess missing information.

Return ONLY a valid JSON array.
Do not include Markdown or explanatory text.

For every paper, return exactly these fields:

paper_id
method_category
setting_category
setting_detail
main_finding
finding_direction
explicit_limitation

Allowed method_category values:
experimental
observational
survey
qualitative
simulation_modeling
review_theoretical
mixed_methods
other

Allowed setting_category values:
workplace_organization
education
healthcare
technology_software
public_sector
finance_business
consumer_general_population
other

Allowed finding_direction values:
positive
negative
mixed
null
descriptive
unclear

Rules:

1. method_category:
Choose the closest allowed category.
If the abstract does not provide enough information, use "other".

2. setting_category:
Choose the closest allowed category.

3. setting_detail:
Use a short phrase of no more than 8 words.
Examples:
"US software developers"
"university students in China"
"public-sector employees"

If not stated, return "not stated".

4. main_finding:
Maximum 20 words.
Only summarize a finding explicitly stated in the abstract.
If no finding is stated, return "not stated".

5. finding_direction:
Use positive, negative, mixed, null, descriptive, or unclear.
Use "unclear" when the abstract does not clearly support a direction.

6. explicit_limitation:
Maximum 20 words.
ONLY extract a limitation, unresolved issue, or future-work statement
if it is explicitly stated in the abstract.

Do NOT infer a limitation yourself.

If none is explicitly stated, return exactly:
"not stated"
""".strip()

    user_prompt = (
        f"Research topic/question:\n"
        f"{research_query}\n\n"
        f"Papers:\n"
        f"{json.dumps(paper_payload)}"
    )

    reply = litellm.completion(
        model=PROFILE_MODEL,
        vertex_location="global",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    ).choices[0].message.content

    cleaned = _strip_json_fence(reply)

    parsed = json.loads(cleaned)

    if not isinstance(parsed, list):
        raise ValueError(
            "Profile extractor did not return a JSON array."
        )

    return parsed


def extract_paper_profiles(
    session: dict,
    paper_ids: list[str] | None = None,
) -> str:
    """
    Extract structured profiles from paper abstracts stored in session.

    Papers are processed in batches of five using Gemini.

    Args:
        session:
            Current backend session containing papers and profiles.

        paper_ids:
            Optional list of OpenAlex paper IDs.
            If omitted, all papers currently stored in the session
            are processed.

    Returns:
        A compact JSON summary for the main agent.

        Full profiles are stored in:
            session["profiles"]
    """

    papers_store = session.get("papers", {})
    profiles_store = session.get("profiles", {})
    research_query = session.get("research_query")

    if not research_query:
        return json.dumps({
            "error": (
                "No research query is stored in this session. "
                "Run search_papers first."
            )
        })

    if not papers_store:
        return json.dumps({
            "error": (
                "No papers are stored in this session. "
                "Run search_papers first."
            )
        })

    # If Gemini does not specify IDs,
    # process every paper currently stored.
    if paper_ids is None:
        requested_ids = list(
            papers_store.keys()
        )
    else:
        requested_ids = paper_ids

    valid_ids = [
        paper_id
        for paper_id in requested_ids
        if paper_id in papers_store
    ]

    missing_ids = [
        paper_id
        for paper_id in requested_ids
        if paper_id not in papers_store
    ]

    if not valid_ids:
        return json.dumps({
            "error": (
                "None of the requested paper IDs "
                "exist in the current session."
            ),
            "missing_paper_ids": missing_ids,
        })

    succeeded_ids = []
    failed_ids = []

    # -----------------------------------------------------
    # Process five papers per Gemini call
    # -----------------------------------------------------

    for start in range(
        0,
        len(valid_ids),
        PROFILE_BATCH_SIZE,
    ):

        batch_ids = valid_ids[
            start:start + PROFILE_BATCH_SIZE
        ]

        batch_papers = [
            papers_store[paper_id]
            for paper_id in batch_ids
        ]

        batch_success = False

        # Retry JSON extraction once if parsing fails.
        for attempt in range(2):

            try:
                raw_profiles = (
                    _extract_batch_profiles(
                        research_query,
                        batch_papers,
                    )
                )

                expected_ids = set(batch_ids)

                validated_profiles = []

                for profile in raw_profiles:

                    validated = _validate_profile(
                        profile,
                        expected_ids,
                    )

                    if validated:
                        validated_profiles.append(
                            validated
                        )

                returned_ids = {
                    profile["paper_id"]
                    for profile
                    in validated_profiles
                }

                # Store successful profiles.
                for profile in validated_profiles:

                    paper_id = profile["paper_id"]

                    profiles_store[
                        paper_id
                    ] = profile

                    succeeded_ids.append(
                        paper_id
                    )

                # Any paper missing from the model response
                # is considered failed.
                for paper_id in batch_ids:

                    if (
                        paper_id
                        not in returned_ids
                    ):
                        failed_ids.append(
                            paper_id
                        )

                batch_success = True
                break

            except (
                json.JSONDecodeError,
                ValueError,
                TypeError,
            ):
                # Retry once.
                continue

            except Exception:
                # Model/API error:
                # do not crash the whole agent.
                break

        if not batch_success:

            for paper_id in batch_ids:
                if paper_id not in failed_ids:
                    failed_ids.append(
                        paper_id
                    )

    # Make sure the updated dictionary
    # remains attached to session.
    session["profiles"] = profiles_store

    # -----------------------------------------------------
    # Build lightweight aggregate statistics
    # -----------------------------------------------------

    processed_profiles = [
        profiles_store[paper_id]
        for paper_id in succeeded_ids
        if paper_id in profiles_store
    ]

    method_counts = {}

    setting_counts = {}

    for profile in processed_profiles:

        method = profile[
            "method_category"
        ]

        setting = profile[
            "setting_category"
        ]

        method_counts[method] = (
            method_counts.get(
                method,
                0,
            )
            + 1
        )

        setting_counts[setting] = (
            setting_counts.get(
                setting,
                0,
            )
            + 1
        )

    # -----------------------------------------------------
    # Compact result returned to main Gemini
    # -----------------------------------------------------

    return json.dumps({
        "requested": len(requested_ids),
        "succeeded": len(
            set(succeeded_ids)
        ),
        "failed": len(
            set(failed_ids)
        ),
        "failed_paper_ids": sorted(
            set(failed_ids)
        ),
        "method_counts": method_counts,
        "setting_counts": setting_counts,
    })


# ---------------------------------------------------------
# Tool 3 helpers
# ---------------------------------------------------------


def _build_coverage_matrix(
    profiles: list[dict],
) -> dict:
    """
    Build a method x setting coverage matrix.

    Returns:
        {
            method_category: {
                setting_category: count
            }
        }
    """

    methods = sorted({
        profile["method_category"]
        for profile in profiles
    })

    settings = sorted({
        profile["setting_category"]
        for profile in profiles
    })

    matrix = {}

    for method in methods:

        matrix[method] = {}

        for setting in settings:

            matrix[method][setting] = 0

    for profile in profiles:

        method = profile["method_category"]
        setting = profile["setting_category"]

        matrix[method][setting] += 1

    return matrix


def _find_sparse_combinations(
    coverage_matrix: dict,
) -> list[dict]:
    """
    Find method x setting combinations represented
    by zero or one paper in the retrieved sample.
    """

    sparse = []

    for method, settings in coverage_matrix.items():

        for setting, count in settings.items():

            if count <= 1:

                sparse.append({
                    "method_category": method,
                    "setting_category": setting,
                    "count": count,
                })

    return sparse


def _normalize_limitation(
    text: str,
) -> str:
    """
    Normalize a limitation phrase for lightweight counting.

    This is intentionally simple.
    Tool 3 later asks Gemini to interpret recurring themes.
    """

    return (
        text
        .strip()
        .lower()
        .replace(".", "")
        .replace(",", "")
    )


def _collect_limitations(
    profiles: list[dict],
) -> list[dict]:
    """
    Collect explicit limitations that were actually stated
    in abstracts.

    Returns limitation text with paper IDs.
    """

    limitation_groups = defaultdict(list)

    for profile in profiles:

        limitation = profile.get(
            "explicit_limitation",
            "not stated",
        )

        if (
            not limitation
            or limitation == "not stated"
        ):
            continue

        normalized = _normalize_limitation(
            limitation
        )

        limitation_groups[
            normalized
        ].append(
            profile["paper_id"]
        )

    results = []

    for limitation, paper_ids in limitation_groups.items():

        results.append({
            "limitation": limitation,
            "count": len(paper_ids),
            "paper_ids": paper_ids,
        })

    return sorted(
        results,
        key=lambda x: x["count"],
        reverse=True,
    )


def _find_direction_conflicts(
    profiles: list[dict],
) -> list[dict]:
    """
    Detect settings where multiple finding directions appear.

    Example:
        workplace_organization:
            positive + negative
    """

    setting_directions = defaultdict(
        lambda: defaultdict(list)
    )

    for profile in profiles:

        setting = profile[
            "setting_category"
        ]

        direction = profile[
            "finding_direction"
        ]

        if direction in {
            "unclear",
            "descriptive",
        }:
            continue

        setting_directions[
            setting
        ][
            direction
        ].append(
            profile["paper_id"]
        )

    conflicts = []

    for setting, direction_map in setting_directions.items():

        directions = [
            direction
            for direction, paper_ids
            in direction_map.items()
            if paper_ids
        ]

        if len(directions) >= 2:

            conflicts.append({
                "setting_category": setting,
                "directions": directions,
                "paper_ids_by_direction": dict(
                    direction_map
                ),
            })

    return conflicts

def _interpret_gap_candidates(
    research_query: str,
    coverage_matrix: dict,
    sparse_combinations: list[dict],
    recurring_limitations: list[dict],
    conflicting_findings: list[dict],
    max_gaps: int,
) -> list[dict]:
    """
    Ask Gemini to interpret only the evidence patterns
    already computed by Python.

    Gemini must not invent unsupported gaps.
    """

    system_prompt = """
You are interpreting candidate research gaps from structured
evidence generated from a retrieved literature sample.

You must NOT claim that a research gap is definitively absent
from the full academic literature.

You may only interpret the evidence patterns explicitly provided.


Allowed gap types:

1. method_coverage_gap
   A research method that is absent or represented by only one paper
   within a particular context in the retrieved sample.

2. recurring_limitation
   Multiple retrieved papers explicitly state a similar limitation
   or future-work need.

3. conflicting_findings
   Papers in a similar setting report different finding directions.


Your goal is to SYNTHESIZE the evidence into a small number of
meaningful candidate gaps.

Rules:

- Return at most THREE candidate gaps in total.

- Return at most ONE candidate gap for each gap type.

- If multiple signals belong to the same gap type, combine them into
  one coherent candidate gap instead of listing them separately.

- For example, if experimental and simulation-based methods are both
  underrepresented in the same research context, describe them
  together as one method_coverage_gap.

- Do not mechanically convert every detected signal into a separate gap.

- Select and summarize the broader evidence pattern represented by
  multiple related signals.

- Write each description in natural academic language.

- Vary sentence structure naturally. Do not repeatedly use fixed
  templates such as "X appears underrepresented in the retrieved sample."

- A description may use one or two sentences when needed to clearly
  synthesize the evidence.

- Explain what the evidence pattern suggests without overstating it.

- Never say "no research exists".

- Do not invent evidence that is not present in the input.

- Treat every result as a candidate gap inferred from the retrieved
  sample, not as proof of a gap in the full academic literature.


Return ONLY a valid JSON array.

Each item must contain exactly:

gap_type
description
""".strip()

    evidence_payload = {
        "research_query": research_query,
        "coverage_matrix": coverage_matrix,
        "sparse_combinations": sparse_combinations,
        "recurring_limitations": recurring_limitations,
        "conflicting_findings": conflicting_findings,
        "max_gaps": max_gaps,
    }

    reply = litellm.completion(
        model=PROFILE_MODEL,
        vertex_location="global",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(
                    evidence_payload
                ),
            },
        ],
    ).choices[0].message.content

    cleaned = _strip_json_fence(
        reply
    )

    parsed = json.loads(
        cleaned
    )

    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            "Gap interpreter did not return a JSON array."
        )

    valid_gap_types = {
        "method_coverage_gap",
        "recurring_limitation",
        "conflicting_findings",
    }

    gaps = []

    for gap in parsed[:max_gaps]:

        gap_type = gap.get(
            "gap_type"
        )

        if gap_type not in valid_gap_types:
            continue

        description = gap.get(
            "description"
        )

        if not description:
            continue

        gaps.append({
            "gap_type": gap_type,
            "description": description
        })

    return gaps

def find_research_gaps(
    session: dict,
    paper_ids: list[str] | None = None,
    max_gaps: int = 5,
) -> str:
    """
    Find evidence-backed candidate research gaps
    from structured paper profiles stored in session.

    Args:
        session:
            Current research session.

        paper_ids:
            Optional list of paper IDs to include.
            If omitted, all available profiles are used.

        max_gaps:
            Maximum number of candidate gaps to return.
            Defaults to 5.

    Returns:
        Compact JSON containing candidate gaps.
    """

    research_query = session.get(
        "research_query"
    )

    profiles_store = session.get(
        "profiles",
        {},
    )

    if not research_query:
        return json.dumps({
            "error": (
                "No research query is stored. "
                "Run search_papers first."
            )
        })

    if not profiles_store:
        return json.dumps({
            "error": (
                "No paper profiles are available. "
                "Run extract_paper_profiles first."
            )
        })

    if (
        not isinstance(
            max_gaps,
            int,
        )
        or max_gaps < 1
        or max_gaps > 5
    ):
        return json.dumps({
            "error": (
                "max_gaps must be between 1 and 5."
            )
        })

    # -----------------------------------------------------
    # Choose profiles
    # -----------------------------------------------------

    if paper_ids is None:

        selected_profiles = list(
            profiles_store.values()
        )

    else:

        selected_profiles = [
            profiles_store[paper_id]
            for paper_id in paper_ids
            if paper_id in profiles_store
        ]

    # -----------------------------------------------------
    # Minimum sample-size guardrail
    # -----------------------------------------------------

    if len(selected_profiles) < 8:

        return json.dumps({
            "error": (
                "Too few paper profiles are available "
                "to infer candidate research gaps. "
                "At least 8 profiles are required. "
                "Broaden the search or analyze more papers first."
            ),
            "profile_count": len(
                selected_profiles
            ),
        })

    # -----------------------------------------------------
    # Python evidence analysis
    # -----------------------------------------------------

    coverage_matrix = (
        _build_coverage_matrix(
            selected_profiles
        )
    )

    sparse_combinations = (
        _find_sparse_combinations(
            coverage_matrix
        )
    )

    recurring_limitations = (
        _collect_limitations(
            selected_profiles
        )
    )

    conflicting_findings = (
        _find_direction_conflicts(
            selected_profiles
        )
    )

    # -----------------------------------------------------
    # Ask Gemini to interpret evidence patterns
    # -----------------------------------------------------

    try:

        gaps = _interpret_gap_candidates(
            research_query=research_query,
            coverage_matrix=coverage_matrix,
            sparse_combinations=sparse_combinations,
            recurring_limitations=recurring_limitations,
            conflicting_findings=conflicting_findings,
            max_gaps=max_gaps,
        )

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as e:

        return json.dumps({
            "error": (
                "Could not interpret candidate gaps: "
                f"{str(e)}"
            )
        })

    except Exception as e:

        return json.dumps({
            "error": (
                "Gap interpretation model call failed: "
                f"{type(e).__name__}: "
                f"{str(e)[:200]}"
            )
        })

    # -----------------------------------------------------
    # Gap-type summary
    # -----------------------------------------------------

    gap_type_counts = Counter(
        gap["gap_type"]
        for gap in gaps
    )

    # -----------------------------------------------------
    # Save full result in session
    # -----------------------------------------------------

    session["gaps"] = gaps

    # -----------------------------------------------------
    # Compact model-facing result
    # -----------------------------------------------------

    return json.dumps({
        "profile_count": len(
            selected_profiles
        ),
        "gap_count": len(gaps),
        "gap_type_counts": dict(
            gap_type_counts
        ),
        "gaps": gaps,
        "disclaimer": (
            "These are candidate gaps inferred from abstracts "
            "and the retrieved sample. They are hypotheses for "
            "further literature validation, not proof that no "
            "prior research exists."
        ),
    })

# ---------------------------------------------------------
# Tool definitions shown to Gemini
# ---------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Search OpenAlex for recent journal articles with "
                "available abstracts relevant to a research topic "
                "or research question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The research topic or research question."
                        ),
                    },
                    "years": {
                        "type": "integer",
                        "description": (
                            "Number of recent calendar years to search. "
                            "Defaults to 3."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Number of papers to retrieve. "
                            "Must be between 10 and 25. "
                            "Defaults to 15."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "extract_paper_profiles",
            "description": (
                "Analyze abstracts of papers already stored in the "
                "current research session and extract structured "
                "method, setting, finding, and explicit limitation profiles. "
                "Call search_papers before using this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Optional list of OpenAlex paper IDs to analyze. "
                            "If omitted, analyze all papers currently stored "
                            "in the session."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_research_gaps",
            "description": (
                "Identify evidence-backed candidate research gaps "
                "from structured paper profiles already stored in "
                "the current research session. "
                "Call search_papers and extract_paper_profiles first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Optional list of paper IDs whose profiles "
                            "should be included. If omitted, use all "
                            "available profiles."
                        ),
                    },
                    "max_gaps": {
                        "type": "integer",
                        "description": (
                            "Maximum number of candidate gaps to return. "
                            "Must be between 1 and 5. Defaults to 5."
                        ),
                    },
                },
                "required": [],
            },
        },    
     },
]

# ---------------------------------------------------------
# Map tool names to Python functions
# ---------------------------------------------------------

TOOL_MAP = {
    "search_papers": search_papers,
    "extract_paper_profiles": extract_paper_profiles,
    "find_research_gaps": find_research_gaps,
}


# ---------------------------------------------------------
# Generic tool executor
# ---------------------------------------------------------

def run_tool(
    name: str,
    args: dict,
    session: dict | None = None,
) -> str:
    """
    Execute a registered tool.
    """

    if name not in TOOL_MAP:
        return json.dumps({
            "error": (
                f"Unknown tool '{name}'. "
                f"Available tools: {list(TOOL_MAP)}"
            )
        })

    try:
        if name in {
            "extract_paper_profiles",
            "find_research_gaps",
        }:

            if session is None:
                return json.dumps({
                    "error": (
                        f"{name} requires an active session."
                    )
                })

            return TOOL_MAP[name](
                session=session,
                **args,
            )

        return TOOL_MAP[name](**args)

    except TypeError as e:
        return json.dumps({
            "error": (
                f"Bad arguments for "
                f"'{name}': {str(e)}"
            )
        })