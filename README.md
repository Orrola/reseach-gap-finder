# Research Gap Finder

A tool-calling agent that helps users identify potantial research gaps from recent academic literature.

Given a research topic or question, the agent searches recent journal articles, extracts structured research profiles from their abstracts, and identifies underrepresented methodological or contextual combinations and conflicting findings.

The system is designed to avoid generating research gaps from the language model alone. Instead, it first builds structured evidence from retrieved papers and uses that evidence to support gap detection.

## How It Works

The agent follows a three-step workflow:

1. **Search recent literature**
   - Retrieves recent journal articles from OpenAlex.
   - By default, searches the current calendar year and the previous two years.
   - Only journal articles with available abstracts are included.
   - The default search returns 15 papers, with a supported range of 10–25.

2. **Extract structured paper profiles**
   - Processes abstracts in batches of five using Gemini.
   - Extracts standardized research characteristics:
     - paper id
     - method_catogory: research method
     - setting_category: research field
     - setting_detail: the detailed discription of research setting
     - main finding
     - finding direction: positive / negative / mixed / discriptive / unclear
     - explicitly stated limitation
   - Missing information is recorded as `not stated` rather than inferred.

3. **Find candidate research gaps**
   - Uses Python to analyze structured paper profiles.
   - Detects:
     - **Method Coverage Gap** — method × setting combinations that are absent or rare in the retrieved sample.
     - **Recurring limitations** — limitations explicitly repeated across multiple papers.
     - **Conflicting findings** — papers in similar settings that report different finding directions.
   - Gemini is used only to interpret patterns already identified from the structured evidence.

The final response presents a concise numbered list of candidate research gaps.

## Tools

### `search_papers`

Searches OpenAlex for recent journal articles relevant to the user's research topic.

**Inputs**
- `query`
- `years` — default: 3
- `max_results` — default: 15, range: 10–25

**Data retrieved**
- OpenAlex paper ID
- title
- publication year
- journal
- DOI
- citation count
- abstract

Full paper data is stored in backend session state. Only compact search information is returned to the main agent, reducing unnecessary model-token usage.

### `extract_paper_profiles`

Reads abstracts already stored in the current session and converts them into standardized research profiles.

Papers are processed in batches of five to reduce token usage.

Each profile contains:
- `method_category`
- `setting_category`
- `setting_detail`
- `main_finding`
- `finding_direction`
- `explicit_limitation`

The tool is instructed not to infer information that is not supported by the abstract.

### `find_research_gaps`

Analyzes the structured profiles produced by `extract_paper_profiles`.

Python first computes coverage patterns and finding distributions. Gemini then interprets only those structured patterns to generate candidate research gaps.

The tool requires at least eight successfully extracted paper profiles before attempting gap detection.

## Example Queries

The following queries can be used to test the deployed agent:

1. `Find research gaps in recent literature on generative AI and employee productivity.`

2. `What research gaps appear in recent journal articles about AI adoption in healthcare?`

3. `Identify candidate research gaps in recent research on large language models in education.`
