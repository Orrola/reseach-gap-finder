# Research Gap Finder

A tool-calling agent that helps users identify potantial research gaps from recent academic literature.

Given a research topic or question, the agent searches recent journal articles, extracts structured research profiles from their abstracts, and identifies uncovered methodological or contextual combinations and conflicting findings.

The system is designed to avoid generating research gaps from the language model alone. Instead, it first builds structured evidence from retrieved papers and uses that evidence to support gap detection.

## Tools
- `search_papers`: Searches OpenAlex for journal articles from the last 3 calendar years (default 15 papers, range 10–25). Full records are kept in backend session state; the model only sees a compact summary.
- `extract_paper_profiles`: Extracts structured profiles from the stored abstracts in batches of five, failed papers are retried once and reported.
- `find_research_gaps`: Finds candidate gaps from the profiles. Requires at least 8 successful profiles.

   _Demo-scale sample size:_
     The default sample size is intentionally limited to 15 papers to keep retrieval, profile extraction, and gap analysis fast and cost-efficient for an interactive demonstration. In a production or research setting, the workflow could be expanded to approximately 50–200 papers using staged or batched processing.

## How It Works

Given a research topic, the agent runs three steps automatically:

1. **Search:** retrieves recent journal articles with abstracts from OpenAlex.
2. **Profile:** Gemini turns each abstract into a structured profile (method, setting, finding direction, explicit limitation). Missing information is recorded as `not stated`, never inferred.
3. **Detect:**
   - **Method coverage gap:** a method × setting combination that is absent or rare in the sample
   - **Recurring limitation:** a limitation explicitly stated by several papers
   - **Conflicting findings:** papers in a similar setting that report different finding directions

The final answer lists each candidate gap with its supporting papers. Follow-up questions are answered from the session without re-running the tools.

The following queries can be used to test the deployed agent:

1. `Find research gaps in recent literature on generative AI and employee productivity.`

2. `What research gaps appear in recent journal articles about AI adoption in healthcare?`

3. `Identify candidate research gaps in recent research on large language models in education.`

The agent also supports follow-up questions within the same session, such as *“What’s the evidence behind your finding?”*, allowing users to further explore the reasoning and evidence behind identified candidate gaps.
