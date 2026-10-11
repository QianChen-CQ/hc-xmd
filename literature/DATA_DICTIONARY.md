# Data dictionary

## Source audit: `verified_sources.csv`

Each row is one canonical source target, not necessarily one unique work. The following fields are metadata, not measures of scientific correctness.

| Field | Meaning |
|---|---|
| `target_id` | Stable audit identity: DOI, arXiv identifier, or original primary URL. |
| `record_ids` | Linked search, exploratory or specialist record IDs, separated by ` | `. |
| `bibkeys` | Linked manuscript bibliography citation keys; blank if none. |
| `original_titles`, `original_urls` | JSON arrays of recorded input values, retained for traceability. |
| `verified_title` | Title read from the verified primary source or its corroborated publication. |
| `authors`, `editors` | Names as available from the primary metadata; role is kept separate. A blank field does not assert absence of a role. |
| `year` | Year of the verified publication/version, not necessarily first arXiv submission. Version and online/issue-year differences are explained in `notes` where relevant. |
| `venue` | Journal, proceedings, book, report series or repository, including available volume, issue and pages. |
| `publication_status` | `published`, `preprint`, `working_paper`, `report`, or `web_record`. This is separate from review status. |
| `document_type` | Form of the verified item; vocabulary below. |
| `review_status` | `venue_level_peer_review`, `not_established`, or `not_applicable`. Venue-level classification is not an independent audit of the individual manuscript's reviewing process. |
| `primary_url` | Readable primary publication, repository, institutional source or official contents page used to identify the item. |
| `additional_primary_urls` | Supplementary primary or publisher-deposited metadata URLs; blank when none was needed. |
| `verification_status` | Source-check outcome recorded by the audit; detailed meanings below. |
| `verification_scope` | Explicitly limits verification to bibliographic identity, metadata and publication form. |
| `verification_evidence_note` | Additional primary-evidence description retained where supplied by the source audit; blank is not a claim of absent evidence. |
| `initially_unverified_fields` | Fields explicitly unresolved in the initial source check, preserved even when supplementary publisher-deposited metadata later resolved them. |
| `unverified_fields` | Specifically identified unresolved metadata fields; blank means none was explicitly flagged, not that every possible metadata field was checked. |
| `notes` | Title/version corrections, source access limits, publication distinctions and other cautions. |
| `check_date` | Date of the source audit, ISO 8601. |

Document forms include `conference_paper`, `journal_article`, `journal_essay`, `conference_extended_abstract`, `one_page_conference_abstract`, `student_abstract`, `senior_member_presentation`, `senior_member_blue_sky_paper`, `senior_member_bridge_sky_paper`, `selected_conference_paper`, `newsletter_survey`, `authored_book`, `edited_reference_work`, `book_chapter`, `preprint`, `working_paper`, `technical_report`, `review_web_page`, and `whole_issue_or_contents`. An extended abstract is not automatically a full paper; the single-page EC records are distinguished from the longer STOC extended abstract. Senior-member agenda contributions are not automatically primary empirical studies.

`verified`, `verified_metadata`, and `verified_primary_metadata` indicate a primary metadata identity check. `verified_primary_indexed_metadata_with_corroboration` uses indexed primary metadata together with a corroborating primary record where the direct publisher extractor was blocked. `verified_with_metadata_correction` flags a corrected input description; `verified_with_version_link_mismatch` flags a specific version-link mismatch; `verified_with_source_discrepancy` preserves a disagreement between official metadata channels. `verified_nonpaper_search_hit` identifies a real page that is not itself a research paper. None means that scientific findings have been independently replicated.

## Linked verification tables

`record_verification.csv` has exactly one row per confirmatory raw hit. `record_id`, `historical_title`, `historical_url`, `historical_canonical_id`, `historical_decision` and `historical_reason` preserve its recorded identity and screening decision. `retained_in_confirmatory_map` is `true` only for the historical `retain_in_evidence_map` decision. Remaining columns repeat the linked source's identity and audit fields for convenient reading.

`bibliography_verification.csv` has one row per citation key. `coding_record_id` is the stable coding row ID; `citation_key` is the manuscript BibTeX-style key; `historical_title`, `historical_source_url` and `study_group_id` retain the input bibliography inventory. These records can cite different versions of the same study. The remaining columns link to the verified source.

`evidence_map_verification.csv` links each of the 79 merged-map rows. `record_id`, `branch`, `historical_title`, `historical_canonical_id` and `historical_decision` identify the earlier map row; the remaining fields give its source audit. It does not deduplicate the map into 74 families.

## Search and coding files

| Field | Meaning |
|---|---|
| `record_id` | Stable record identifier, such as R6_16; not a citation number. |
| `branch`, `query_id`, `position` | Retrieval branch, exact logged query identifier and displayed rank/position. No implied engine-wide total. |
| `title`, `url` | Historically recorded values in raw/archive files; normalized current values only in `included_coding.csv` and `merged_evidence_map.csv`. |
| `read_url`, `source_url` | Specific source used for reading or current source metadata; may differ from a search-result URL. |
| `access_date`, `accessed_date` | Date of the historical access attempt. |
| `canonical_id`, `study_group_id` | Historical deduplication/work-family identifier; distinct from the audit's source target ID. |
| `decision`, `reason` | Historical screening outcome and rationale. Inclusion, exclusion and duplicate/overlap decisions are not interchangeable. |
| `screening_level`, `read_level`, `access_level` | Abstract, metadata or full-text access actually recorded. Access does not imply line-by-line validation. |
| `access_status` | Historical access outcome, retained even if a later audit found another accessible primary source. |
| `evidence_type` | Kind of evidence used by the review: theory, computational, human study, conceptual, survey, etc. Separate from publication type. |
| `explanation_object` | Review coding for outcome justification, incentives, mechanism/rule inspection or combinations. |
| `family`, `mechanism_family` | Mechanism/application family used in reviewer synthesis. |
| `four_axis`, `axes` | Relevance tags. P: performance; I: incentive guarantees; S: structural inspection; B: participant-facing burden/behavior. Historical H corresponds to the last dimension. Tags do not assert quantitative measurement of every dimension. |
| `evidence_summary`, `supported_claim` | Limited claim used in the review, not a comprehensive article summary. |
| `limitations`, `limits`, `code_note` | Boundaries of the coded evidence; `code_note` is a qualitative coding note, not a software audit. |
| `citation_key` | Stable key used to link a bibliography entry to its source. |
| `collection`, `strand`, `novel_to_v63` | Historical provenance/grouping labels; not publication-quality rankings. |
| `locators` | Human-readable section, table or page evidence pointers where recorded. |
| `recorded_title`, `recorded_url`, `recorded_four_axis` | Original values retained next to normalized current map fields. |
| `publication_status`, `document_type`, `review_status`, `target_id` | Current source-audit fields joined into the two current evidence maps. |
| `metadata_checked_date`, `coding_revision_note` | Date and description of current normalization/coding clarification. |

Historical `rawhits.csv`, `candidates.csv`, `exploratory_starting_candidates.csv` and archived maps preserve their recorded values. `candidates.csv` contains the 58 records advanced to primary-access attempts. `included_coding.csv` contains the same 54 retained record IDs as its historical archive. `merged_evidence_map.csv` retains its 79 row identities; five historical cross-branch overlaps reduce the family count to 74. The historical `coding/reference_coding.csv` contains 68 bibliography entries and 66 study groups.

The JSON files preserve search protocols, exact queries, visible returned titles/URLs, access attempts and evidence notes. Their dates and decisions are historical. `summary.json` at the literature root describes the new source audit; `search/summary.json`, `search/merged_summary.json` and `coding/summary.json` describe their separate historical inventories. These counts must not be added together or treated as one systematic-review flow.
