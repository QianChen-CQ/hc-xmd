# HC-XMD literature companion metadata

These files support the structured narrative review and its source audit. They contain bibliographic and coding metadata, not publisher article PDFs or independent replications of published results.

Prepared on **11 October 2026** for upload to <https://github.com/QianChen-CQ/hc-xmd>. The preparation audit does **not** establish that this new folder was already online. The repository owner must upload the complete `literature/` folder; the relative links below work once it is uploaded at the repository root.

## Which files to use

- [verified_sources.csv](verified_sources.csv): authoritative source identity and publication-form audit, with 182 canonical URL/DOI targets, primary URLs, authors, dates, and source-specific limitations. **182 is not a count of distinct papers.** Multiple targets can be versions or pages of the same work.
- [search/included_coding.csv](search/included_coding.csv): current, normalized 54-row confirmatory evidence map. Titles, source URLs, document types and necessary coding clarifications are updated; historical retention decisions and row identities are unchanged. `recorded_title` and `recorded_url` retain the earlier values. The complete earlier coding is archived in [search/historical/included_coding_recorded.csv](search/historical/included_coding_recorded.csv).
- [search/rawhits.csv](search/rawhits.csv): all 115 historically observed confirmatory search records, with their original titles and screening decisions. An included raw hit, excluded webpage, or duplicate is not automatically a distinct research paper. A historical title can be inaccurate: use the linked audit for the corrected source identity.
- [record_verification.csv](record_verification.csv): one audit link for each of the 115 raw records; includes the historical decision and current verified identity.
- [search/merged_evidence_map.csv](search/merged_evidence_map.csv) and [evidence_map_verification.csv](evidence_map_verification.csv): 79 rows from the confirmatory, exploratory and specialist branches, representing 74 historical evidence families after five cross-branch overlaps. The earlier map is archived in `search/historical/merged_evidence_map_recorded.csv`.
- [coding/reference_coding.csv](coding/reference_coding.csv): the historical 68-entry manuscript bibliography inventory (66 historical study groups). [bibliography_verification.csv](bibliography_verification.csv) maps every citation key to its verified source record. Coding row IDs are not manuscript citation numbers.
- [summary.json](summary.json) and [DATA_DICTIONARY.md](DATA_DICTIONARY.md): coverage checks, count definitions and schemas.

The remaining `search/` CSV and JSON files retain the protocol, visible search results, screening/access logs and specialist evidence notes. The historical exploratory logs may contain other leads outside the defined 182-target audit; their presence is not a claim that every exploratory lead was independently verified. Internal session citation locators have been removed from the public copies. Actual URLs, historical decisions, access dates and substantive coding are preserved, except for the explicitly identified current maps.

## Publication and evidence distinctions

`publication_status`, `document_type` and `review_status` answer different questions. A real preprint is not a fabricated reference, but it is not thereby a peer-reviewed journal article. A formally published journal essay, selected proceedings contribution or scholarly newsletter survey remains a published item even when its individual reviewing process is not established. Conference student abstracts and one-page extended abstracts remain explicitly labeled; they are not full technical papers. Whole-issue tables of contents and review webpages are webpage records, not research papers.

The verified record may describe a formally published counterpart of an original arXiv URL when primary evidence established the match. Where the audit instead verifies a particular preprint version, its preprint status and version are retained. Read `notes` before substituting one version for another. The v3/v4 description-paper link discrepancy and the two official BundleFlow pagination schemes are recorded rather than silently discarded.

Evidence tags describe relevance, not a guarantee that each axis was measured. The current maps use P (performance), I (incentive guarantees), S (structural inspection), and B (participant-facing burden or behavioral evidence). Historical search files used H for the last dimension; archived values are retained. Formal simplicity, explanation length and statistical guarantees do not by themselves demonstrate human understanding or strategic behavior.

## Audit scope and historical counts

The audit covers **115/115 confirmatory raw records, 79/79 merged-map rows and 68/68 bibliography entries**, linked through 182 source targets. Titles, authors, venue/year and publication form were checked against publisher, conference, arXiv, institutional or publisher-deposited metadata. Access and version limitations remain in each record. This is a bibliographic identity and publication-form audit, not a claim of line-by-line review or independent validation of every result.

The historical confirmatory screen retained 54 evidence families after screening 115 visible records. These are observed returned records, not total search-engine hit estimates or a systematic-review denominator. Later access success and metadata corrections do not retrospectively change the historical screen. The primary-source URLs are supplied so readers can check the records directly.

Files are UTF-8. Lists within cells use ` | ` unless identified as JSON arrays in the dictionary. SHA-256 checksums in `SHA256SUMS` identify the prepared companion files; the checksum file does not include itself.
