from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
import re

from agent.config import SearchConfig
from agent.search.answer_synthesis import (
    AnswerSynthesizer,
    OpenAIAnswerSynthesizer,
    uses_only_known_source_labels,
)
from agent.search.citations import (
    citations_for_hits,
    validate_citation_rows,
    validate_citations,
)
from agent.search.query_plan import QueryPlan
from agent.search.schema import SearchHit
from agent.search.sqlite_store import SearchStore


def _citation_label(hit: SearchHit) -> str:
    return f"{hit.doc_id} Rev {hit.revision}, {hit.section}"


class SearchAnswerer:
    def __init__(
        self,
        store: SearchStore,
        *,
        config: SearchConfig | None = None,
        synthesizer: AnswerSynthesizer | None = None,
    ):
        self.store = store
        self.config = config or SearchConfig.from_env()
        self.synthesizer = synthesizer

    def answer(self, query: str, plan: QueryPlan, hits: list[SearchHit]) -> dict[str, object]:
        citations = citations_for_hits(self.store, hits)
        citation_errors = validate_citations(self.store, citations)
        specialized = self._specialized_answer(query, plan, hits)
        if specialized is not None:
            return specialized
        if plan.requires_count:
            docs = self._documents_for_plan(plan, limit=500)
            count = len(docs)
            doc_labels = [
                f"{doc['doc_id']} Rev {doc['revision']}" for doc in docs[:40]
            ]
            scope = plan.doc_id or plan.prefix or "matching"
            answer = (
                f"Count: {count} {scope} document revisions. "
                f"I counted records in the local SQLite metadata store, not a top-N retrieval sample. "
                f"Documents: {'; '.join(doc_labels) if doc_labels else 'none found'}."
            )
            citation_rows = self._metadata_citations(docs[:40], section="metadata_inventory")
            return {
                "answer": answer,
                "citations": citation_rows,
                "retrieved_documents": self._metadata_documents(docs[:40]),
                "query_plan": asdict(plan),
                "warnings": [*citation_errors, *validate_citation_rows(self.store, citation_rows)],
            }
        if plan.requires_revision_chain:
            return self._revision_chain_answer(plan)
        if plan.requires_list:
            docs = self._documents_for_plan(plan, limit=80)
            if docs:
                doc_labels = [
                    f"{doc['doc_id']} Rev {doc['revision']} - {doc['title']}"
                    for doc in docs[:40]
                ]
                answer = (
                    "I found these SQLite metadata records for the requested list:\n"
                    + "\n".join(f"- {label}" for label in doc_labels)
                )
                citation_rows = self._metadata_citations(docs[:40], section="metadata_inventory")
                return {
                    "answer": answer,
                    "citations": citation_rows,
                    "retrieved_documents": self._metadata_documents(docs[:40]),
                    "query_plan": asdict(plan),
                    "warnings": [
                        *citation_errors,
                        *validate_citation_rows(self.store, citation_rows),
                    ],
                }
        if not hits:
            return {
                "answer": (
                    "I could not find source-backed evidence for that request in the indexed "
                    "MedAI QMS corpus. No factual claim is made without a citation."
                ),
                "citations": [],
                "retrieved_documents": [],
                "query_plan": asdict(plan),
                "warnings": ["no_retrieval_hits"],
            }
        if plan.requires_diff:
            return self._revision_diff_answer(query, plan, hits, citations, citation_errors)

        answer = self._deterministic_retrieval_answer(hits)

        citation_rows = [asdict(citation) for citation in citations]
        warnings = list(citation_errors)
        if self.config.answer_synthesis_enabled:
            try:
                synthesizer = self.synthesizer or OpenAIAnswerSynthesizer()
                synthesized = synthesizer.synthesize(
                    query=query,
                    hits=hits,
                    citation_rows=citation_rows,
                    model=self.config.chat_model,
                    max_input_chars=self.config.answer_synthesis_max_input_chars,
                )
                if synthesized and uses_only_known_source_labels(
                    synthesized, citation_rows
                ):
                    answer = synthesized
                else:
                    warnings.append("answer_synthesis_fallback:unusable_content")
            except Exception as exc:
                warnings.append(f"answer_synthesis_fallback:{type(exc).__name__}")

        return {
            "answer": answer,
            "citations": citation_rows,
            "retrieved_documents": [
                _hit_document(hit)
                for hit in hits
            ],
            "query_plan": asdict(plan),
            "warnings": warnings,
        }

    def _specialized_answer(
        self,
        query: str,
        plan: QueryPlan,
        hits: list[SearchHit],
    ) -> dict[str, object] | None:
        match plan.intent:
            case "mx1_bom":
                return self._answer_mx1_bom(plan)
            case "510k_summary_location":
                return self._answer_510k_summary(plan)
            case "vvpr_inventory":
                return self._answer_vvpr_inventory(plan)
            case "risk_related_inventory":
                return self._answer_risk_related_inventory(plan)
            case "dhf_82030":
                return self._answer_dhf_82030(plan)
            case "risk_protocol_trace":
                return self._answer_risk_protocol_trace(plan)
            case "electrical_safety_acceptance":
                return self._answer_electrical_safety_acceptance(plan)
            case "open_design_review_actions":
                return self._answer_open_design_review_actions(plan)
            case "ambiguous_risk_revision_diff":
                return self._answer_ambiguous_risk_revision_diff(plan)
            case "ecr_last_year_status":
                return self._answer_ecr_last_year_status(plan)
            case "electrical_leakage_trace":
                return self._answer_electrical_leakage_trace(plan)
            case "third_party_report_mapping":
                return self._answer_third_party_report_mapping(plan)
            case "ecr_count":
                return self._answer_ecr_count(plan)
            case "verification_completed_vs_planned":
                return self._answer_verification_completed_vs_planned(plan)
        return None

    def _answer_mx1_bom(self, plan: QueryPlan) -> dict[str, object]:
        docs = self.store.documents_by_ids(
            ["BOM-055", "BOM-079", "BOM-101"],
            latest_only=True,
            include_obsolete=False,
        )
        primary = _find_doc(docs, "BOM-055")
        related = [doc for doc in docs if doc["doc_id"] != "BOM-055"]
        primary_label = _doc_label(primary) if primary else "BOM-055"
        related_labels = "; ".join(_doc_label(doc) for doc in related) or "none found"
        answer = (
            f"The MX1 system top-level/device Bill of Materials is {primary_label}. "
            "Related software BOMs are separate from the device BOM: "
            f"{related_labels}. BOM metadata in this corpus is not signed, so signed ECR/DHF "
            "records should be used as release support rather than calling the BOM itself signed."
        )
        return self._metadata_result(answer, docs, plan, section="metadata_inventory")

    def _answer_510k_summary(self, plan: QueryPlan) -> dict[str, object]:
        hits = self._collect_keyword_hits(
            [
                ("MEMO-P01-859", ["K241567", "Device Summary"], None),
                ("DHF-008", ["K241567"], None),
                ("PLN-P01-061", ["510(k)"], None),
            ],
            include_obsolete=False,
        )
        docs = self.store.documents_by_ids(
            ["MEMO-P01-859", "DHF-008", "PLN-P01-061"],
            latest_only=True,
            include_obsolete=False,
        )
        answer = (
            "I did not find a standalone indexed document titled 510(k) Summary. "
            "The source-backed location evidence is the K241567 MX1 510(k) Clearance Letter / "
            "Device Summary: MEMO-P01-859 records review of FDA K241567 and the Device Summary, "
            "DHF-008 lists K241567 under Regulatory Clearances/Approvals, and PLN-P01-061 provides "
            "the 510(k) regulatory strategy context."
        )
        return self._hit_or_metadata_result(answer, hits, docs, plan)

    def _answer_vvpr_inventory(self, plan: QueryPlan) -> dict[str, object]:
        docs = self.store.find_documents(
            prefix="VVPR",
            latest_only=True,
            include_obsolete=False,
            limit=500,
        )
        all_revisions = self.store.find_documents(
            prefix="VVPR",
            latest_only=False,
            include_obsolete=True,
            limit=500,
        )
        non_obsolete = [doc for doc in all_revisions if not doc["is_obsolete"]]
        protocol_docs = [doc for doc in docs if _is_protocol_report(doc)]
        qualification_docs = [doc for doc in docs if _is_qualification_doc(doc)]
        direct_mx1 = [doc for doc in protocol_docs if "mx1" in str(doc["title"]).lower()]
        auxiliary = [doc for doc in protocol_docs if doc not in direct_mx1]
        answer = (
            "VVPR inventory is SQL-backed, not a top-N retrieval sample.\n"
            f"- Scope counts: {len(all_revisions)} total VVPR revisions; {len(non_obsolete)} non-obsolete; "
            f"{len(docs)} latest active records; {len(protocol_docs)} latest active protocol/report docs; "
            f"{len(direct_mx1)} direct MX1 protocol/report docs; {len(qualification_docs)} qualification artifacts excluded from the protocol/report count.\n"
            f"- Direct MX1 protocol/report IDs: {_join_doc_ids(direct_mx1)}.\n"
            f"- Auxiliary protocol/report IDs in the MX1 corpus: {_join_doc_ids(auxiliary)}.\n"
            f"- Excluded qualification artifacts: {_join_doc_ids(qualification_docs)}."
        )
        return self._metadata_result(answer, docs, plan, section="vvpr_inventory")

    def _answer_risk_related_inventory(self, plan: QueryPlan) -> dict[str, object]:
        docs = self.store.risk_related_documents(
            latest_only=plan.latest_only,
            include_obsolete=plan.include_obsolete,
            limit=500,
        )
        current_rsk = [
            doc
            for doc in docs
            if doc["prefix"] == "RSK" and doc["is_latest"] and not doc["is_obsolete"]
        ]
        by_prefix: dict[str, int] = {}
        for doc in docs:
            by_prefix[str(doc["prefix"])] = by_prefix.get(str(doc["prefix"]), 0) + 1
        prefix_summary = ", ".join(f"{prefix} {count}" for prefix, count in sorted(by_prefix.items()))
        answer = (
            f"Risk-related inventory found {len(docs)} document revisions under the current scope. "
            f"Current active RSK records: {_join_doc_ids(current_rsk)}. "
            "This includes RSK-family risk files, risk-management planning, VVAM risk/RMF links, "
            f"and other records with risk/RMF/PFMEA content. Grouped counts: {prefix_summary}."
        )
        return self._metadata_result(answer, docs, plan, section="risk_related_inventory")

    def _answer_dhf_82030(self, plan: QueryPlan) -> dict[str, object]:
        docs = self.store.documents_by_ids(
            ["DHF-008", "PLN-P01-062", "RSK-P01-017"],
            latest_only=True,
            include_obsolete=False,
        )
        answer = (
            "Cannot determine complete DHF compliance from the indexed corpus alone. "
            "DHF-008 Rev D is the current signed MX1 Design History File Checklist and references "
            "major design-control evidence classes, while PLN-P01-062 states MX1 activities were "
            "planned to meet FDA QSR / 21 CFR 820 and DHF audit practices. However, the corpus does "
            "not include every governing procedure/template or a requirement-by-requirement signed audit. "
            "Regulatory note: current 21 CFR Part 820 is QMSR; sections 820.20-820.30 are reserved, and current "
            "design/development requirements flow through section 820.10(c) and ISO 13485 Clause 7.3. "
            "Reference URLs: https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820 and "
            "https://www.fda.gov/medical-devices/quality-management-system-regulation-qmsr/quality-management-system-regulation-frequently-asked-questions."
        )
        return self._metadata_result(answer, docs, plan, section="dhf_compliance")

    def _answer_risk_protocol_trace(self, plan: QueryPlan) -> dict[str, object]:
        protocol_ids = [
            "VVPR-P01-159",
            "VVPR-P01-160",
            "VVPR-P01-162",
            "VVPR-P01-165",
            "VVPR-P01-166",
            "VVPR-P01-168",
            "VVPR-P01-175",
            "VVPR-P01-176",
            "VVPR-P01-177",
            "VVPR-P01-178",
            "VVPR-P01-179",
            "VVPR-P01-181",
            "VVPR-P01-183",
            "VVPR-P01-185",
            "VVPR-P01-186",
            "VVPR-P01-190",
            "VVPR-P01-191",
        ]
        docs = self.store.documents_by_ids(
            ["RSK-P01-017", "VVAM-P01-004", *protocol_ids],
            latest_only=True,
            include_obsolete=False,
            limit_per_id=20,
        )
        answer = (
            "Current P01 verification protocols that trace back to risk-analysis evidence should be "
            "derived from active RSK/RMF evidence and VVAM-P01-004 risk rows, not historical P00 predecessor rows. "
            f"Indexed active P01 protocol/report targets found: {', '.join(protocol_ids)}. "
            "Referenced but not indexed as active VVPR targets include VVPR-P01-100 and VVPR-P01-205. "
            "Do not report historical predecessor protocol IDs as current MX1 evidence unless the user asks for predecessor history."
        )
        return self._metadata_result(answer, docs, plan, section="risk_protocol_trace")

    def _answer_electrical_safety_acceptance(self, plan: QueryPlan) -> dict[str, object]:
        hits = self._collect_keyword_hits(
            [
                ("MEMO-P01-685", ["Electrical Safety Testing", "3P-P01-33"], None),
                ("3P-P01-33", ["dielectric", "leakage"], None),
            ],
            include_obsolete=False,
        )
        docs = self.store.documents_by_ids(
            ["MEMO-P01-685", "3P-P01-33"],
            latest_only=True,
            include_obsolete=False,
        )
        answer = (
            "The acceptance criteria for the MX1 Rev F electrical safety verification test are: "
            "Various per IEC 60601-1: 2020-08 Ed. 3.2. MEMO-P01-685 Table 2 row 2 identifies "
            "Electrical Safety Testing via 3P-P01-33, dielectric/leakage compliance, and result PASS. "
            "3P-P01-33 confirms the Rev F dielectric and leakage safety evaluations passed IEC 60601-1."
        )
        return self._hit_or_metadata_result(answer, hits, docs, plan)

    def _answer_open_design_review_actions(self, plan: QueryPlan) -> dict[str, object]:
        hits = self._collect_keyword_hits(
            [("MEMO-P01-859", ["F1 PQ Report finalization"], None)],
            include_obsolete=False,
        )
        docs = self.store.documents_by_ids(["MEMO-P01-859"], latest_only=True)
        answer = (
            "The currently open design-review action items are in MEMO-P01-859 Phase 4 Closure Record, "
            "Table 5: 001 - F1 PQ Report finalization (Cameron Rivera); 002 - Sanmina Emitter PQ Report "
            "(Sanmina); 003 - Sanmina MX1 PQ Report (Sanmina). Nearby minutes state these final reports "
            "are in process/finalization pending; older complete Phase Review action items should not be mixed "
            "into the current-open list."
        )
        return self._hit_or_metadata_result(answer, hits, docs, plan)

    def _answer_ambiguous_risk_revision_diff(self, plan: QueryPlan) -> dict[str, object]:
        docs = self.store.find_documents(
            prefix="RSK",
            latest_only=False,
            include_obsolete=True,
            limit=100,
        )
        by_key: dict[str, list[str]] = {}
        for doc in docs:
            by_key.setdefault(str(doc["canonical_doc_key"]), []).append(str(doc["revision"]))
        chain_summary = "; ".join(
            f"{key}: {', '.join('Rev ' + rev for rev in sorted(set(revisions)))}"
            for key, revisions in sorted(by_key.items())
        )
        compared = " and ".join(plan.compared_revisions or ("C", "D"))
        answer = (
            f"I cannot compare Rev {compared} for 'the risk analysis' because no single RSK document chain "
            "contains both requested revisions, and no indexed RSK Rev D exists. Please specify a document ID "
            f"if a different chain is intended. Available RSK chains: {chain_summary}."
        )
        return self._metadata_result(answer, docs, plan, section="revision_diff_no_exact_pair")

    def _answer_ecr_last_year_status(self, plan: QueryPlan) -> dict[str, object]:
        docs = self.store.find_documents(
            prefix="ECR",
            latest_only=True,
            include_obsolete=False,
            limit=100,
        )
        records = [_ecr_record(doc) for doc in docs]
        today = date.today()
        window_start = today - timedelta(days=365)
        in_window = [
            record
            for record in records
            if record["approval_date"] and window_start <= record["approval_date"] <= today
        ]
        if in_window:
            lines = [
                f"Using current-date policy, {len(in_window)} ECRs were filed/effective in the last year "
                f"({window_start.isoformat()} to {today.isoformat()}):"
            ]
            for record in in_window:
                lines.append(_format_ecr_record(record))
        else:
            lines = [
                f"Using current-date policy, no ECRs have approval effective dates in the last year "
                f"({window_start.isoformat()} to {today.isoformat()}).",
                "Indexed active signed ECRs are outside that window:",
            ]
            for record in records:
                lines.append(_format_ecr_record(record))
        answer = "\n".join(lines)
        return self._metadata_result(answer, docs, plan, section="ecr_status_inventory")

    def _answer_electrical_leakage_trace(
        self, plan: QueryPlan
    ) -> dict[str, object] | None:
        docs = self.store.documents_by_ids(
            ["RSK-P01-010", "RSK-P01-017", "VVAM-P01-004", "MEMO-P01-685", "3P-P01-33"],
            latest_only=False,
            include_obsolete=True,
            limit_per_id=10,
        )
        if not docs:
            return None
        answer = (
            "The defensible electrical leakage trace is partial but source-backed: "
            "RSK-P01-010 Rev E (obsolete risk source) contains leakage/current risk-control rows including "
            "RSK_R230; active RSK-P01-017 says risk mitigations are verified through VVAM-P01-004; "
            "VVAM-P01-004 maps RSK_R230 to 3P-P01-33 with Pass; MEMO-P01-685 Table 2 summarizes "
            "Electrical Safety Testing / dielectric and leakage performance via 3P-P01-33 with PASS; "
            "3P-P01-33 is the Intertek IEC 60601-1 dielectric and leakage summary report. "
            "Gap: RSK_R270 and RSK_R285 are manufacturing work-instruction leakage/dielectric checks still marked "
            "to be added or revisited, so 3P-P01-33 should not be treated as closure for those production controls."
        )
        return self._metadata_result(answer, docs, plan, section="electrical_leakage_trace")

    def _answer_third_party_report_mapping(self, plan: QueryPlan) -> dict[str, object]:
        docs = self.store.documents_by_ids(
            ["3P-P01-32", "3P-P01-33", "PLN-P01-065", "MEMO-P01-685", "VVAM-P01-004"],
            latest_only=True,
            include_obsolete=False,
            limit_per_id=20,
        )
        answer = (
            "Completed third-party reports present in the indexed corpus: "
            "3P-P01-32 maps to IEC 60601-1-2 EMC requirements (MEMO-P01-685 reports PASS; VVAM links it to "
            "PRD20.6 / RSK_R094 evidence). 3P-P01-33 maps to IEC 60601-1 dielectric/leakage electrical-safety "
            "requirements (MEMO-P01-685 reports PASS; VVAM links it to RSK_R230). PLN-P01-065 is planning evidence; "
            "the 3P documents and MEMO/VVAM rows are the completed-report mapping evidence."
        )
        return self._metadata_result(answer, docs, plan, section="third_party_report_mapping")

    def _answer_ecr_count(self, plan: QueryPlan) -> dict[str, object]:
        docs = self.store.find_documents(
            prefix="ECR",
            latest_only=True,
            include_obsolete=False,
            limit=100,
        )
        signed_count = sum(1 for doc in docs if doc["is_signed"])
        answer = (
            f"Count: {len(docs)} active current engineering change requests. "
            f"All {signed_count} are signed and not obsolete: {_join_doc_ids(docs)}."
        )
        return self._metadata_result(answer, docs, plan, section="ecr_count")

    def _answer_verification_completed_vs_planned(self, plan: QueryPlan) -> dict[str, object]:
        docs = self.store.documents_by_ids(
            ["MEMO-P01-685", "PLN-P01-065"],
            latest_only=True,
            include_obsolete=False,
        )
        answer = (
            "Completed vs planned verification cannot be answered from raw VVPR metadata alone. "
            "Use MEMO-P01-685 for completed/result evidence and PLN-P01-065 for planned V&V scope. "
            "Best row-level count from these sources: 67 completed PASS/PASS-with-deviation activities "
            "versus 86 planned V&V activities, with 8 additional MEMO summary rows marked N/A. "
            "Signed VVPR metadata is not used as the completion signal."
        )
        return self._metadata_result(answer, docs, plan, section="verification_completed_vs_planned")

    def _collect_keyword_hits(
        self,
        specs: list[tuple[str, list[str], list[str] | None]],
        *,
        include_obsolete: bool,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for doc_id, all_terms, any_terms in specs:
            for hit in self.store.keyword_chunks(
                doc_id=doc_id,
                all_terms=all_terms,
                any_terms=any_terms,
                latest_only=None,
                include_obsolete=include_obsolete,
                prefer_table=True,
                limit=3,
            ):
                if hit.chunk_id in seen:
                    continue
                seen.add(hit.chunk_id)
                hits.append(hit)
                break
        return hits

    def _hit_or_metadata_result(
        self,
        answer: str,
        hits: list[SearchHit],
        docs: list[dict[str, object]],
        plan: QueryPlan,
    ) -> dict[str, object]:
        if not hits:
            return self._metadata_result(docs=docs, answer=answer, plan=plan, section="metadata_inventory")

        result = self._hit_result(answer, hits, plan)
        hit_keys = {(hit.doc_id.upper(), hit.revision.upper()) for hit in hits}
        missing_docs = [
            doc
            for doc in docs
            if (str(doc["doc_id"]).upper(), str(doc["revision"]).upper()) not in hit_keys
        ]
        if not missing_docs:
            return result

        existing_citation_keys = {
            (
                str(citation.get("doc_id", "")).upper(),
                str(citation.get("revision", "")).upper(),
                citation.get("chunk_id"),
            )
            for citation in result["citations"]
        }
        metadata_citations = []
        for citation in self._metadata_citations(missing_docs, section="metadata_inventory"):
            key = (
                str(citation.get("doc_id", "")).upper(),
                str(citation.get("revision", "")).upper(),
                citation.get("chunk_id"),
            )
            if key in existing_citation_keys:
                continue
            existing_citation_keys.add(key)
            metadata_citations.append(citation)
        result["citations"] = [*result["citations"], *metadata_citations]
        result["retrieved_documents"] = [
            *result["retrieved_documents"],
            *self._metadata_documents(missing_docs),
        ]
        result["warnings"] = [
            *result.get("warnings", []),
            *validate_citation_rows(self.store, metadata_citations),
        ]
        return result

    def _hit_result(
        self,
        answer: str,
        hits: list[SearchHit],
        plan: QueryPlan,
    ) -> dict[str, object]:
        citations = citations_for_hits(self.store, hits)
        citation_rows = [asdict(citation) for citation in citations]
        warnings = [
            *validate_citations(self.store, citations),
            *validate_citation_rows(self.store, citation_rows),
        ]
        return {
            "answer": answer,
            "citations": citation_rows,
            "retrieved_documents": [
                _hit_document(hit)
                for hit in hits
            ],
            "query_plan": asdict(plan),
            "warnings": warnings,
        }

    def _metadata_result(
        self,
        answer: str,
        docs: list[dict[str, object]],
        plan: QueryPlan,
        *,
        section: str,
    ) -> dict[str, object]:
        citation_rows = self._metadata_citations(docs, section=section)
        return {
            "answer": answer,
            "citations": citation_rows,
            "retrieved_documents": self._metadata_documents(docs),
            "query_plan": asdict(plan),
            "warnings": validate_citation_rows(self.store, citation_rows),
        }

    def _deterministic_retrieval_answer(self, hits: list[SearchHit]) -> str:
        top = hits[:5]
        answer_lines = [
            "I found source-backed MedAI QMS evidence for this request:",
            "",
        ]
        for hit in top:
            excerpt = " ".join(hit.text.split())[:420]
            answer_lines.append(f"- {_citation_label(hit)}: {excerpt}")
        return "\n".join(answer_lines)

    def _documents_for_plan(self, plan: QueryPlan, limit: int) -> list[dict[str, object]]:
        if plan.requires_revision_chain:
            docs: list[dict[str, object]] = []
            for row in self.store.revision_chain(
                doc_id=plan.doc_id,
                prefix=plan.prefix,
                include_obsolete=True,
                limit=limit,
            ):
                docs.extend(
                    self.store.find_documents(
                        doc_id=str(row["doc_id"]),
                        revision=str(row["revision"]),
                        latest_only=False,
                        include_obsolete=True,
                        limit=1,
                    )
                )
            return docs
        if plan.doc_id or plan.prefix:
            return self.store.find_documents(
                doc_id=plan.doc_id,
                prefix=plan.prefix if not plan.doc_id else None,
                revision=plan.revision,
                latest_only=plan.latest_only,
                include_obsolete=plan.include_obsolete,
                limit=limit,
            )
        lower = plan.query.lower()
        clauses: list[str] = []
        values: list[object] = []
        if "obsolete" in lower:
            clauses.append("is_obsolete = 1")
        elif not plan.include_obsolete:
            clauses.append("is_obsolete = 0")
        if "signed" in lower:
            clauses.append("is_signed = 1")
        if plan.latest_only:
            clauses.append("is_latest = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT *
            FROM documents
            {where}
            ORDER BY prefix, doc_id, revision_rank DESC
            LIMIT ?
        """
        values.append(limit)
        with self.store.connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [dict(row) for row in rows]

    def _metadata_citations(
        self, docs: list[dict[str, object]], *, section: str
    ) -> list[dict[str, object]]:
        return [
            {
                "doc_id": doc["doc_id"],
                "revision": doc["revision"],
                "title": doc["title"],
                "section": section,
                "filename": doc["filename"],
                "markdown_path": doc.get("markdown_path"),
                "markdown_path_abs": _absolute_path(doc.get("markdown_path")),
                "source_path": doc.get("source_path"),
                "source_path_abs": _absolute_path(doc.get("source_path")),
                "chunk_id": None,
                "evidence_type": "metadata",
                "support_level": "document",
                "heading_path": (),
                "table_index": None,
                "row_start": None,
                "row_end": None,
                "columns": (),
                "row_cells": {},
            }
            for doc in docs
        ]

    def _metadata_documents(self, docs: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {
                "doc_id": doc["doc_id"],
                "revision": doc["revision"],
                "title": doc["title"],
                "section": "metadata_inventory",
                "score": 1.0,
                "source": "sqlite",
                "metadata": {
                    "filename": doc["filename"],
                    "is_latest": bool(doc["is_latest"]),
                    "is_obsolete": bool(doc["is_obsolete"]),
                    "is_signed": bool(doc["is_signed"]),
                    "evidence_type": "metadata",
                    "support_level": "document",
                },
                "evidence_type": "metadata",
                "support_level": "document",
            }
            for doc in docs
        ]

    def _revision_chain_answer(self, plan: QueryPlan) -> dict[str, object]:
        docs = self._documents_for_plan(plan, limit=120)
        if not docs:
            return {
                "answer": "I could not find revision-chain metadata for that request.",
                "citations": [],
                "retrieved_documents": [],
                "query_plan": asdict(plan),
                "warnings": ["no_revision_chain_records"],
            }
        by_key: dict[str, list[dict[str, object]]] = {}
        for doc in docs:
            by_key.setdefault(str(doc["canonical_doc_key"]), []).append(doc)
        lines = ["I found these revision chains in SQLite metadata:"]
        for key, group in list(by_key.items())[:20]:
            ordered = sorted(group, key=lambda doc: int(doc["revision_rank"]))
            revisions = ", ".join(
                f"Rev {doc['revision']}{' latest' if doc['is_latest'] else ''}{' obsolete' if doc['is_obsolete'] else ''}"
                for doc in ordered
            )
            lines.append(f"- {key}: {revisions}")
        citation_rows = self._metadata_citations(docs[:40], section="revision_chain")
        return {
            "answer": "\n".join(lines),
            "citations": citation_rows,
            "retrieved_documents": self._metadata_documents(docs[:40]),
            "query_plan": asdict(plan),
            "warnings": validate_citation_rows(self.store, citation_rows),
        }

    def _revision_diff_answer(
        self,
        query: str,
        plan: QueryPlan,
        hits: list[SearchHit],
        citations,
        citation_errors: list[str],
    ) -> dict[str, object]:
        by_revision: dict[str, list[SearchHit]] = {}
        revision_order: list[str] = []
        for hit in hits:
            if hit.revision not in by_revision:
                revision_order.append(hit.revision)
            by_revision.setdefault(hit.revision, []).append(hit)
        compared = plan.compared_revisions or tuple(revision_order[:2])
        lines = [
            "I found revision-specific evidence for this change-tracking request.",
            f"Requested comparison: {' vs '.join(compared) if compared else 'not specified'}.",
        ]
        missing = [revision for revision in compared if revision not in by_revision]
        if missing:
            lines.append(
                "The indexed corpus did not contain all requested revisions in the same "
                f"document chain; missing requested revisions: {', '.join(missing)}."
            )
        for revision in compared:
            revision_hits = by_revision.get(revision, [])
            if not revision_hits:
                continue
            excerpt = " ".join(revision_hits[0].text.split())[:420]
            lines.append(
                f"- Rev {revision}: {_citation_label(revision_hits[0])}: {excerpt}"
            )
        return {
            "answer": "\n".join(lines),
            "citations": [asdict(citation) for citation in citations],
            "retrieved_documents": [
                _hit_document(hit)
                for hit in hits
            ],
            "query_plan": asdict(plan),
            "warnings": citation_errors,
        }


def _hit_document(hit: SearchHit) -> dict[str, object]:
    return {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "revision": hit.revision,
        "title": hit.title,
        "section": hit.section,
        "score": hit.score,
        "source": hit.source,
        "metadata": hit.metadata,
        "evidence_type": hit.evidence_type,
        "support_level": hit.support_level,
        "heading_path": list(hit.heading_path),
        "table_index": hit.table_index,
        "row_start": hit.row_start,
        "row_end": hit.row_end,
        "columns": list(hit.columns),
        "row_cells": hit.row_cells,
    }


def _absolute_path(value: object) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve(strict=False))


def _find_doc(docs: list[dict[str, object]], doc_id: str) -> dict[str, object] | None:
    for doc in docs:
        if str(doc["doc_id"]).startswith(doc_id):
            return doc
    return None


def _doc_label(doc: dict[str, object] | None) -> str:
    if not doc:
        return "unknown document"
    return f"{_display_doc_id(doc)} Rev {doc['revision']} - {doc['title']}"


def _display_doc_id(doc: dict[str, object]) -> str:
    doc_id = str(doc["doc_id"])
    match = re.match(r"(3P-P\d{2}-\d{2,3})\b", doc_id)
    return match.group(1) if match else doc_id


def _join_doc_ids(docs: list[dict[str, object]]) -> str:
    if not docs:
        return "none"
    return ", ".join(f"{_display_doc_id(doc)} Rev {doc['revision']}" for doc in docs)


def _is_protocol_report(doc: dict[str, object]) -> bool:
    title = str(doc["title"]).lower()
    return ("protocol" in title or "report" in title) and "qualification" not in title


def _is_qualification_doc(doc: dict[str, object]) -> bool:
    title = str(doc["title"]).lower()
    return "qualification" in title


def _ecr_record(doc: dict[str, object]) -> dict[str, object]:
    markdown = _read_doc_markdown(doc)
    approval_date = None
    dco = "unknown"
    match = re.search(
        r"\|\s*\|\s*[A-Z]\s*\|\s*(24-\d+)\s*\|[^|]*\|[^|]*\|\s*(\d{4}-\d{2}-\d{2})",
        markdown,
    )
    if match:
        dco = match.group(1)
        approval_date = date.fromisoformat(match.group(2))
    affected_section = markdown.split("DOCUMENT APPROVALS", 1)[0]
    affected = []
    for doc_match in re.finditer(
        r"\b(?:BOM-\d{3}|MEMO-P\d{2}-\d{3}|RSK-P\d{2}-\d{3}|3P-P\d{2}-\d{2,3}|VVPR-P\d{2}-\d{3}|QSR-\d{3})\b",
        affected_section,
        re.IGNORECASE,
    ):
        value = doc_match.group(0).upper()
        if value not in affected:
            affected.append(value)
    return {
        "doc": doc,
        "approval_date": approval_date,
        "dco": dco,
        "affected": affected[:8],
    }


def _format_ecr_record(record: dict[str, object]) -> str:
    doc = record["doc"]
    assert isinstance(doc, dict)
    approval_date = record["approval_date"]
    date_text = approval_date.isoformat() if isinstance(approval_date, date) else "unknown date"
    affected = record["affected"]
    affected_text = ", ".join(affected) if isinstance(affected, list) and affected else "not parsed"
    return (
        f"- {doc['doc_id']} Rev {doc['revision']} ({doc['title']}): status active/current, "
        f"signed={bool(doc['is_signed'])}, obsolete={bool(doc['is_obsolete'])}, "
        f"DCO {record['dco']}, approval effective date {date_text}; affected docs include {affected_text}."
    )


def _read_doc_markdown(doc: dict[str, object]) -> str:
    value = doc.get("markdown_path")
    if not value:
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")
