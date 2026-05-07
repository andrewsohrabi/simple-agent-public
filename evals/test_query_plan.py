from agent.search.query_plan import plan_query


def test_query_plan_routes_enumeration_to_sql_count():
    plan = plan_query("How many engineering change requests are in the system?")
    assert plan.category == "enumeration"
    assert plan.strategy == "sql_count"
    assert plan.prefix == "ECR"
    assert plan.requires_count is True


def test_query_plan_extracts_revision_diff():
    plan = plan_query("What changed between Rev C and Rev D of the risk analysis?")
    assert plan.category == "revision_diff"
    assert plan.requires_diff is True
    assert plan.include_obsolete is True
    assert plan.compared_revisions == ("C", "D")


def test_query_plan_prefers_ecr_when_query_mentions_bom_change_record():
    plan = plan_query("Find the ECR for BOM-055 Rev G.")
    assert plan.prefix == "ECR"
    assert plan.doc_id == "BOM-055"


def test_query_plan_classifies_core_sample_queries():
    cases = {
        "Find the Bill of Materials for the MX1 system and return the latest active revision.": (
            "known_item",
            "exact_then_hybrid",
            "BOM",
            "mx1_bom",
        ),
        "What verification test protocols do we have for the MX1? Group them by protocol family.": (
            "exploratory",
            "sql_list",
            "VVPR",
            "vvpr_inventory",
        ),
        "How many engineering change requests are in the system?": (
            "enumeration",
            "sql_count",
            "ECR",
            "ecr_count",
        ),
        "What are the acceptance criteria for the electrical safety verification test?": (
            "extraction",
            "hybrid",
            "VVPR",
            "electrical_safety_acceptance",
        ),
        "Does our Design History File include everything required by FDA 21 CFR 820.30?": (
            "compliance",
            "hybrid",
            "DHF",
            "dhf_82030",
        ),
        "Trace the requirement for electrical leakage testing from the risk file through to the verification report.": (
            "traceability",
            "multi_hop",
            "RSK",
            "electrical_leakage_trace",
        ),
    }
    for query, expected in cases.items():
        plan = plan_query(query)
        assert (plan.category, plan.strategy, plan.prefix, plan.intent) == expected


def test_query_plan_sets_specialized_qms_intents():
    cases = {
        "Where is the 510(k) summary for the device?": "510k_summary_location",
        "Show me all risk-related documents": "risk_related_inventory",
        "Which verification protocols trace back to the risk analysis?": "risk_protocol_trace",
        "Summarize all design review action items that are still open": "open_design_review_actions",
        "Show me all ECRs filed in the last year and their status": "ecr_last_year_status",
        "Map all third-party test reports to the regulatory requirements they satisfy": "third_party_report_mapping",
        "How many verification protocols have we completed vs. planned?": "verification_completed_vs_planned",
    }
    for query, intent in cases.items():
        assert plan_query(query).intent == intent


def test_query_plan_routes_revision_inventory_to_sql():
    count_plan = plan_query("How many BOM revisions are present for BOM-055?")
    assert count_plan.category == "enumeration"
    assert count_plan.strategy == "sql_count"
    assert count_plan.doc_id == "BOM-055"
    assert count_plan.latest_only is False

    chain_plan = plan_query("What is the latest active revision of BOM-055 and which older revisions exist?")
    assert chain_plan.category == "revision_diff"
    assert chain_plan.strategy == "revision_chain"
    assert chain_plan.requires_revision_chain is True
    assert chain_plan.include_obsolete is True


def test_query_plan_obsolete_scope_is_explicit():
    default_plan = plan_query("Find bill of materials evidence for MX1.")
    assert default_plan.include_obsolete is False

    obsolete_plan = plan_query("Find obsolete bill of materials evidence for MX1.")
    assert obsolete_plan.include_obsolete is True

    historical_plan = plan_query("Find historical bill of materials evidence for MX1.")
    assert historical_plan.include_obsolete is True

    all_revisions_plan = plan_query("Find all revisions for the MX1 bill of materials.")
    assert all_revisions_plan.include_obsolete is True
