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
        ),
        "What verification test protocols do we have for the MX1? Group them by protocol family.": (
            "exploratory",
            "sql_list",
            "VVPR",
        ),
        "How many engineering change requests are in the system?": (
            "enumeration",
            "sql_count",
            "ECR",
        ),
        "What are the acceptance criteria for the electrical safety verification test?": (
            "extraction",
            "hybrid",
            "VVPR",
        ),
        "Does our Design History File include everything required by FDA 21 CFR 820.30?": (
            "compliance",
            "hybrid",
            "DHF",
        ),
        "Trace the requirement for electrical leakage testing from the risk file through to the verification report.": (
            "traceability",
            "multi_hop",
            "RSK",
        ),
    }
    for query, expected in cases.items():
        plan = plan_query(query)
        assert (plan.category, plan.strategy, plan.prefix) == expected


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
