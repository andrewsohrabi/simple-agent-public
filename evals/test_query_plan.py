from agent.search.query_plan import plan_query


def test_query_plan_routes_enumeration_to_sql_count():
    plan = plan_query("How many engineering change requests are in the system?")
    assert plan.category == "enumeration"
    assert plan.strategy == "sql_count"
    assert plan.prefix == "ECR"
    assert plan.requires_count is True


def test_query_plan_extracts_revision_diff():
    plan = plan_query("What changed between Rev C and Rev D of the risk analysis?")
    assert plan.category == "revision_change_tracking"
    assert plan.requires_diff is True
    assert plan.include_obsolete is True


def test_query_plan_prefers_ecr_when_query_mentions_bom_change_record():
    plan = plan_query("Find the ECR for BOM-055 Rev G.")
    assert plan.prefix == "ECR"
    assert plan.doc_id == "BOM-055"
