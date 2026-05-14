from agent.search.query_expansion import expand_query, normalize_query


def test_query_expansion_normalizes_510k_k_numbers_and_dhf_terms():
    expansion = expand_query("  find k123456 and 510k DHF evidence  ")

    assert expansion.normalized_query == "find K123456 and 510(k) DHF evidence"
    assert expansion.matched_groups == ("510k", "dhf")
    assert "premarket notification" in expansion.terms
    assert "design history file" in expansion.terms
    assert "21 CFR 820.30" in expansion.terms
    assert "FDA 510(k)" in expansion.expanded_query


def test_query_expansion_covers_change_risk_trace_and_third_party_aliases():
    expansion = expand_query("Map dco updates from rmf and pfmea to vvam trace matrix and 3p lab reports")

    assert expansion.normalized_query == (
        "Map DCO updates from RMF and PFMEA to VVAM trace matrix and 3P lab reports"
    )
    assert expansion.matched_groups == (
        "ecr_dco",
        "rmf_rsk",
        "fmea",
        "vvam_traceability",
        "third_party",
    )
    assert "engineering change request" in expansion.terms
    assert "ECR" in expansion.terms
    assert "RSK" in expansion.terms
    assert "risk management file" in expansion.terms
    assert "DFMEA" in expansion.terms
    assert "requirements traceability matrix" in expansion.terms
    assert "third-party" in expansion.terms


def test_query_expansion_covers_iec_electrical_safety_and_architecture_terms():
    expansion = expand_query("iec60601-1 leakage dielectric sw architecture")

    assert expansion.normalized_query == "IEC 60601-1 leakage dielectric sw architecture"
    assert expansion.matched_groups == (
        "iec_60601",
        "electrical_safety",
        "software_architecture",
    )
    assert "IEC 60601-1-2" in expansion.terms
    assert "electrical safety" in expansion.terms
    assert "leakage current" in expansion.terms
    assert "dielectric strength" in expansion.terms
    assert "software architecture" in expansion.terms
    assert "system architecture" in expansion.terms


def test_query_expansion_is_deterministic_and_deduplicated():
    first = expand_query("third party 3p IEC 60601 EMC")
    second = expand_query("third party 3p IEC 60601 EMC")

    assert first == second
    assert len(first.terms) == len({term.casefold() for term in first.terms})


def test_normalize_query_preserves_plain_queries():
    assert normalize_query("latest BOM evidence") == "latest BOM evidence"
