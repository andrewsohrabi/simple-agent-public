import pytest

from agent.search.table_diff import diff_table_rows


def test_diff_table_rows_reports_added_removed_changed_and_unchanged_rows():
    before = [
        {"Req ID": "REQ-1", "Status": "Open", "Owner": "QA"},
        {"Req ID": "REQ-2", "Status": "Pass", "Owner": "V&V"},
        {"Req ID": "REQ-3", "Status": "Pass", "Owner": "QA"},
    ]
    after = [
        {"Req ID": "REQ-1", "Status": "Closed", "Owner": "QA"},
        {"Req ID": "REQ-2", "Status": "Pass", "Owner": "V&V"},
        {"Req ID": "REQ-4", "Status": "Open", "Owner": "RA"},
    ]

    diff = diff_table_rows(before, after, preferred_key_column="req id")

    assert diff.key_column == "Req ID"
    assert diff.added == ({"Req ID": "REQ-4", "Status": "Open", "Owner": "RA"},)
    assert diff.removed == ({"Req ID": "REQ-3", "Status": "Pass", "Owner": "QA"},)
    assert [change.key for change in diff.changed] == ["REQ-1"]
    assert diff.changed[0].changed_cells == ("Status",)
    assert diff.unchanged == ({"Req ID": "REQ-2", "Status": "Pass", "Owner": "V&V"},)


def test_diff_table_rows_tracks_added_and_removed_cells_as_changed_cells():
    before = [{"ID": "T-1", "Result": "Pass", "Notes": "old note"}]
    after = [{"ID": "T-1", "Result": "Fail", "Comment": "new note"}]

    diff = diff_table_rows(before, after, preferred_key_column="ID")

    assert diff.changed[0].changed_cells == ("Result", "Notes", "Comment")
    assert diff.changed[0].before == before[0]
    assert diff.changed[0].after == after[0]


def test_diff_table_rows_can_infer_a_unique_key_column():
    diff = diff_table_rows(
        [{"Test ID": "TC-001", "Result": "Pass"}],
        [{"Test ID": "TC-001", "Result": "Pass"}],
    )

    assert diff.key_column == "Test ID"
    assert diff.unchanged == ({"Test ID": "TC-001", "Result": "Pass"},)


def test_diff_table_rows_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate key 'REQ-1'"):
        diff_table_rows(
            [{"Req ID": "REQ-1"}, {"Req ID": "REQ-1"}],
            [{"Req ID": "REQ-1"}],
            preferred_key_column="Req ID",
        )


def test_diff_table_rows_rejects_missing_key_values():
    with pytest.raises(ValueError, match="missing a non-empty key"):
        diff_table_rows(
            [{"Req ID": "REQ-1"}],
            [{"Req ID": ""}],
            preferred_key_column="Req ID",
        )
