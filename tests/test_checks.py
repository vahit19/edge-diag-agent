"""Checks that run without the manuals or a model: identifier parsing and the answer gates."""

from diag.agent import ack_conflicts, excerpt, ground, stated_ack
from diag.retrieve import Doc, identifiers

KNOWN = {"F07900", "A07900", "F07807", "F01910", "p2040", "p1900", "r0035"}


def doc(id_, text, why=("exact",), kind="message"):
    return Doc(id_, "SINAMICS G120C", "manual.pdf", 1, kind, "title", text, why=list(why))


def test_identifiers_normalise_the_shapes_technicians_type():
    assert identifiers("G120C shows F7900 in automatic mode", KNOWN) == ["F07900"]
    assert identifiers("drive tripped on O.F 7807", KNOWN) == ["F07807"]
    assert identifiers("getting the 1910 error after a PLC restart", KNOWN) == ["F01910"]
    assert identifiers("is P2040 too low? r0035 shows 40", KNOWN) == ["p2040", "r0035"]


def test_identifiers_drop_numbers_that_are_not_in_the_manuals():
    assert identifiers("F1234 and p9999", KNOWN) == []


def test_ground_accepts_ids_from_question_and_excerpts():
    d = doc("F07900", "F07900 Motor blocked\nRemedy:\ncheck p1900 and p0600")
    mentioned, unsupported, invented = ground("F7900: run p1900, see p0600 and F07807", "why F7900?", [d], KNOWN)
    assert mentioned == ["F07900", "F07807", "p1900", "p0600"]
    assert unsupported == ["F07807"]   # in the manuals, not shown to the model
    assert invented == []              # p0600 was in the excerpt, though not in this parameter list


def test_ground_flags_numbers_that_exist_nowhere():
    _, unsupported, invented = ground("set p4711", "q", [doc("F07900", "F07900")], KNOWN)
    assert (unsupported, invented) == ([], ["p4711"])


def test_stated_ack_reads_the_mode_next_to_the_word():
    assert stated_ack("How to acknowledge: POWER ON (switch off and on).") == {"power on"}
    assert stated_ack("Acknowledgment is not required for this alarm.") == {"none"}
    assert stated_ack("Carry out a power on. Then check the cable.") == set()  # no acknowledgment statement


def test_stated_ack_ignores_the_manuals_without_power_on_wording():
    assert stated_ack("It can be acknowledged without a power on via Acknowledge internal event.") == set()


def test_ack_conflict_catches_reaction_copied_as_acknowledgment():
    fact = [{"code": "F01018", "acknowledge": "POWER ON"}]
    assert ack_conflicts("(4) How to acknowledge: NONE", fact) == ["F01018"]
    assert ack_conflicts("(4) How to acknowledge: POWER ON", fact) == []
    assert ack_conflicts("Check the cable and restart.", fact) == []


def test_ack_conflict_accepts_either_mode_of_a_combined_entry():
    fact = [{"code": "F30682", "acknowledge": "IMMEDIATELY (POWER ON)"}]
    assert ack_conflicts("Acknowledge immediately.", fact) == []


def test_ack_conflict_is_skipped_when_several_codes_matched():
    facts = [{"code": "A07409", "acknowledge": "NONE"}, {"code": "F07900", "acknowledge": "IMMEDIATELY"}]
    assert ack_conflicts("Acknowledge: NONE", facts) == []


def test_short_excerpts_keep_the_matched_entry_whole():
    long = "x" * 1000
    assert excerpt(doc("F07900", long), other_chars=400) == long
    assert excerpt(doc("p1900", long, why=("fused",), kind="parameter"), other_chars=400) == "x" * 400 + " ..."
