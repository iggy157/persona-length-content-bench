import random

from pcb.content_items import (
    budgets_from_floor,
    floor_length,
    make_bring,
    make_day,
    make_room,
    make_time,
    sample_item_set,
)
from pcb.metrics_b import judge as judge_b
from pcb.textnorm import normalize, to_second_person


def test_normalize():
    assert normalize("i m a mom with two sons .") == "I'm a mom with two sons."
    assert normalize("i don t like burritos . i ve got 3 cats") == "I don't like burritos. I've got 3 cats."


def test_second_person():
    assert to_second_person("Am afraid that I will do the wrong thing.") == "You are afraid that you will do the wrong thing."
    assert to_second_person("Leave a mess in my room.") == "You leave a mess in your room."


def test_item_patterns():
    t = make_time(3, "pm")
    assert t.hit("See you at 3pm!") and t.hit("starts at 3 PM") and t.hit("three o'clock") and t.hit("3:00pm")
    assert not t.hit("at 4pm") and not t.hit("3am")
    r = make_room(202)
    assert r.hit("Room 202") and r.hit("rm. 202") and r.hit("in 202.") and not r.hit("2020")
    b = make_bring("laptop")
    assert b.hit("Bring your laptops") and not b.hit("lap top")
    d = make_day("Monday")
    assert d.hit("moved to Mon.") and d.hit("Monday") and not d.hit("money")
    assert make_day("Sunday").hit("Sun") and not make_day("Sunday").hit("sunny")


def test_floor_and_budgets():
    items = [make_time(3, "pm"), make_room(202), make_bring("laptop")]
    assert floor_length(items, "words") == 3
    assert floor_length(items, "chars") == len("3pm 202 laptop")
    assert budgets_from_floor(5, [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 0]) == [3, 4, 5, 8, 10, 15, 0]
    for it in items + [make_day("Friday")]:
        assert it.hit(it.minimal)
        # minimal must be the shortest accepted form: no single word of it can be dropped
        words = it.minimal.split()
        for i in range(len(words)):
            shorter = " ".join(words[:i] + words[i + 1:])
            assert not shorter or not it.hit(shorter), (it.kind, shorter)


def test_sample_item_set_deterministic():
    a = sample_item_set(random.Random(1), 3)
    b = sample_item_set(random.Random(1), 3)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert len({x.kind for x in a}) == 3


S1 = {"raw": "i am a pediatrician .", "text": "I am a pediatrician.", "attribute": "pediatrician"}
S2 = {"raw": "i am a nurse .", "text": "I am a nurse.", "attribute": "nurse"}


def test_judge_b():
    items = [make_time(3, "pm"), make_room(202), make_bring("laptop")]
    j = judge_b("Hey! 3pm, room 202, bring your laptop. See you there!", items, 10, "words")
    assert j["length"] == 10 and j["compliant"] and j["content_all"]
    j = judge_b("3pm room 202 bring laptop", items, 3, "words")
    assert j["length"] == 5 and not j["compliant"] and abs(j["overshoot"] - 2 / 3) < 1e-9
    j = judge_b("Room 202 tomorrow.", items, 0, "words")
    assert j["compliant"] and j["content_recall"] == 1 / 3


def test_tolerance_and_two_sided_length():
    from pcb.build_b import tolerance
    from pcb.metrics_b import judge as judge_b
    assert [tolerance(b) for b in (2, 5, 15, 20, 30, 50)] == [1, 1, 2, 2, 3, 5]
    items = [make_time(3, "pm"), make_room(202), make_bring("laptop")]
    j = judge_b("3pm 202 laptop", items, 3, "words", 2, 4)
    assert j["compliant"] and j["length_score"] == 1.0
    j = judge_b("ok see you at 3pm room 202 bring laptop", items, 3, "words", 2, 4)
    assert not j["compliant"] and j["length_score"] == 0.0 and j["overshoot"] > 0
    j = judge_b("3pm", items, 3, "words", 2, 4)
    assert not j["compliant"] and j["undershoot"] > 0
