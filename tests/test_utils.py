from src.utils import normalize_game_name, safe_filename, clamp


def test_normalize():
    assert normalize_game_name("csgo") == "Counter-Strike 2"
    assert normalize_game_name("CS2") == "Counter-Strike 2"
    assert normalize_game_name("Some Game") == "Some Game"
    assert normalize_game_name("") == ""


def test_safe():
    assert safe_filename("a/b:c") == "abc"
    assert safe_filename("") == "clip"
    assert safe_filename("x" * 200).__len__() == 60


def test_clamp():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0