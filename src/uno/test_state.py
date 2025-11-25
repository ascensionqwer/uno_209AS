"""Tests for UNO state and Action class."""

import pytest
from .state import Action
from .cards import RED, YELLOW, BLACK, WILD


def test_action_validation_must_specify_one():
    """Test Action validation: must specify X_1 OR n, not both."""
    # Valid: only X_1
    action1 = Action(X_1=(RED, 5))
    assert action1.is_play() is True
    assert action1.is_draw() is False

    # Valid: only n
    action2 = Action(n=1)
    assert action2.is_play() is False
    assert action2.is_draw() is True

    # Invalid: both specified
    with pytest.raises(ValueError, match="Must specify EITHER"):
        Action(X_1=(RED, 5), n=1)

    # Invalid: neither specified
    with pytest.raises(ValueError, match="Must specify EITHER"):
        Action()


def test_action_validation_draw_n_values():
    """Test draw action validation: n must be in {1, 2, 4}."""
    # Valid values
    Action(n=1)
    Action(n=2)
    Action(n=4)

    # Invalid values
    with pytest.raises(ValueError, match="n must be in"):
        Action(n=0)

    with pytest.raises(ValueError, match="n must be in"):
        Action(n=3)

    with pytest.raises(ValueError, match="n must be in"):
        Action(n=5)


def test_action_type_checking():
    """Test action type checking (is_play, is_draw)."""
    play_action = Action(X_1=(RED, 5))
    assert play_action.is_play() is True
    assert play_action.is_draw() is False

    draw_action = Action(n=1)
    assert draw_action.is_play() is False
    assert draw_action.is_draw() is True


def test_action_representation_play():
    """Test Action representation for play actions."""
    action = Action(X_1=(RED, 5))
    repr_str = repr(action)
    assert "X_1" in repr_str
    assert "RED" in repr_str or "5" in repr_str

    # With wild color
    wild_action = Action(X_1=(BLACK, WILD), wild_color=YELLOW)
    repr_str = repr(wild_action)
    assert "wild_color" in repr_str
    assert "YELLOW" in repr_str or "Y" in repr_str


def test_action_representation_draw():
    """Test Action representation for draw actions."""
    action = Action(n=1)
    repr_str = repr(action)
    assert "Y_1" in repr_str or "n=1" in repr_str

    # After drawing (Y_n filled)
    action.Y_n = [(RED, 5)]
    repr_str = repr(action)
    assert "Y_1" in repr_str or "Y_n" in repr_str


def test_action_wild_color():
    """Test Action with wild_color parameter."""
    action = Action(X_1=(BLACK, WILD), wild_color=RED)
    assert action.wild_color == RED
    assert action.X_1 == (BLACK, WILD)
