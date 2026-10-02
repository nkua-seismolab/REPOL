"""SeisComP event routing and comment persistence contracts."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from repol.core.models import PickResult
from repol.scclient import LOAD_INVENTORY, OUTPUT_GROUP, SUBSCRIPTIONS

datamodel = pytest.importorskip("seiscomp.datamodel")
listener = pytest.importorskip("repol.scclient.listener")
pick_io = pytest.importorskip("repol.scclient.pick_io")


def test_fixed_topology_and_event_only_trigger():
    assert SUBSCRIPTIONS == ("EVENT", "LOCATION")
    assert OUTPUT_GROUP == "PICK"
    assert LOAD_INVENTORY is False
    app = SimpleNamespace(pending_events={}, processed_events=set(), wait_time=300)
    for kind in (datamodel.Pick, datamodel.Origin, datamodel.FocalMechanism):
        listener.EventListenerApp.addObject(app, "", kind.Create())
    assert app.pending_events == {}
    event = datamodel.Event.Create()
    listener.EventListenerApp.addObject(app, "", event)
    first_seen = app.pending_events[event.publicID()][1]
    listener.EventListenerApp.addObject(app, "", event)
    assert len(app.pending_events) == 1
    assert app.pending_events[event.publicID()][1] == first_seen


def test_database_children_loaded():
    origin = datamodel.Origin.Create()
    pick = datamodel.Pick.Create()
    query = Mock()
    query.getPicks.return_value = iter([pick])
    app = SimpleNamespace(query=lambda: query)
    assert listener.EventListenerApp._load_origin(app, origin.publicID()) == origin
    query.loadArrivals.assert_called_once_with(origin)
    assert listener.EventListenerApp._load_picks(app, origin) == [pick]
    query.loadComments.assert_called_once_with(pick)


def test_reprocessing_updates_existing_json_comment():
    pick = datamodel.Pick.Create()
    result = PickResult(pick_id=pick.publicID(), polarity="U", onset="I", sp_ratio=2.5)
    assert pick_io.apply_result(pick, result)
    result.sp_ratio = 3.5
    assert pick_io.apply_result(pick, result)
    assert pick.commentCount() == 1
    assert json.loads(pick.comment(0).text())["sp_ratio"] == "3.500"
