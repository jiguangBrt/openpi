from openpi_client import websocket_client_policy


class _FakePacker:
    def pack(self, value):
        del value
        return b"request"


class _FakeWebsocket:
    def send(self, value):
        assert value == b"request"

    def recv(self):
        return b"response"


def test_infer_reports_client_transport_segments(monkeypatch):
    policy = object.__new__(websocket_client_policy.WebsocketClientPolicy)
    policy._packer = _FakePacker()
    policy._ws = _FakeWebsocket()
    monkeypatch.setattr(
        websocket_client_policy.msgpack_numpy,
        "unpackb",
        lambda value: {
            "actions": [],
            "server_timing": {
                "request_deserialization_ms": 0.1,
                "queue_ms": 0.2,
                "infer_ms": 0.3,
                "response_serialization_ms": 0.1,
            },
        },
    )

    result = policy.infer({"observation": 1})

    timing = result["client_timing"]
    assert timing["request_serialization_ms"] >= 0.0
    assert timing["transport_round_trip_ms"] >= 0.0
    assert timing["network_round_trip_estimate_ms"] >= 0.0
    assert timing["response_decode_ms"] >= 0.0
    assert timing["total_ms"] >= timing["transport_round_trip_ms"]
