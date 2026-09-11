from local_live.audio import AudioNode, _parse_nodes, _prefer_usb
from local_live.doctor import _github_auth_is_valid


def test_github_auth_marker_overrides_zero_exit_code():
    assert _github_auth_is_valid(0, "github.com\n  X Failed to log in", "") is False
    assert _github_auth_is_valid(0, "github.com\n  ✓ Logged in to github.com", "") is True


def test_pipewire_parser_does_not_treat_endpoints_as_physical_nodes():
    status = """
Audio
 ├─ Sinks:
 │  * 54. USB Speaker [vol: 1.00]
 ├─ Sink endpoints:
 │     69. USB Speaker Endpoint
 ├─ Sources:
 │  * 55. USB Microphone [vol: 1.00]
 ├─ Source endpoints:
 │     70. USB Microphone Endpoint
 └─ Streams:
"""
    assert [node.node_id for node in _parse_nodes(status, "Sinks:")] == [54]
    assert [node.node_id for node in _parse_nodes(status, "Sources:")] == [55]


def test_explicit_usb_audio_is_preferred_over_other_usb_capture_devices():
    nodes = [
        AudioNode(55, "GoStream アナログステレオ", "source"),
        AudioNode(71, "USB Audio アナログステレオ", "source"),
    ]
    assert _prefer_usb(nodes).node_id == 71
