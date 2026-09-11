from local_live.audio import build_echo_cancel_args, build_pulse_echo_cancel_args


def test_pipewire_echo_cancel_args_use_webrtc_and_named_nodes():
    args = build_echo_cancel_args(
        sink_name="Local Live Echo Cancellation Sink",
        source_name="Local Live Echo Cancellation Source",
        capture_name="Local Live Echo Cancellation Capture",
        playback_name="Local Live Echo Cancellation Playback",
        latency="1024/48000",
    )
    assert "libpipewire-module-echo-cancel" not in args
    assert "aec/libspa-aec-webrtc" in args
    assert 'node.name = "Local Live Echo Cancellation Sink"' in args
    assert 'node.name = "Local Live Echo Cancellation Source"' in args
    assert "node.latency = 1024/48000" in args


def test_pulse_echo_cancel_loader_targets_usb_masters_and_webrtc():
    args = build_pulse_echo_cancel_args(
        sink_name="Local Live Echo Cancellation Sink",
        source_name="Local Live Echo Cancellation Source",
        sink_master=54,
        source_master=55,
    )
    assert "source_name=Local_Live_Echo_Cancellation_Source" in args
    assert "sink_name=Local_Live_Echo_Cancellation_Sink" in args
    assert "source_master=55" in args
    assert "sink_master=54" in args
    assert "aec_method=webrtc" in args
