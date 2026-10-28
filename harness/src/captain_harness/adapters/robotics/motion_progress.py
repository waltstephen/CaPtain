"""Readable motion phases from existing public controller measurements."""
def motion_progress(value):
    issued=value.get("command_endpoint_issued")
    reached=value.get("endpoint_reached")
    settled=value.get("endpoint_settled")
    if issued is False:stage="path_not_fully_commanded"
    elif issued is True and reached is False:stage="endpoint_tracking"
    elif reached is True and settled is False:stage="endpoint_settling"
    elif reached is True and settled is True:stage="endpoint_settled"
    else:stage="not_assessed"
    steps=value.get("control_steps");paced=value.get("command_pacing_steps")
    return {"stage":stage,"stop_reason":value.get("stopped_reason"),
        "command_endpoint_issued":issued,"measured_endpoint_reached":reached,"endpoint_settled":settled,
        "tracking_wait_control_steps":paced,"total_control_steps":steps,
        "tracking_wait_fraction":paced/steps if isinstance(paced,(int,float)) and isinstance(steps,(int,float)) and steps>0 else None,
        "elapsed_s":value.get("elapsed_s"),"minimum_command_duration_s":value.get("minimum_command_duration_s"),
        "effective_duration_limit_s":value.get("effective_duration_limit_s"),
        "endpoint_settle_samples":value.get("endpoint_settle_samples"),
        "scope":"measured command/tracking/settling status; unchanged stop and success criteria"}
