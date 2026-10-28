from test_partial_attachment_feedback import setup_case


def test_partial_cloud_reports_unknown_motion_instead_of_failed_motion_threshold(tmp_path):
    f,track,handle,mark=setup_case(tmp_path,partial=True)
    r=f.backend.assert_entity_attached(handle,current_point_handle=mark)
    checks={x['name']:x for x in r.value['attachment_evidence_gaps']['checks']}
    assert checks['visible_extent_comparable']['status']=='unmet'
    assert checks['target_translation']['status']=='unknown'
    assert checks['target_vertical_translation']['status']=='unknown'
    assert r.value['verified'] is False
    assert checks['target_vertical_translation']['threshold_m']==.025


def test_comparable_subthreshold_motion_reports_measured_gap(tmp_path):
    f,track,handle,mark=setup_case(tmp_path,partial=False)
    r=f.backend.assert_entity_attached(handle,current_point_handle=mark)
    gaps=r.value['attachment_evidence_gaps']
    checks={x['name']:x for x in gaps['checks']}
    assert checks['visible_extent_comparable']['status']=='met'
    assert checks['target_translation']['status']=='unmet'
    assert checks['target_translation']['measured_m']==0.
    assert gaps['task_recovery_verified'] is False
    assert gaps['sustained_retention_verified'] is False


def test_missing_source_camera_and_appearance_remain_unknown(tmp_path):
    f,track,handle,mark=setup_case(tmp_path,partial=False)
    track.source_camera_name=None
    track.appearance_rgb_mean=None
    r=f.backend.assert_entity_attached(handle,current_point_handle=mark)
    checks={x['name']:x for x in r.value['attachment_evidence_gaps']['checks']}
    assert r.value['motion_diagnosis']['observation_comparability']=='unresolved'
    assert checks['same_camera_coverage']['status']=='unknown'
    assert checks['appearance_consistent']['status']=='unknown'
    assert checks['target_translation']['status']=='unknown'
