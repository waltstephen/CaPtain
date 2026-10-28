"""Portable public measurement record integrity; no environment interpretation."""
import hashlib,json
def content_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def observation_content(body):
    # IDs, wall-clock labels and requested hypothetical poses are not new observations.
    sources=[]
    for source in body["sources"]:
        sources.append({key:source.get(key) for key in ("camera","rgb_pixels_sha256","rgb_shape",
            "depth_values_sha256","mask_sha256","intrinsics_3x3","camera_to_robot_base_4x4")})
    return sorted(sources,key=lambda row:content_digest(row))
def make_measurement(body):
    digest=content_digest(body)
    return {"measurement_id":"measurement://"+digest,"body_sha256":digest,
        "evidence_content_sha256":content_digest(observation_content(body)),
        "measurement_body":body,"depth_status":"derived_public_measurement"}
def validate_measurement(record):
    if not isinstance(record,dict) or not isinstance(record.get("measurement_body"),dict):
        raise ValueError("measurement body must be an object")
    body=record["measurement_body"]
    if (not isinstance(body.get("sources"),list) or not body["sources"] or
        any(not isinstance(source,dict) for source in body["sources"])):
        raise ValueError("measurement needs source observation objects")
    if not isinstance(body.get("kind"),str) or not isinstance(body.get("result"),dict):
        raise ValueError("measurement needs a kind and structured result")
    digest=content_digest(body)
    if record.get("body_sha256")!=digest or record.get("measurement_id")!="measurement://"+digest:
        raise ValueError("measurement content/ID mismatch")
    if record.get("evidence_content_sha256")!=content_digest(observation_content(body)):
        raise ValueError("measurement observation signature mismatch")
    if not isinstance(body.get("sources"),list) or not body["sources"]:
        raise ValueError("measurement has no source observations")
    return body
