"""Single source for the placement standard: the contract text and the host completion gate both read these."""
CENTER_STANDARD_M = 0.02          # put-on: object-to-destination XY offset norm must be below this
LOW_CONFIDENCE_SCORE = 0.02       # detections scored below this are not completion evidence
MAX_UNCERTAINTY_CREDIT_M = 0.01   # reported measurement uncertainty may widen the standard by at most this
