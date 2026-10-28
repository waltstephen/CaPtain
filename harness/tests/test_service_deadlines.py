import numpy as np
import pytest
from captain_harness.adapters.capx.services import CapXHttpToolServices


def test_retries_share_one_request_timeout(monkeypatch):
    import captain_harness.adapters.capx.services as module
    clock=[100.];calls=[]
    monkeypatch.setattr(module.time,"monotonic",lambda:clock[0])
    class Transport:
        def request(self,method,url,*,payload,timeout_s):
            calls.append(timeout_s)
            clock[0]+=7. if len(calls)==1 else 1.
            if len(calls)==1: raise OSError("first attempt failed")
            return {"ok":True}
    services=CapXHttpToolServices(transport=Transport(),timeout_s=10.,attempts=2)
    assert services._request("POST","http://service",{})=={"ok":True}
    assert calls==[10.,3.]


def test_late_success_is_not_returned_as_in_budget(monkeypatch):
    import captain_harness.adapters.capx.services as module
    clock=[100.]
    monkeypatch.setattr(module.time,"monotonic",lambda:clock[0])
    class Transport:
        def request(self,*args,**kwargs):
            clock[0]+=11.
            return {"ok":True}
    services=CapXHttpToolServices(transport=Transport(),timeout_s=10.,attempts=1)
    with pytest.raises(TimeoutError): services._request("POST","http://service",{})


def test_timeout_keeps_prior_attempts_without_replaying():
    calls=[]
    class Transport:
        def request(self,*args,**kwargs):
            calls.append(kwargs["timeout_s"])
            if len(calls)==1:raise OSError("connection reset")
            raise TimeoutError("socket deadline")
    services=CapXHttpToolServices(transport=Transport(),timeout_s=10.,attempts=3)
    with pytest.raises(TimeoutError) as error:services._request("POST","http://service",{})
    assert len(calls)==2
    assert [a["error_type"] for a in error.value.attempts]==["OSError","TimeoutError"]
