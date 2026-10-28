"""Bounded local reader pages. This byte allowance is not an LLM token budget."""
import json
from .read_errors import ToolInputError

LOCAL_RPC_BYTES=65536


def encode(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)


def rpc_size(value):
    return len((encode({"event":"public_data_result","success":True,"value":value})+"\n").encode("utf-8"))


def artifact_page(value, *, identity, path, offset, limit, describe):
    total=len(value) if isinstance(value,(list,dict,str)) else 1
    if offset>total or (not isinstance(value,(list,dict,str)) and offset):
        raise ToolInputError("offset exceeds this value",code="artifact_page_out_of_range",
            read_tool="read_tool_result",
            read_args={"artifact_id":identity,"path":list(path),"offset":0,"limit":limit})
    unit="characters" if isinstance(value,str) else "items"
    def page(data,count):
        end=offset+count
        next_offset=end if end<total else None
        return {"artifact_id":identity,"path":list(path),"data":data,"type":type(value).__name__,
            "offset":offset,"next_offset":next_offset,"total_items":total,"total_count":total,
            "returned_count":count,"page_unit":unit,"rpc_byte_limit":LOCAL_RPC_BYTES,
            "read_delivery":"local-direct-v2","read_tool":"read_tool_result",
            "next_read_args":{"artifact_id":identity,"path":list(path),"offset":end,"limit":limit}
                if next_offset is not None else None,
            "scope":"immutable historical value; not refreshed sensor state"}
    def fits(data,count):return rpc_size(page(data,count))<=LOCAL_RPC_BYTES
    if isinstance(value,str):
        # At most this many characters can fit in the byte frame; JSON escaping
        # and UTF-8 are measured on the complete final response below.
        lo=offset;hi=min(total,offset+LOCAL_RPC_BYTES)
        if fits(value[offset:hi],hi-offset):return page(value[offset:hi],hi-offset)
        while hi-lo>1:
            mid=(lo+hi)//2
            if fits(value[offset:mid],mid-offset):lo=mid
            else:hi=mid
        if lo==offset:
            raise ToolInputError("read metadata exceeds local RPC allowance",code="artifact_metadata_too_large")
        return page(value[offset:lo],lo-offset)
    if not isinstance(value,(list,dict)):
        result=page(value,1)
        if rpc_size(result)>LOCAL_RPC_BYTES:
            raise ToolInputError("scalar exceeds local RPC allowance",code="artifact_metadata_too_large")
        return result
    pairs=list(value.items()) if isinstance(value,dict) else list(enumerate(value))
    selected=[]
    def pack(pairs):return dict(pairs) if isinstance(value,dict) else [v for _,v in pairs]
    for key,child in pairs[offset:offset+limit]:
        if not fits(pack([(key,child)]),1):
            child_path=[*path,key]
            if len(child_path)<=32:
                ref={"artifact_id":identity,"path":child_path,"type":type(child).__name__,
                    "size_bytes":len(encode(child).encode("utf-8")),
                    "length":len(child) if isinstance(child,(dict,list,str)) else None,
                    "scope":"child of the same immutable public artifact"}
            else:
                # A direct immutable subtree, not a serialized response/page.
                ref=describe(child)
            child={"__kind__":"artifact","reference":ref}
        candidate=[*selected,(key,child)]
        if not fits(pack(candidate),len(candidate)):break
        selected=candidate
    if not selected and offset<total:
        raise ToolInputError("field metadata exceeds local RPC allowance",code="artifact_metadata_too_large")
    result=page(pack(selected),len(selected))
    if rpc_size(result)>LOCAL_RPC_BYTES:
        raise ToolInputError("page metadata exceeds local RPC allowance",code="artifact_metadata_too_large")
    return result
