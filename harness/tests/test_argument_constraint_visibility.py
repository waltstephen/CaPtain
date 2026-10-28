from captain_harness.ports.capability import CapabilityManifest, CapabilitySpec, SideEffect


def manifest():
    return CapabilityManifest('generic/v1','generic',(CapabilitySpec('adjust_channel','Generic input constraints.',
        {'type':'object','properties':{
            'gain':{'type':'number','exclusiveMinimum':0,'maximum':18,'default':18},
            'mode':{'enum':['soft','firm']},
            'samples':{'type':'array','items':{'type':'number'},'minItems':3,'maxItems':3},
            'vector':{'type':['array','null'],'items':{'type':'number'},'default':None},
            'axis':{'type':['array','null'],'items':{'type':'number'},'default':None}},
         'required':['mode'],'oneOf':[{'required':['vector'],'properties':{'axis':{'type':'null'}}},
                                    {'required':['axis'],'properties':{'vector':{'type':'null'}}}]},
        {'type':'object'}, {},{},1,SideEffect.COMPUTE,
        argument_order=('gain','mode','samples','vector','axis')),))


def test_rendered_interface_shows_limits_and_exclusive_argument_groups():
    text=manifest().render_python_interface()
    assert 'maximum' in text and '18' in text and 'exclusiveMinimum' in text
    assert 'minItems' in text and 'maxItems' in text
    assert 'oneOf' in text and 'required' in text
    assert 'soft' in text and 'firm' in text
    assert 'mode: str' in text


def test_constraint_projection_retains_zero_false_and_does_not_mutate_schema():
    from copy import deepcopy
    from captain_harness.ports.schema_constraints import public_argument_constraints
    schema={'type':'object','properties':{'x':{'type':'integer','minimum':0,'maximum':1},
        'enabled':{'const':False},'ids':{'type':'array','uniqueItems':True,'minItems':0},
        'record':{'type':'object','properties':{'long_details':{'type':'object'}}}},'required':['x']}
    before=deepcopy(schema)
    value=public_argument_constraints(schema)
    assert value['arguments']['x']=={'minimum':0,'maximum':1}
    assert value['arguments']['enabled']=={'const':False}
    assert value['arguments']['ids']['minItems']==0
    assert 'record' in value['nested_object_arguments']
    assert schema==before


def test_nested_array_item_bounds_are_visible_and_explicit_types_remain_authoritative():
    from captain_harness.ports.capability import _python_annotation
    from captain_harness.ports.schema_constraints import public_argument_constraints
    schema={'type':'object','properties':{'angles':{'type':['array','null'],
        'items':{'type':'number','minimum':-3.14,'maximum':3.14},'maxItems':12}}}
    result=public_argument_constraints(schema)
    assert result['arguments']['angles']['items']=={'minimum':-3.14,'maximum':3.14}
    assert _python_annotation({'oneOf':[{'type':'string'},{'type':'integer'}]})=='str | int'
    assert _python_annotation({'enum':[False,None]})=='bool | None'
    assert _python_annotation({'type':'number','enum':[0,1]})=='float'
