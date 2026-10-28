"""Planning-only convex payloads from measured points in the EEF frame."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from captain_harness.ports.deadline import check_deadline


def augment_robot_model(original, points_eef):
    check_deadline()
    if original.carried_geometry_name is not None:
        raise ValueError('a carried model cannot be used as an empty robot description')
    points=np.asarray(points_eef,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or not np.isfinite(points).all():
        raise ValueError('carried geometry requires at least four finite 3D points')
    try:
        hull=ConvexHull(points)
    except QhullError as exc:
        raise ValueError('carried geometry must define a 3D hull') from exc
    check_deadline()
    if hull.volume<=1e-12:
        raise ValueError('carried geometry hull volume is insufficient')
    vertices=points[hull.vertices]
    root=ET.fromstring(original._description_xml)
    prefix='observed_carried_'
    if any(e.get('name','').startswith(prefix) for e in root.iter()):
        raise ValueError('reserved observed geometry namespace is already present')
    eef_name=original._description_options['end_effector']
    bodies=[e for e in root.iter('body') if e.get('name')==eef_name]
    if len(bodies)!=1:
        raise ValueError('detached description must identify exactly one EEF body')
    asset=root.find('asset')
    if asset is None:asset=ET.SubElement(root,'asset')
    mesh_name=prefix+'mesh';geom_name=prefix+'geometry'
    ET.SubElement(asset,'mesh',name=mesh_name,vertex=' '.join(format(float(v),'.17g') for v in vertices.ravel()))
    body=ET.SubElement(bodies[0],'body',name=prefix+'body')
    # Use an unused mask bit to avoid implicit payload/gripper contact pairs.
    # Explicit pairs below still test every non-gripper collision geometry,
    # including adjacent arm links that the ordinary parent filter would skip.
    m=original.model
    used=int(np.bitwise_or.reduce(np.r_[m.geom_contype,m.geom_conaffinity],initial=0))
    if used < 0:raise ValueError('negative robot contact masks are unsupported for payload augmentation')
    bit=1<<used.bit_length()
    if bit>1<<30:raise ValueError('no unused payload contact mask bit')
    ET.SubElement(body,'geom',name=geom_name,type='mesh',mesh=mesh_name,mass='0',
        contype=str(bit),conaffinity='0')
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    pairs=[]
    for geometry in original.geometries:
        if geometry.collidable and not geometry.name.startswith(original.gripper_prefix):
            ET.SubElement(contact,'pair',geom1=geom_name,geom2=geometry.name)
            pairs.append(geometry.name)
    check_deadline()
    result=type(original)(ET.tostring(root,encoding='unicode'),**original._description_options)
    check_deadline()
    if not np.array_equal(result.limits,original.limits):
        raise ValueError('adding observed geometry changed robot joint limits')
    result.carried_geometry_name=geom_name
    result._rigid_eef_geometry_names.add(geom_name)
    result.carried_geometry_receipt={'scope':'partial_observed_convex_hull_in_detached_planning_model',
        'input_points':int(len(points)),'hull_vertices':int(len(vertices)),
        'hull_volume_m3':float(hull.volume),'geometry_name':geom_name,
        'non_gripper_self_collision_pairs':pairs,
        'limitation':'Rigid attachment is assumed from external public evidence; unobserved surfaces remain unknown.'}
    return result
