# coding=utf-8
import itertools
import os
import traceback
from typing import List

import numpy as np
import oyaml as yaml

import geometry as geo
from duckietown_serialization_ds1 import Serializable
from .duckiebot import DB18
from .duckietown_map import DuckietownMap
from .other_objects import (
    Barrier,
    Building,
    Bus,
    Cone,
    Duckie,
    GenericObject,
    House,
    SIGNS,
    Tree,
    Truck,
    Watchtower,
    REGIONS,
    ACTORS,
    Decoration
)
from .tags_db import FloorTag, TagInstance
from .tile import Tile
from .tile_map import TileMap
from .tile_template import load_tile_types
from .traffic_light import TrafficLight
from .. import logger
from ..geo import Scale2D, SE2Transform, PlacedObject
from ..geo.measurements_utils import iterate_by_class
from ..world_duckietown import TileCoords

__all__ = ["create_map", "list_maps", "construct_map", "load_map", "load_map_layers"]

default_layer_kind = {
    0: "asphalt",
    1: "sign_stop",
    2: "floor_tag",
    3: "watchtower",
    4: "parking",
    5: "duckie",
    6: "house"
}

def create_map(H=3, W=3) -> TileMap:
    tile_map = TileMap(H=H, W=W)

    for i, j in itertools.product(range(H), range(W)):
        tile = Tile(kind="mykind", drivable=True)
        tile_map.add_tile(i, j, "N", tile)

    return tile_map


def list_maps() -> List[str]:
    maps_dir = get_maps_dir()

    def f():
        for map_file in os.listdir(maps_dir):
            map_name = map_file.split(".")[0]
            yield map_name

    return list(f())


def get_maps_dir() -> str:
    abs_path_module = os.path.realpath(__file__)
    module_dir = os.path.dirname(abs_path_module)
    d = os.path.join(module_dir, "../data/gd1/maps")
    assert os.path.exists(d), d
    return d


def get_texture_dirs() -> List[str]:
    abs_path_module = os.path.realpath(__file__)
    module_dir = os.path.dirname(abs_path_module)
    d = os.path.join(module_dir, "../data/gd1/textures")
    assert os.path.exists(d), d
    d2 = os.path.join(module_dir, "../data/gd1/meshes")
    assert os.path.exists(d2), d2

    d3 = os.path.join(module_dir, "../data/tag36h11")
    assert os.path.exists(d3), d3

    return [d, d2, d3]


def _get_map_yaml(map_name: str) -> str:
    maps_dir = get_maps_dir()
    fn = os.path.join(maps_dir, map_name + ".yaml")
    if not os.path.exists(fn):
        msg = "Could not find file %s" % fn
        raise ValueError(msg)
    with open(fn) as _:
        data = _.read()
    return data


def load_map(map_name: str) -> DuckietownMap:
    logger.info("loading map %s" % map_name)
    data = _get_map_yaml(map_name)
    yaml_data = yaml.load(data, Loader=yaml.SafeLoader)

    return construct_map(yaml_data)


def load_map_layers(map_dir_name: str) -> DuckietownMap:
    logger.info("loading map from %s" % map_dir_name)

    import os
    abs_path_module = os.path.realpath(__file__)
    logger.info("abs_path_module: " + str(abs_path_module))
    module_dir = os.path.dirname(abs_path_module)
    logger.info("module_dir: " + str(module_dir))
    map_dir = os.path.join(module_dir, "../data/gd2", map_dir_name)
    logger.info("map_dir: " + str(map_dir))
    assert os.path.exists(map_dir), map_dir

    from ..yaml_include import YamlIncludeConstructor
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=map_dir)

    yaml_data_layers = {}
    for layer in os.listdir(map_dir):
        fn = os.path.join(map_dir, layer)
        with open(fn) as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        yaml_data_layers[layer] = yaml_data

    return construct_map_layers(yaml_data_layers["main.yaml"])


def obj_idx():
    if not hasattr(obj_idx, 'idx'):
        obj_idx.idx = -1
    obj_idx.idx += 1
    return obj_idx.idx


def construct_map_layers(yaml_data_layer_main: dict) -> DuckietownMap:
    yaml_data = yaml_data_layer_main["main"]
    from pprint import pprint
    pprint(yaml_data)
    # ============================================= 0 layer =============================================
    yaml_layer0 = yaml_data["layer0"]
    tile_size = yaml_layer0["tile_size"]
    dm = DuckietownMap(tile_size)
    tiles = {p: t for p, t in yaml_layer0.items() if isinstance(p, tuple)}
    assert len(tiles) > 1

    # Create the grid
    A = max(map(lambda t: t[1], tiles)) + 1
    B = max(map(lambda t: t[0], tiles)) + 1
    tm = TileMap(H=B, W=A)

    rect_checker = []
    for i in range(A):
        rect_checker.append([0] * B)
    for p in tiles:
        rect_checker[p[1]][p[0]] += 1
    for j in range(A):
        for i in range(B):
            if rect_checker[j][i] == 0:
                msg = "missing tile at pose " + str([i, j, 0])
                raise ValueError(msg)
            if rect_checker[j][i] >= 2:
                msg = "duplicated tile at pose " + str([i, j, 0])
                raise ValueError(msg)

    templates = load_tile_types()

    DEFAULT_ORIENT = "E"
    for (p, t) in tiles.items():
        if "angle" in t:
            kind = t["type"].strip()
            orient = t["angle"].strip()
            drivable = True
        else:
            kind = t["type"].strip()
            orient = DEFAULT_ORIENT
            drivable = (kind == "4way")

        tile = Tile(kind=kind, drivable=drivable)
        if kind in templates:
            tile.set_object(kind, templates[kind], ground_truth=SE2Transform.identity())

        tm.add_tile(p[0], (A - 1) - p[1], orient, tile)

    dm.set_object("tilemap", tm, ground_truth=Scale2D(tile_size))
    # =========================================== other layers ==========================================
    for layer_num in range(1, 7):
        yaml_layer = yaml_data.get("layer" + str(layer_num), {})
        for p, desc in yaml_layer.items():
            if "kind" not in desc:
                desc["kind"] = default_layer_kind[layer_num]
            kind = desc["kind"]
            obj_name = "ob%02d-%s" % (obj_idx(), kind)

            obj = get_object(desc)

            if layer_num == 4:
                wrapper = PlacedObject()
                wrapper.set_object(obj_name, obj, ground_truth=TileCoords(p[0], (tm.W - 1) - p[1], "E"))
                obj = wrapper
                transform = Scale2D(tile_size)
            else:
                transform = get_transform_(p, desc, tile_size, tm.W)

            dm.set_object(obj_name, obj, ground_truth=transform)
    # ============================================= ending =============================================
    for it in list(iterate_by_class(tm, Tile)):
        ob = it.object
        if "slots" in ob.children:
            slots = ob.children["slots"]
            for k, v in list(slots.children.items()):
                if not v.children:
                    slots.remove_object(k)
            if not slots.children:
                ob.remove_object("slots")
    return dm


def get_transform_(p, desc, tile_size, W):
    if desc.get("relative", False):
        (u, v), r = get_uvr_relative(p, tile_size, W)
    else:
        if isinstance(p[0], tuple):
            x, y = get_xy_slot(p[1])
            i, j = p[0]
            u, v = (x + i) * tile_size, (y + j) * tile_size
            r = 0 if len(p) == 2 else p[4]
        else:
            u, v, r = p[0], p[1], p[5]
    return SE2Transform([u, v], r)


def get_uvr_relative(p, tile_size, W):
    u = float(p[0]) * tile_size
    v = float(W - p[1]) * tile_size
    r = 0 if len(p) == 2 else -p[5]
    return (u, v), r


def construct_map(yaml_data: dict) -> DuckietownMap:
    tile_size = yaml_data["tile_size"]
    dm = DuckietownMap(tile_size)
    tiles = yaml_data["tiles"]
    assert len(tiles) > 0
    assert len(tiles[0]) > 0

    # Create the grid
    A = len(tiles)
    B = len(tiles[0])
    tm = TileMap(H=B, W=A)

    templates = load_tile_types()
    for a, row in enumerate(tiles):
        if len(row) != B:
            msg = "each row of tiles must have the same length"
            raise ValueError(msg)

        # For each tile in this row
        for b, tile in enumerate(row):
            tile = tile.strip()

            if tile == "empty":
                continue

            DEFAULT_ORIENT = "E"  # = no rotation
            if "/" in tile:
                kind, orient = tile.split("/")
                kind = kind.strip()
                orient = orient.strip()

                drivable = True
            elif "4" in tile:
                kind = "4way"
                # angle = 2
                orient = DEFAULT_ORIENT
                drivable = True
            else:
                kind = tile
                # angle = 0
                orient = DEFAULT_ORIENT
                drivable = False

            tile = Tile(kind=kind, drivable=drivable)
            if kind in templates:
                tile.set_object(kind, templates[kind], ground_truth=SE2Transform.identity())
            else:
                pass
                # msg = 'Could not find %r in %s' % (kind, templates)
                # logger.debug(msg)

            tm.add_tile(b, (A - 1) - a, orient, tile)

    def go(obj_name, desc):
        obj = get_object(desc)
        transform = get_transform(desc, tm, tile_size)
        dm.set_object(obj_name, obj, ground_truth=transform)

    objects = yaml_data.get("objects", [])
    if isinstance(objects, list):
        for obj_idx, desc in enumerate(objects):
            kind = desc["kind"]
            obj_name = "ob%02d-%s" % (obj_idx, kind)
            go(obj_name, desc)
    elif isinstance(objects, dict):
        for obj_name, desc in objects.items():
            go(obj_name, desc)
    else:
        raise ValueError(objects)

    for it in list(iterate_by_class(tm, Tile)):
        ob = it.object
        if "slots" in ob.children:
            slots = ob.children["slots"]
            for k, v in list(slots.children.items()):
                if not v.children:
                    slots.remove_object(k)
            if not slots.children:
                ob.remove_object("slots")

    dm.set_object("tilemap", tm, ground_truth=Scale2D(tile_size))
    return dm


def get_object(desc):
    kind = desc["kind"]

    attrs = {}
    if "tag" in desc:
        tag_desc = desc["tag"]
        # tag_id = tag_desc.get('tag_id')
        # size = tag_desc.get('size', DEFAULT_TAG_SIZE)
        # family = tag_desc.get('family', DEFAULT_FAMILY)
        attrs["tag"] = Serializable.from_json_dict(tag_desc)

    if kind == "floor_tag":
        attrs["tag"] = TagInstance(desc["tag_id"], desc["family"], desc["size"])

    kind2klass = {
        "trafficlight": TrafficLight,
        "duckie": Duckie,
        "cone": Cone,
        "barrier": Barrier,
        "building": Building,
        "duckiebot": DB18,
        "tree": Tree,
        "house": House,
        "bus": Bus,
        "truck": Truck,
        "floor_tag": FloorTag,
        "watchtower": Watchtower
    }
    kind2klass.update(SIGNS)
    kind2klass.update(REGIONS)
    kind2klass.update(ACTORS)
    if kind in kind2klass:
        klass = kind2klass[kind]
        try:
            obj = klass(**attrs)
        except TypeError:
            msg = "Could not initialize %s with attrs %s:\n%s" % (
                klass.__name__,
                attrs,
                traceback.format_exc(),
            )
            raise Exception(msg)

    else:
        logger.debug("Do not know special kind %s" % kind)
        obj = GenericObject(kind=kind)
    return obj


def get_transform(desc, tm, tile_size):
    rotate_deg = desc.get("rotate", 0)
    rotate = np.deg2rad(rotate_deg)

    if "pos" in desc:
        pos = desc["pos"]
        x = float(pos[0]) * tile_size
        # account for non-righthanded
        y = float(tm.W - pos[1]) * tile_size
        # account for non-righthanded
        rotate = -rotate
        transform = SE2Transform([x, y], rotate)
        return transform

    else:

        if "pose" in desc:
            pose = Serializable.from_json_dict(desc["pose"])
        else:
            pose = SE2Transform.identity()

        if "attach" in desc:
            attach = desc["attach"]
            tile_coords = tuple(attach["tile"])
            slot = str(attach["slot"])

            x, y = get_xy_slot(slot)
            i, j = tile_coords

            u, v = (x + i) * tile_size, (y + j) * tile_size
            transform = SE2Transform([u, v], rotate)

            q = geo.SE2.multiply(transform.as_SE2(), pose.as_SE2())

            return SE2Transform.from_SE2(q)
        else:
            return pose


def get_xy_slot(i):
    # tile_offset
    # to = 0.20
    # # tile_curb
    # tc = 0.05
    to = 0.09  # [m] the longest of two distances from the center of the tag to the edge
    tc = 0.035  # [m] the shortest of two distances from the center of the tag to the edge

    positions = {
        0: (+to, +tc),
        1: (+tc, +to),
        2: (-tc, +to),
        3: (-to, +tc),
        4: (-to, -tc),
        5: (-tc, -to),
        6: (+tc, -to),
        7: (+to, -tc),
    }
    x, y = positions[int(i)]
    return x, y


def get_texture_file(tex_name: str) -> str:
    res = []
    tried = []
    for d in get_texture_dirs():
        suffixes = ["", "_1", "_2", "_3", "_4"]
        for s in suffixes:
            for ext in [".jpg", ".png", ""]:
                path = os.path.join(d, tex_name + s + ext)
                tried.append(path)
                if os.path.exists(path):
                    res.append(path)

    if not res:
        msg = "Could not find any texture for %s" % tex_name
        logger.debug("tried %s" % tried)
        raise KeyError(msg)
    return res[0]
