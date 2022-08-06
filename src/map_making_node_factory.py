from .input_map_adapter import InputMapAdapter
from .occupancy_map_maker import OccupancyMapMaker
from .output_map_adapter import OutputMapAdapter
from .map_making_node import MapMakingNode


class MapMakingNodeFactory(object):
    def __init__(self):
        pass

    def make_map_making_node(self):
        input_map_adapter = InputMapAdapter()
        occupancy_map_maker = OccupancyMapMaker()
        output_map_adapter = OutputMapAdapter()

        map_making_node = MapMakingNode(input_map_adapter, occupancy_map_maker, output_map_adapter)
