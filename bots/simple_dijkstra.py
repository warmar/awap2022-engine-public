import sys

import random
import math
import numpy as np

from src.game import *
from src.player import *
from src.structure import *
from src.game_constants import GameConstants as GC

class AugmentedTile(Tile):
    def __init__(self, tile, closest_tile):
        Tile.__init__(self, tile.x ,tile.y, tile.passability, tile.population, tile.structure)
        self.closest_tile = closest_tile
        self.dist = math.hypot(tile.x - closest_tile.x, tile.y - closest_tile.y)
        self.reached = False

    def update_closest_tile(self, tile):
        dist = math.hypot(tile.x - tile.x, tile.y - tile.y)
        if dist < self.dist:
            self.closest_tile = tile

    def dist_to(self, tile):
        return math.hypot(tile.x - self.x, tile.y - self.y)

    def _copy(self):
        return AugmentedTile(Tile(self.x, self.y, self.passability,
                                    self.population, Structure.make_copy(self.structure)),
                                    self.closest_tile)

    def __str__(self):
        return f"[{(self.x, self.y)} {self.passability}, {self.population} {self.structure} {self.closest_tile}]"

    def __repr__(self):
        return f"[{(self.x, self.y)} {self.passability}, {self.population} {self.structure} {self.closest_tile}]"

class MyPlayer(Player):

    def __init__(self):
        print("Init")
        self.turn = 0
        self.prev_builds = []
        self.population_tiles = []
        self.team = None
        self.generators = []

        self.target = None
        self.last_target_dist = 9e9
        self.target_dist_turn_decreased = 0
        self.TARGET_GIVE_UP = 10
        return
    
    def get_closest_tile(self, x, y, tiles):
        dists = np.array([math.hypot(x - tile.x, y - tile.y) for tile in tiles])
        return tiles[np.argmin(dists)]

    def load_info(self, map, player_info):
        self.MAP_WIDTH = len(map)
        self.MAP_HEIGHT = len(map[0])

        self.generators = []
        for x in range(self.MAP_WIDTH):
            for y in range(self.MAP_HEIGHT):
                st = map[x][y].structure
                if st is not None and  st.team == player_info.team:
                    self.generators.append(st)

        for x in range(self.MAP_WIDTH):
            for y in range(self.MAP_HEIGHT):
                tile = map[x][y]
                if tile is None:
                    continue
                if tile.population > 0:
                    self.population_tiles.append(AugmentedTile(tile, self.get_closest_tile(tile.x, tile.y, self.generators)))

    def init_turn(self, turn_num, map, player_info):
        if turn_num == 0:
            self.load_info(map, player_info)
        self.check_prev_builds(map, player_info)
        self.prev_builds = []
        self.load_next_target(turn_num, map, player_info)

    def check_prev_builds(self, map, player_info):
        for build_type, x, y in self.prev_builds:
            tile = map[x][y]
            st = tile.structure
            if st is None or st.team != player_info.team:
                continue
            if st.type == build_type:
                for tile in self.population_tiles:
                    tile.update_closest_tile(tile)
                if st.type == StructureType.TOWER and \
                    self.target is not None and \
                    self.target.dist_to(tile) <= 2:
                    self.target = None

    def load_next_target(self, turn_num, map, player_info):
        if self.target is not None and self.target_dist_turn_decreased + self.TARGET_GIVE_UP > turn_num:
            self.target.reached = True
            self.target = None
        if self.target is not None:
            return

        dists = np.array(tile.dist + (9e9 if tile.reached else 0) for tile in self.population_tiles)
        i = np.argmin(dists)
        self.target = self.population_tiles[i]

    def play_turn(self, turn_num, map, player_info):
        self.init_turn(turn_num, map, player_info)
        return
