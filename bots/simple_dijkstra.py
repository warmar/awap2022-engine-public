import math
import heapq
import numpy as np

from src.game import *
from src.player import *
from src.structure import *
from src.game_constants import GameConstants as GC

ROAD_COST = 10
TOWER_COST = 250

class AugmentedTile(Tile):
    def __init__(self, tile, closest_tile):
        Tile.__init__(self, tile.x ,tile.y, tile.passability, tile.population, tile.structure)
        self.closest_tile = closest_tile
        self.dist = math.hypot(tile.x - closest_tile.x, tile.y - closest_tile.y)
        self.reached = False

    def update_closest_tile(self, tile):
        dist = math.hypot(tile.x - self.closest_tile.x, tile.y - self.closest_tile.y)
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
        self.ups = None # uncovered populations

        self.target = None
        self.last_target_dist = 9e9
        self.target_dist_turn_decreased = 0
        self.TARGET_GIVE_UP = 10
        return

    def is_valid_pos(self, map, pos):
        if pos[0] < 0:
            return False
        if pos[1] < 0:
            return False
        if pos[0] >= len(map):
            return False
        if pos[1] >= len(map[0]):
            return False
        return True

    def neighbors(self, map, pos):
        x, y = pos
        result = [neigh for neigh in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)] if self.is_valid_pos(map, neigh)]
        return result

    def find_lowest_cost_road(self, map, team, start, end):
        self.MAP_WIDTH = len(map)
        self.MAP_HEIGHT = len(map[0])
        distances = np.full((self.MAP_WIDTH, self.MAP_HEIGHT), np.Inf)
        previous = np.full((self.MAP_WIDTH, self.MAP_HEIGHT), None)

        distances[start[0]][start[1]] = 0

        q = []
        heapq.heappush(q, (0, start))
        while len(q) > 0:
            curr = heapq.heappop(q)

            curr_distance = curr[0]
            curr_pos = curr[1]
            # Termination Case
            
            if curr_pos == end:
                # Compute path
                path = [curr_pos]
                while True:
                    if path[-1] == start:
                        return path, distances[curr_pos[0]][curr_pos[1]]
                    path.append(previous[path[-1][0]][path[-1][1]])

            for neigh in self.neighbors(map, curr_pos):
                # todo check if existing structure is there
                cost = ROAD_COST * map[neigh[0]][neigh[1]].passability
                struct = map[neigh[0]][neigh[1]].structure
                if struct is not None:
                    if struct.team == team:
                        cost = 0
                    else:
                        cost = np.Inf

                alt = curr_distance + cost
                if alt < distances[neigh[0]][neigh[1]]:
                    previous[neigh[0]][neigh[1]] = curr_pos
                    distances[neigh[0]][neigh[1]] = alt
                    heapq.heappush(q, (alt, neigh))
    
        return None
    
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

    def pos_is_covered(self, map, team, pos):
        for other in self.coverage_positions(map, pos):
            struct = map[other[0]][other[1]].structure
            if struct is None:
                continue

            if struct.type != StructureType.TOWER:
                continue

            if struct.team != team:
                continue

            return True

        return False

    def uncovered_populations(self, map, team):
        result = []
        for i in range(0, len(map)):
            for j in range(0, len(map[0])):
                tile = map[i][j]
                if tile.population == 0:
                    continue
                if self.pos_is_covered(map, team, (i,j)):
                    continue
                result.append((i,j))
        return result

    def coverage_positions(self, map, pos):
        result = []
        for offset_i in range(-2,3):
            for offset_j in range(-2,3):
                other = (pos[0] + offset_i, pos[1] + offset_j)
                if not self.is_valid_pos(map, other):
                    continue
                if abs(offset_i) + abs(offset_j) > 2:
                    continue
                result.append(other)
        return result

    def best_tower_location_for_up(self, map, team, pos):
        best_pos = None
        best_cost = None
        for other in self.coverage_positions(map, pos):
            cost = map[other[0]][other[1]].passability
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pos = other
        return best_pos, best_cost

    def play_turn(self, turn_num, map, player_info):
        self.init_turn(turn_num, map, player_info)

        team = player_info.team
        
        generators = []
        for i in range(0, len(map)):
            for j in range(0, len(map[0])):
                tile = map[i][j]
                struct = tile.structure
                if struct is None:
                    continue
                if struct.type != StructureType.GENERATOR:
                    continue
                if struct.team != team:
                    continue
                generators.append((i, j))

        if self.ups is None:
            self.ups = self.uncovered_populations(map, team)
        else:
            self.ups = [up for up in self.ups if not self.pos_is_covered(map, team, up)]

        # find minimum cost path to a tile which covers a population
        min_cost_path = None
        min_cost = None
        for up in self.ups:
            best_pos, best_cost = self.best_tower_location_for_up(map, team, up)
            for generator in generators:
                res = self.find_lowest_cost_road(map, team, generator, best_pos)
                if res is None:
                    continue
                path, cost = res
                cost += TOWER_COST*best_cost
                if min_cost is None or cost < min_cost:
                    min_cost = cost
                    min_cost_path = path

        if min_cost_path is None:
            return

        if player_info.money < min_cost:
            return

        end = min_cost_path[0]
        start = min_cost_path[-1]
        for intermediate in reversed(min_cost_path[1:-1]):
            self.build(StructureType.ROAD, intermediate[0], intermediate[1])
        self.build(StructureType.TOWER, end[0], end[1])

        return
