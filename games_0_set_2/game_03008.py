import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Set the video driver to dummy to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to place a tile. Shift to rotate the current tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Place path tiles to connect all the islands before you run out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 16, 11
    NUM_ISLANDS = 5
    TILE_BUDGET = 15
    MAX_STEPS = 500

    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400

    # Visuals
    ISO_TILE_W, ISO_TILE_H = 40, 20
    COLOR_WATER = (40, 53, 91)
    COLOR_ISLAND = (67, 160, 71)
    COLOR_ISLAND_CONNECTED = (173, 255, 47)
    COLOR_TILE = (117, 117, 117)
    COLOR_PATH = (224, 224, 224)
    COLOR_CURSOR = (255, 238, 88)
    COLOR_INVALID = (239, 83, 80, 150)
    COLOR_TEXT = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Pygame setup (must be after setting the video driver)
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Tile definitions: N=0, E=1, S=2, W=3
        self.TILE_PATHS = {
            0: {0, 2},  # Straight: N-S
            1: {1, 3},  # Straight: E-W
            2: {0, 1},  # Corner: N-E
            3: {1, 2},  # Corner: E-S
            4: {2, 3},  # Corner: S-W
            5: {3, 0},  # Corner: W-N
        }
        self.DIRECTIONS = {
            0: (0, -1), # N
            1: (1, 0),  # E
            2: (0, 1),  # S
            3: (-1, 0)  # W
        }
        self.OPPOSITE_DIR = {0: 2, 1: 3, 2: 0, 3: 1}

        self.np_random = None
        self.placed_tiles = {}
        self.islands = []
        self.island_ids = []
        self.tile_deck = []
        self.cursor_pos = (0, 0)
        self.current_tile_type = None
        self.current_tile_rot = 0
        self.dsu_parent = {}
        self.dsu_size = {}
        self.main_island_component = -1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.last_action_feedback = None
        self.feedback_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.last_action_feedback = None
        self.feedback_timer = 0

        self._generate_islands()
        self._generate_tile_deck()
        
        self.placed_tiles = {}
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.current_tile_rot = 0
        self._draw_next_tile()

        self._update_all_connections()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1
        self.last_action_feedback = None

        # 1. Rotate Tile
        if shift_held and self.current_tile_type is not None:
            self.current_tile_rot = (self.current_tile_rot + 1) % 4

        # 2. Move Cursor
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1  # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1  # Right
        self.cursor_pos = (
            max(0, min(self.GRID_WIDTH - 1, self.cursor_pos[0] + dx)),
            max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + dy))
        )

        # 3. Place Tile
        if space_held and self.current_tile_type is not None:
            if self._is_valid_placement(self.cursor_pos):
                old_components = self._count_island_components()
                
                self.placed_tiles[self.cursor_pos] = (self.current_tile_type, self.current_tile_rot)
                self._update_all_connections()
                
                new_components = self._count_island_components()

                # Reward for connecting previously unconnected island groups
                if new_components < old_components:
                    reward += (old_components - new_components) * 1.0
                
                self._draw_next_tile()
            else:
                reward -= 0.1
                self.last_action_feedback = "invalid"
                self.feedback_timer = 10

        # 4. Check Termination Conditions
        terminated = False
        num_island_components = self._count_island_components()
        
        if self.islands and num_island_components == 1:
            if not self.game_over:
                reward += 10 # Bonus for the connecting move
                reward += 100 # Win bonus
                self.win_message = "YOU WIN!"
            terminated = True
            self.game_over = True

        elif self.current_tile_type is None and not self.game_over: # No tiles left
            reward -= 100 # Loss penalty
            self.win_message = "OUT OF TILES"
            terminated = True
            self.game_over = True
        
        elif self.steps >= self.MAX_STEPS:
            self.win_message = "TIME UP"
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Helper and State Methods ---

    def _pos_to_id(self, x, y):
        if (x, y) in self.islands:
            return self.GRID_WIDTH * self.GRID_HEIGHT + self.islands.index((x,y))
        return y * self.GRID_WIDTH + x

    def _init_dsu(self):
        self.dsu_parent = {}
        self.dsu_size = {}
        num_grid_cells = self.GRID_WIDTH * self.GRID_HEIGHT
        for i in range(num_grid_cells + self.NUM_ISLANDS):
            self.dsu_parent[i] = i
            self.dsu_size[i] = 1

    def _dsu_find(self, i):
        if self.dsu_parent[i] == i:
            return i
        self.dsu_parent[i] = self._dsu_find(self.dsu_parent[i])
        return self.dsu_parent[i]

    def _dsu_union(self, i, j):
        root_i = self._dsu_find(i)
        root_j = self._dsu_find(j)
        if root_i != root_j:
            if self.dsu_size[root_i] < self.dsu_size[root_j]:
                root_i, root_j = root_j, root_i
            self.dsu_parent[root_j] = root_i
            self.dsu_size[root_i] += self.dsu_size[root_j]

    def _count_island_components(self):
        if not self.islands: return 0
        return len({self._dsu_find(island_id) for island_id in self.island_ids})

    def _update_all_connections(self):
        self._init_dsu()
        for pos in self.placed_tiles:
            self._update_connections_for_tile(pos)
        
        if self.island_ids:
            roots = [self._dsu_find(i) for i in self.island_ids]
            if roots:
                try:
                    self.main_island_component = max(set(roots), key=roots.count)
                except ValueError:
                    self.main_island_component = -1
            else:
                self.main_island_component = -1
        else:
            self.main_island_component = -1

    def _update_connections_for_tile(self, pos):
        tile_id = self._pos_to_id(pos[0], pos[1])
        tile_type, tile_rot = self.placed_tiles[pos]
        
        tile_world_paths = self._get_tile_world_dirs(tile_type, tile_rot)

        for path_dir in tile_world_paths:
            dx, dy = self.DIRECTIONS[path_dir]
            neighbor_pos = (pos[0] + dx, pos[1] + dy)
            
            if not (0 <= neighbor_pos[0] < self.GRID_WIDTH and 0 <= neighbor_pos[1] < self.GRID_HEIGHT):
                continue

            neighbor_id = self._pos_to_id(neighbor_pos[0], neighbor_pos[1])
            opposite = self.OPPOSITE_DIR[path_dir]

            if neighbor_pos in self.islands:
                self._dsu_union(tile_id, neighbor_id)
            elif neighbor_pos in self.placed_tiles:
                neighbor_type, neighbor_rot = self.placed_tiles[neighbor_pos]
                neighbor_world_paths = self._get_tile_world_dirs(neighbor_type, neighbor_rot)
                if opposite in neighbor_world_paths:
                    self._dsu_union(tile_id, neighbor_id)

    def _get_tile_world_dirs(self, tile_type, tile_rot):
        return {(d + tile_rot) % 4 for d in self.TILE_PATHS[tile_type]}

    def _generate_islands(self):
        self.islands = []
        self.island_ids = []
        occupied = set()
        margin = 2
        attempts = 0
        while len(self.islands) < self.NUM_ISLANDS and attempts < 1000:
            x = self.np_random.integers(margin, self.GRID_WIDTH - margin)
            y = self.np_random.integers(margin, self.GRID_HEIGHT - margin)
            if (x, y) not in occupied:
                too_close = False
                for ix, iy in self.islands:
                    if abs(ix - x) + abs(iy - y) < 4:
                        too_close = True
                        break
                if not too_close:
                    self.islands.append((x, y))
                    occupied.add((x, y))
            attempts += 1
        self.island_ids = [self._pos_to_id(ix, iy) for ix, iy in self.islands]

    def _generate_tile_deck(self):
        self.tile_deck = list(self.np_random.integers(0, len(self.TILE_PATHS), size=self.TILE_BUDGET))

    def _draw_next_tile(self):
        if self.tile_deck:
            self.current_tile_type = self.tile_deck.pop(0)
            self.current_tile_rot = 0
        else:
            self.current_tile_type = None

    def _is_valid_placement(self, pos):
        return pos not in self.placed_tiles and pos not in self.islands

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_WATER)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _world_to_screen(self, x, y):
        screen_x = (self.SCREEN_WIDTH / 2) + (x - y) * self.ISO_TILE_W / 2
        screen_y = 50 + (x + y) * self.ISO_TILE_H / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_poly(self, surface, color, x, y, alpha=255):
        sx, sy = self._world_to_screen(x, y)
        points = [
            (sx, sy - self.ISO_TILE_H / 2),
            (sx + self.ISO_TILE_W / 2, sy),
            (sx, sy + self.ISO_TILE_H / 2),
            (sx - self.ISO_TILE_W / 2, sy),
        ]
        if alpha < 255:
            temp_surf = pygame.Surface((self.ISO_TILE_W + 4, self.ISO_TILE_H + 4), pygame.SRCALPHA)
            points_local = [(p[0] - (sx - (self.ISO_TILE_W/2 + 2)), p[1] - (sy - (self.ISO_TILE_H/2 + 2))) for p in points]
            pygame.draw.polygon(temp_surf, (*color, alpha), points_local)
            surface.blit(temp_surf, (sx - (self.ISO_TILE_W/2 + 2), sy - (self.ISO_TILE_H/2 + 2)))
        else:
            pygame.gfxdraw.filled_polygon(surface, points, color)
            pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_tile_paths(self, surface, pos, tile_type, rotation, color, thickness=3):
        sx, sy = self._world_to_screen(pos[0], pos[1])
        world_paths = self._get_tile_world_dirs(tile_type, rotation)
        
        center = (sx, sy)
        offsets = {
            0: (0, -self.ISO_TILE_H / 4), # N
            1: (self.ISO_TILE_W / 4, 0),  # E
            2: (0, self.ISO_TILE_H / 4),  # S
            3: (-self.ISO_TILE_W / 4, 0)  # W
        }

        for direction in world_paths:
            end_point = (center[0] + offsets[direction][0], center[1] + offsets[direction][1])
            pygame.draw.line(surface, color, center, end_point, thickness)

    def _render_game(self):
        # Draw water animation
        time_factor = self.steps * 0.05
        for i in range(10):
            y_offset = math.sin(time_factor + i * 0.5) * 2
            pygame.draw.line(self.screen, (50, 63, 101), (0, i * 50 + y_offset), (self.SCREEN_WIDTH, i * 50 + y_offset), 20)

        # Draw grid entities from back to front
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                pos = (x, y)
                if pos in self.islands:
                    island_id = self._pos_to_id(x, y)
                    is_connected = self.main_island_component != -1 and self._dsu_find(island_id) == self.main_island_component
                    color = self.COLOR_ISLAND_CONNECTED if is_connected else self.COLOR_ISLAND
                    if is_connected: # Glow effect
                        self._draw_iso_poly(self.screen, self.COLOR_ISLAND_CONNECTED, x, y, alpha=100)
                    self._draw_iso_poly(self.screen, color, x, y)
                elif pos in self.placed_tiles:
                    tile_type, tile_rot = self.placed_tiles[pos]
                    self._draw_iso_poly(self.screen, self.COLOR_TILE, x, y)
                    self._draw_tile_paths(self.screen, pos, tile_type, tile_rot, self.COLOR_PATH)

        # Draw cursor and tile preview
        if self.current_tile_type is not None and not self.game_over:
            cx, cy = self.cursor_pos
            is_valid = self._is_valid_placement(self.cursor_pos)
            
            preview_color = (200, 200, 200)
            self._draw_iso_poly(self.screen, preview_color, cx, cy, alpha=150)
            self._draw_tile_paths(self.screen, self.cursor_pos, self.current_tile_type, self.current_tile_rot, self.COLOR_PATH, thickness=2)
            
            sx, sy = self._world_to_screen(cx, cy)
            points = [
                (sx, sy - self.ISO_TILE_H / 2), (sx + self.ISO_TILE_W / 2, sy),
                (sx, sy + self.ISO_TILE_H / 2), (sx - self.ISO_TILE_W / 2, sy),
            ]
            pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 2)
        
        if self.feedback_timer > 0:
            if self.last_action_feedback == "invalid":
                self._draw_iso_poly(self.screen, self.COLOR_INVALID[:3], self.cursor_pos[0], self.cursor_pos[1], alpha=self.COLOR_INVALID[3])
            self.feedback_timer -= 1


    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        tiles_left = len(self.tile_deck) + (1 if self.current_tile_type is not None else 0)
        tiles_text = self.font_small.render(f"TILES: {tiles_left}/{self.TILE_BUDGET}", True, self.COLOR_TEXT)
        self.screen.blit(tiles_text, (10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tiles_remaining": len(self.tile_deck) + (1 if self.current_tile_type is not None else 0),
            "island_components": self._count_island_components(),
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # --- Agent Interaction Test ---
    print("\n" + "="*30)
    print("AGENT INTERACTION TEST")
    print("="*30 + "\n")
    
    env = GameEnv()
    try:
        obs, info = env.reset(seed=42)
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            if terminated or truncated:
                done = True
        print(f"Random Agent Finished in {step_count} steps. Total Reward: {total_reward:.2f}")
        print(f"Final Info: {info}")
    finally:
        env.close()

    # --- Manual Play ---
    # This part requires a display. It will not run in a headless environment.
    # To run, comment out the `os.environ` line at the top of the file.
    if os.getenv("SDL_VIDEODRIVER") != "dummy":
        try:
            pygame.display.init()
            pygame.font.init()
            screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
            pygame.display.set_caption("Island Connector")
            clock = pygame.time.Clock()
            
            env = GameEnv()
            obs, info = env.reset()
            done = False
            
            print("\n" + "="*30)
            print("MANUAL PLAY")
            print(env.user_guide)
            print("="*30 + "\n")

            while not done:
                movement = 0 # no-op
                space = 0
                shift = 0

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP: movement = 1
                        elif event.key == pygame.K_DOWN: movement = 2
                        elif event.key == pygame.K_LEFT: movement = 3
                        elif event.key == pygame.K_RIGHT: movement = 4
                        elif event.key == pygame.K_SPACE: space = 1
                        elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 1
                        elif event.key == pygame.K_r: # Reset
                            obs, info = env.reset()
                            done = False
                            continue

                action = [movement, space, shift]
                
                if any(action):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")

                screen.blit(env.screen, (0, 0))
                pygame.display.flip()
                
                clock.tick(30)

        except pygame.error as e:
            print(f"Skipping manual play because no display is available: {e}")
        finally:
            pygame.quit()