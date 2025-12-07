
# Generated: 2025-08-28T06:44:50.778617
# Source Brief: brief_03025.md
# Brief Index: 3025

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper data classes to organize state
class Crystal:
    """Represents a movable crystal in the game."""
    def __init__(self, x, y, color, network_id):
        self.pos = (x, y)
        self.last_pos = (x, y) # For rendering trails
        self.color = color
        self.network_id = network_id
        self.is_illuminated = False

class PressurePlate:
    """Represents a static pressure plate on the grid."""
    def __init__(self, x, y, network_id):
        self.pos = (x, y)
        self.network_id = network_id
        self.is_active = False

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to slide all crystals in a direction. "
        "The goal is to move crystals onto pressure plates to light up all connected crystals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Slide crystals onto pressure plates to "
        "illuminate all of them within the 50-move limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame and Display ---
        self.W, self.H = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 36, bold=True)

        # --- Visual Style ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_WALL = (100, 110, 120)
        self.COLOR_PLATE_INACTIVE = (70, 80, 90)
        self.COLOR_PLATE_ACTIVE = (0, 255, 255)
        self.CRYSTAL_COLORS = [
            (255, 0, 128),   # Pink
            (0, 255, 0),     # Green
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
            (255, 255, 0),   # Yellow
        ]
        self.COLOR_PATH = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)

        # --- Game Grid and Isometric Projection ---
        self.grid_size = 12
        self.tile_width = 32
        self.tile_height = self.tile_width // 2
        self.origin_x = self.W // 2
        self.origin_y = 80

        # --- Game State (initialized in reset) ---
        self.crystals = []
        self.plates = []
        self.walls = set()
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.illuminated_count = 0
        self.last_illuminated_count = 0
        self.total_crystals = 0

        self.reset()
        self.validate_implementation()

    def _define_level(self):
        """Hardcoded solvable level design. Guarantees a consistent puzzle."""
        self.grid_size = 12
        
        walls = set()
        for i in range(self.grid_size):
            walls.add((i, -1)); walls.add((-1, i))
            walls.add((i, self.grid_size)); walls.add((self.grid_size, i))
        
        internal_walls = [
            (3,3), (3,4), (3,5), (8,6), (8,7), (8,8), (1, 8), 
            (2, 8), (9, 1), (9, 2), (5,0), (6,0), (5,11), (6,11)
        ]
        walls.update(internal_walls)
        self.walls = walls

        plate_defs = {
            0: (5, 2), 1: (9, 9), 2: (2, 6), 3: (6, 5), 4: (5, 9),
        }

        crystal_defs = [
            Crystal(1, 1, self.CRYSTAL_COLORS[0], 0), Crystal(1, 4, self.CRYSTAL_COLORS[0], 0),
            Crystal(4, 1, self.CRYSTAL_COLORS[0], 0), Crystal(10, 4, self.CRYSTAL_COLORS[0], 0),
            Crystal(10, 10, self.CRYSTAL_COLORS[1], 1), Crystal(7, 10, self.CRYSTAL_COLORS[1], 1),
            Crystal(10, 7, self.CRYSTAL_COLORS[1], 1), Crystal(4, 7, self.CRYSTAL_COLORS[1], 1),
            Crystal(1, 5, self.CRYSTAL_COLORS[2], 2), Crystal(4, 4, self.CRYSTAL_COLORS[2], 2),
            Crystal(1, 7, self.CRYSTAL_COLORS[2], 2), Crystal(4, 8, self.CRYSTAL_COLORS[2], 2),
            Crystal(5, 5, self.CRYSTAL_COLORS[3], 3), Crystal(7, 4, self.CRYSTAL_COLORS[3], 3),
            Crystal(5, 7, self.CRYSTAL_COLORS[3], 3), Crystal(7, 7, self.CRYSTAL_COLORS[3], 3),
            Crystal(2, 10, self.CRYSTAL_COLORS[4], 4), Crystal(5, 10, self.CRYSTAL_COLORS[4], 4),
            Crystal(8, 10, self.CRYSTAL_COLORS[4], 4), Crystal(10, 1, self.CRYSTAL_COLORS[4], 4),
        ]
        self.total_crystals = len(crystal_defs)
        
        self.plates = [PressurePlate(pos[0], pos[1], net_id) for net_id, pos in plate_defs.items()]
        self.crystals = crystal_defs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._define_level()
        
        self.moves_remaining = 50
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self._update_plate_and_crystal_states()
        self.last_illuminated_count = self.illuminated_count
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # Reset last positions for trails
        for c in self.crystals: c.last_pos = c.pos

        if 1 <= movement <= 4:
            self.moves_remaining -= 1
            reward -= 0.02
            # sfx_push_crystal()
            
            moved_something = self._push_crystals(movement)
            
            if moved_something:
                self._update_plate_and_crystal_states()

        newly_illuminated = self.illuminated_count - self.last_illuminated_count
        if newly_illuminated > 0:
            reward += newly_illuminated * 5.0
            # sfx_illuminate_new()
        self.last_illuminated_count = self.illuminated_count

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.illuminated_count == self.total_crystals:
                reward += 100  # Win
                # sfx_win_level()
            else:
                reward -= 100  # Loss
                # sfx_lose_level()
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _push_crystals(self, direction_id):
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[direction_id]

        x_range = range(self.grid_size - 1, -1, -1) if dx > 0 else range(self.grid_size)
        y_range = range(self.grid_size - 1, -1, -1) if dy > 0 else range(self.grid_size)

        moved_any = False
        crystal_map = {c.pos: c for c in self.crystals}

        for y_start in y_range:
            for x_start in x_range:
                if (x_start, y_start) in crystal_map:
                    crystal = crystal_map[(x_start, y_start)]
                    
                    cx, cy = crystal.pos
                    nx, ny = cx + dx, cy + dy
                    while (nx, ny) not in self.walls and (nx, ny) not in crystal_map:
                        cx, cy = nx, ny
                        nx, ny = cx + dx, cy + dy
                    
                    if (cx, cy) != crystal.pos:
                        del crystal_map[crystal.pos]
                        crystal.pos = (cx, cy)
                        crystal_map[crystal.pos] = crystal
                        moved_any = True
        return moved_any

    def _update_plate_and_crystal_states(self):
        crystal_positions = {c.pos for c in self.crystals}
        
        for plate in self.plates:
            plate.is_active = plate.pos in crystal_positions
        
        active_networks = {p.network_id for p in self.plates if p.is_active}
        
        current_illuminated_count = 0
        for crystal in self.crystals:
            is_now_illuminated = crystal.network_id in active_networks
            crystal.is_illuminated = is_now_illuminated
            if is_now_illuminated:
                current_illuminated_count += 1
        
        self.illuminated_count = current_illuminated_count

    def _check_termination(self):
        if self.illuminated_count >= self.total_crystals or self.moves_remaining <= 0:
            self.game_over = True
            return True
        return False

    def _grid_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * self.tile_width / 2
        iso_y = self.origin_y + (x + y) * self.tile_height / 2
        return int(iso_x), int(iso_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid tiles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                points = [self._grid_to_iso(x,y), self._grid_to_iso(x+1,y), self._grid_to_iso(x+1,y+1), self._grid_to_iso(x,y+1)]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRID)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BG)

        # Render plates
        for plate in self.plates:
            iso_x, iso_y = self._grid_to_iso(plate.pos[0] + 0.5, plate.pos[1] + 0.5)
            color = self.COLOR_PLATE_ACTIVE if plate.is_active else self.COLOR_PLATE_INACTIVE
            radius = int(self.tile_width / 3)
            pygame.gfxdraw.filled_circle(self.screen, iso_x, iso_y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, iso_x, iso_y, radius, self.COLOR_BG)

        # Render illuminated paths
        for plate in [p for p in self.plates if p.is_active]:
            plate_iso = self._grid_to_iso(plate.pos[0] + 0.5, plate.pos[1] + 0.5)
            for crystal in [c for c in self.crystals if c.network_id == plate.network_id]:
                crystal_iso = self._grid_to_iso(crystal.pos[0] + 0.5, crystal.pos[1] + 0.5)
                pygame.draw.aaline(self.screen, self.COLOR_PATH, plate_iso, crystal_iso, blend=1)
        
        # Sort and render entities for correct isometric occlusion
        entities = [{'type': 'wall', 'pos': pos} for pos in self.walls if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size]
        entities.extend([{'type': 'crystal', 'obj': c} for c in self.crystals])
        
        entities.sort(key=lambda e: (e['pos'][0] + e['pos'][1], e['pos'][1]) if e['type'] == 'wall' else (e['obj'].pos[0] + e['obj'].pos[1], e['obj'].pos[1]))
        
        for entity in entities:
            if entity['type'] == 'wall':
                self._render_wall_segment(entity['pos'])
            else:
                c = entity['obj']
                self._render_crystal(c.pos, c.last_pos, c.color, c.is_illuminated)

    def _render_crystal(self, pos, last_pos, color, is_illuminated):
        if pos != last_pos:
            start_iso = self._grid_to_iso(last_pos[0] + 0.5, last_pos[1] + 0.5)
            end_iso = self._grid_to_iso(pos[0] + 0.5, pos[1] + 0.5)
            pygame.draw.line(self.screen, (*color, 100), start_iso, end_iso, 5)

        x, y = pos
        base_x, base_y = self._grid_to_iso(x, y)
        top_offset, side_height = self.tile_height, self.tile_height * 1.5
        
        top = [(base_x, base_y - top_offset), (base_x + self.tile_width / 2, base_y - top_offset + self.tile_height / 2), (base_x, base_y - top_offset + self.tile_height), (base_x - self.tile_width / 2, base_y - top_offset + self.tile_height / 2)]
        left = [(top[3]), (top[2]), (top[2][0], top[2][1] + side_height), (top[3][0], top[3][1] + side_height)]
        right = [(top[1]), (top[2]), (top[2][0], top[2][1] + side_height), (top[1][0], top[1][1] + side_height)]

        dark_color = tuple(max(0, c - 60) for c in color)
        light_color = tuple(min(255, c + 40) for c in color)
        
        if is_illuminated:
            pygame.gfxdraw.filled_polygon(self.screen, top, (*color, 50))
            pygame.gfxdraw.filled_polygon(self.screen, left, (*color, 50))
            pygame.gfxdraw.filled_polygon(self.screen, right, (*color, 50))

        pygame.gfxdraw.filled_polygon(self.screen, left, dark_color)
        pygame.gfxdraw.aapolygon(self.screen, left, light_color)
        pygame.gfxdraw.filled_polygon(self.screen, right, color)
        pygame.gfxdraw.aapolygon(self.screen, right, light_color)
        pygame.gfxdraw.filled_polygon(self.screen, top, light_color)
        pygame.gfxdraw.aapolygon(self.screen, top, light_color)

    def _render_wall_segment(self, pos):
        x, y = pos
        base_x, base_y = self._grid_to_iso(x, y)
        top_offset, side_height = self.tile_height, self.tile_height * 2

        top = [(base_x, base_y - top_offset), (base_x + self.tile_width / 2, base_y - top_offset + self.tile_height / 2), (base_x, base_y - top_offset + self.tile_height), (base_x - self.tile_width / 2, base_y - top_offset + self.tile_height / 2)]
        left = [(top[3]), (top[2]), (top[2][0], top[2][1] + side_height), (top[3][0], top[3][1] + side_height)]
        right = [(top[1]), (top[2]), (top[2][0], top[2][1] + side_height), (top[1][0], top[1][1] + side_height)]
        
        dark = tuple(max(0, c - 40) for c in self.COLOR_WALL)
        light = tuple(min(255, c + 20) for c in self.COLOR_WALL)

        pygame.gfxdraw.filled_polygon(self.screen, left, dark); pygame.gfxdraw.aapolygon(self.screen, left, light)
        pygame.gfxdraw.filled_polygon(self.screen, right, self.COLOR_WALL); pygame.gfxdraw.aapolygon(self.screen, right, light)
        pygame.gfxdraw.filled_polygon(self.screen, top, light); pygame.gfxdraw.aapolygon(self.screen, top, light)

    def _render_ui(self):
        moves_surf = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (10, 10))
        
        crystals_surf = self.font_ui.render(f"Illuminated: {self.illuminated_count}/{self.total_crystals}", True, self.COLOR_TEXT)
        self.screen.blit(crystals_surf, (self.W - crystals_surf.get_width() - 10, 10))

        if self.game_over:
            msg, color = ("LEVEL COMPLETE!", self.COLOR_PLATE_ACTIVE) if self.illuminated_count >= self.total_crystals else ("OUT OF MOVES", self.CRYSTAL_COLORS[0])
            msg_surf = self.font_msg.render(msg, True, color)
            bg_rect = msg_surf.get_rect(center=(self.W/2, self.H/2)).inflate(20, 10)
            pygame.draw.rect(self.screen, (*self.COLOR_BG, 200), bg_rect, border_radius=10)
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(self.W/2, self.H/2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3) and isinstance(info, dict)
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert obs.shape == (400, 640, 3) and isinstance(reward, (int, float))
        assert isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Isometric Crystal Puzzler")
    clock = pygame.time.Clock()
    running = True
    while running:
        action = [0, 0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_r: obs, info = env.reset()
                elif event.key in [pygame.K_q, pygame.K_ESCAPE]: running = False
        
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
            if terminated: print("Game Over! Press 'R' to restart or 'Q' to quit.")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)
    env.close()