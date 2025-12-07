
# Generated: 2025-08-28T05:35:56.549069
# Source Brief: brief_02673.md
# Brief Index: 2673

        
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


class Crystal:
    """A simple data class for a crystal."""
    def __init__(self, x, y, color_idx):
        self.x = x
        self.y = y
        self.color_idx = color_idx

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to push all crystals in the isometric direction. "
        "Align 5 of the same color to win before you run out of moves."
    )

    game_description = (
        "A physics-based puzzle game. Push crystals around an isometric cavern and use gravity "
        "to your advantage. Align five matching crystals to win."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 12
        self.TILE_WIDTH_ISO, self.TILE_HEIGHT_ISO = 36, 18
        self.NUM_CRYSTALS = 70
        self.NUM_COLORS = 5
        self.MAX_STEPS = 120

        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_GRID = (50, 50, 60)
        self.CRYSTAL_COLORS = [
            # Main, Top, Side
            ((255, 80, 80), (255, 150, 150), (180, 40, 40)), # Red
            ((80, 255, 80), (150, 255, 150), (40, 180, 40)), # Green
            ((80, 80, 255), (150, 150, 255), (40, 40, 180)), # Blue
            ((255, 255, 80), (255, 255, 150), (180, 180, 40)), # Yellow
            ((200, 80, 255), (220, 150, 255), (150, 50, 180)), # Purple
        ]
        self.COLOR_UI_TEXT = (220, 220, 240)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Game state variables are initialized in reset()
        self.grid = None
        self.crystals = []
        self.steps = 0
        self.steps_left = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.steps_left = self.MAX_STEPS
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.grid = [[None for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.crystals = []
        
        colors_to_place = []
        for i in range(self.NUM_COLORS):
            colors_to_place.extend([i] * (self.NUM_CRYSTALS // self.NUM_COLORS))
        while len(colors_to_place) < self.NUM_CRYSTALS:
            colors_to_place.append(self.np_random.integers(0, self.NUM_COLORS))
        
        self.np_random.shuffle(colors_to_place)

        available_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(available_coords)
        
        for i in range(self.NUM_CRYSTALS):
            x, y = available_coords[i]
            color_idx = colors_to_place[i]
            crystal = Crystal(x, y, color_idx)
            self.crystals.append(crystal)
            self.grid[y][x] = crystal
            
        self._apply_gravity()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        self.steps += 1
        self.steps_left -= 1
        
        if movement != 0:
            moved = self._apply_push(movement)
            if moved:
                # sfx: crystal_slide.wav
                self._apply_gravity()
                # sfx: crystal_thud.wav
        
        reward, terminated = self._calculate_reward_and_termination()
        
        if movement == 0:
            reward -= 0.1 # Small penalty to discourage doing nothing

        if terminated:
            self.game_over = True
            if self.win_condition_met:
                reward = 100
                # sfx: win_jingle.wav
            else: # Timeout
                reward = -100
                # sfx: lose_sound.wav
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _apply_push(self, movement):
        moves = {1: (-1, -1), 2: (1, 1), 3: (-1, 1), 4: (1, -1)} # up, down, left, right
        dx, dy = moves[movement]

        x_range = range(self.GRID_WIDTH)
        y_range = range(self.GRID_HEIGHT)
        if dx > 0: x_range = reversed(x_range)
        if dy > 0: y_range = reversed(y_range)
        
        moved_something = False
        
        for y in list(y_range):
            for x in list(x_range):
                crystal = self.grid[y][x]
                if crystal:
                    nx, ny = x + dx, y + dy
                    is_in_bounds = 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT
                    
                    if is_in_bounds and self.grid[ny][nx] is None:
                        self.grid[ny][nx] = crystal
                        self.grid[y][x] = None
                        crystal.x, crystal.y = nx, ny
                        moved_something = True
        return moved_something

    def _apply_gravity(self):
        while True:
            moved = False
            for y in range(self.GRID_HEIGHT - 2, -1, -1):
                for x in range(self.GRID_WIDTH):
                    crystal = self.grid[y][x]
                    if crystal and self.grid[y+1][x] is None:
                        self.grid[y+1][x] = crystal
                        self.grid[y][x] = None
                        crystal.y += 1
                        moved = True
            if not moved:
                break

    def _calculate_reward_and_termination(self):
        max_len = 0
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x]:
                    current_max = max(
                        self._check_line(x, y, 1, 0),
                        self._check_line(x, y, 0, 1),
                        self._check_line(x, y, 1, 1),
                        self._check_line(x, y, 1, -1),
                    )
                    if current_max > max_len:
                        max_len = current_max
        
        reward = 0
        if max_len >= 5:
            self.win_condition_met = True
            return 0, True
        elif max_len == 4:
            reward += 2.0
        elif max_len == 3:
            reward += 1.0
            
        adj_pairs = 0
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                crystal = self.grid[y][x]
                if crystal:
                    if x + 1 < self.GRID_WIDTH and self.grid[y][x+1] and self.grid[y][x+1].color_idx == crystal.color_idx:
                        adj_pairs += 1
                    if y + 1 < self.GRID_HEIGHT and self.grid[y+1][x] and self.grid[y+1][x].color_idx == crystal.color_idx:
                        adj_pairs += 1
        reward += adj_pairs * 0.1

        terminated = self.steps_left <= 0
        return reward, terminated

    def _check_line(self, x, y, dx, dy):
        crystal = self.grid[y][x]
        if not crystal: return 0
        color = crystal.color_idx
        length = 0
        for i in range(5):
            nx, ny = x + i*dx, y + i*dy
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny][nx] and self.grid[ny][nx].color_idx == color:
                length += 1
            else:
                break
        return length

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_screen(self, x, y):
        origin_x = self.WIDTH // 2
        origin_y = 70
        screen_x = origin_x + (x - y) * self.TILE_WIDTH_ISO / 2
        screen_y = origin_y + (x + y) * self.TILE_HEIGHT_ISO / 2
        return int(screen_x), int(screen_y)

    def _render_game(self):
        for y in range(self.GRID_HEIGHT + 1):
            p1 = self._grid_to_screen(0, y)
            p2 = self._grid_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for x in range(self.GRID_WIDTH + 1):
            p1 = self._grid_to_screen(x, 0)
            p2 = self._grid_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        sorted_crystals = sorted(self.crystals, key=lambda c: (c.x + c.y, c.y))
        for crystal in sorted_crystals:
            self._draw_iso_cube(crystal.x, crystal.y, crystal.color_idx)

    def _draw_iso_cube(self, x, y, color_idx):
        _, top_color, side_color = self.CRYSTAL_COLORS[color_idx]
        
        p_top_left = self._grid_to_screen(x, y)
        p_top_right = self._grid_to_screen(x + 1, y)
        p_bottom_right = self._grid_to_screen(x + 1, y + 1)
        p_bottom_left = self._grid_to_screen(x, y + 1)
        
        cube_height = self.TILE_HEIGHT_ISO * 1.5
        
        p1_b = (p_top_left[0], p_top_left[1] + cube_height)
        p3_b = (p_bottom_right[0], p_bottom_right[1] + cube_height)
        p4_b = (p_bottom_left[0], p_bottom_left[1] + cube_height)
        
        pygame.gfxdraw.filled_polygon(self.screen, [p_top_left, p_bottom_left, p4_b, p1_b], side_color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_bottom_left, p_bottom_right, p3_b, p4_b], side_color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_top_left, p_top_right, p_bottom_right, p_bottom_left], top_color)
        
        outline_color = (0, 0, 0, 50)
        pygame.gfxdraw.aapolygon(self.screen, [p_top_left, p_top_right, p_bottom_right, p_bottom_left], outline_color)
        pygame.gfxdraw.aapolygon(self.screen, [p_top_left, p_bottom_left, p4_b, p1_b], outline_color)
        pygame.gfxdraw.aapolygon(self.screen, [p_bottom_left, p_bottom_right, p3_b, p4_b], outline_color)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))

        steps_text = self.font_small.render(f"MOVES: {self.steps_left}", True, self.COLOR_UI_TEXT)
        text_rect = steps_text.get_rect(topright=(self.WIDTH - 15, 15))
        self.screen.blit(steps_text, text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "ALIGNMENT!" if self.win_condition_met else "OUT OF MOVES"
            color = (100, 255, 100) if self.win_condition_met else (255, 100, 100)
                
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")