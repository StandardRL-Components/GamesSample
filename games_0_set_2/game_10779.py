import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:03:11.460796
# Source Brief: brief_00779.md
# Brief Index: 779
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A hexagonal tile-laying puzzle game as a Gymnasium environment.

    The player places colored hexagonal tiles on a grid to maximize the score
    from adjacent color matches.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) to move the cursor.
    - actions[1]: Space button (0=released, 1=held). A press places a tile.
    - actions[2]: Shift button (0=released, 1=held). A press cycles the selected tile.

    Observation Space: Box(shape=(400, 640, 3), dtype=uint8)
    - An RGB image of the game state.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Place colored hexagonal tiles on a grid to create color matches with adjacent tiles and maximize your score."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a tile and shift to cycle through your available tiles."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    FPS = 30

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_LINES = (50, 60, 70)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_MATCH_FLASH = (255, 255, 255)
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]

    # Hex Grid Parameters
    HEX_RADIUS = 20
    HEX_GRID_RADIUS = 7

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # Hex grid calculations
        self.hex_width = self.HEX_RADIUS * 2
        self.hex_height = math.sqrt(3) * self.HEX_RADIUS
        self._init_hex_grid()

        # Initialize state variables
        self.grid = {}
        self.cursor_pos = (0, 0)
        self.available_tiles = []
        self.selected_tile_idx = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.placement_feedback = []

    def _init_hex_grid(self):
        """Creates a set of valid (q, r) axial coordinates for the grid."""
        self.valid_coords = set()
        for q in range(-self.HEX_GRID_RADIUS, self.HEX_GRID_RADIUS + 1):
            r_min = max(-self.HEX_GRID_RADIUS, -q - self.HEX_GRID_RADIUS)
            r_max = min(self.HEX_GRID_RADIUS, -q + self.HEX_GRID_RADIUS)
            for r in range(r_min, r_max + 1):
                self.valid_coords.add((q, r))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = {}
        self.cursor_pos = (0, 0)
        self._replenish_tiles()
        self.selected_tile_idx = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.placement_feedback = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Handle Input ---
        # 1. Cycle selected tile on Shift press
        if shift_held and not self.prev_shift_held and self.available_tiles:
            self.selected_tile_idx = (self.selected_tile_idx + 1) % len(self.available_tiles)
            # sfx: cycle_tile

        # 2. Move cursor
        if movement != 0:
            self._move_cursor(movement)

        # 3. Place tile on Space press
        if space_held and not self.prev_space_held:
            placement_reward = self._place_tile()
            if placement_reward is not None:
                reward = placement_reward
                self.score += placement_reward
                # sfx: place_tile_success
            else:
                # sfx: place_tile_fail
                pass

        # --- Update Game State ---
        self._update_feedback_timers()
        self.steps += 1
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Check Termination ---
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
        
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _move_cursor(self, movement_action):
        q, r = self.cursor_pos
        # Axial directions: up/down/left/right are mapped to hex directions
        if movement_action == 1:   # Up -> Up-Right
            new_pos = (q + 1, r - 1)
        elif movement_action == 2: # Down -> Down-Left
            new_pos = (q - 1, r + 1)
        elif movement_action == 3: # Left
            new_pos = (q - 1, r)
        elif movement_action == 4: # Right
            new_pos = (q + 1, r)
        else:
            return

        if new_pos in self.valid_coords:
            self.cursor_pos = new_pos
            # sfx: cursor_move

    def _replenish_tiles(self):
        self.available_tiles = [self.np_random.integers(0, len(self.TILE_COLORS)) for _ in range(3)]
        self.selected_tile_idx = 0

    def _place_tile(self):
        if not self.available_tiles or self.cursor_pos in self.grid:
            return None

        tile_color_idx = self.available_tiles.pop(self.selected_tile_idx)
        self.grid[self.cursor_pos] = tile_color_idx
        
        matches = 0
        matching_neighbors = []
        
        for neighbor_pos in self._get_neighbors(self.cursor_pos):
            if self.grid.get(neighbor_pos) == tile_color_idx:
                matches += 1
                matching_neighbors.append(neighbor_pos)
        
        reward = float(matches)
        if matches >= 3:
            reward += 5.0

        self.placement_feedback.append({'pos': self.cursor_pos, 'timer': 10})
        for neighbor_pos in matching_neighbors:
            self.placement_feedback.append({'pos': neighbor_pos, 'timer': 10})

        if not self.available_tiles:
            self._replenish_tiles()
        
        if self.available_tiles:
            self.selected_tile_idx = min(self.selected_tile_idx, len(self.available_tiles) - 1)

        return reward

    def _update_feedback_timers(self):
        self.placement_feedback = [f for f in self.placement_feedback if f['timer'] > 0]
        for f in self.placement_feedback:
            f['timer'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for q, r in self.valid_coords:
            center_px = self._axial_to_pixel(q, r)
            self._draw_hexagon(self.screen, self.COLOR_GRID_LINES, center_px, self.HEX_RADIUS, width=1)
        
        for pos, color_idx in self.grid.items():
            center_px = self._axial_to_pixel(pos[0], pos[1])
            self._draw_hexagon(self.screen, self.TILE_COLORS[color_idx], center_px, self.HEX_RADIUS)
            
        for feedback in self.placement_feedback:
            alpha = int(255 * (feedback['timer'] / 10.0)**2)
            color = (*self.COLOR_MATCH_FLASH, alpha)
            center_px = self._axial_to_pixel(feedback['pos'][0], feedback['pos'][1])
            self._draw_hexagon(self.screen, color, center_px, self.HEX_RADIUS + 2, width=3, use_alpha=True)

        cursor_px = self._axial_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
        cursor_width = 2 + int(pulse * 2)
        self._draw_hexagon(self.screen, self.COLOR_CURSOR, cursor_px, self.HEX_RADIUS, width=cursor_width)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 10))

        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 20, 40))

        preview_label = self.font_small.render("TILES:", True, self.COLOR_TEXT)
        self.screen.blit(preview_label, (20, self.SCREEN_HEIGHT - 60))
        
        start_x = 60
        start_y = self.SCREEN_HEIGHT - 35
        for i, color_idx in enumerate(self.available_tiles):
            center_px = (start_x + i * (self.hex_width * 0.9), start_y)
            radius = self.HEX_RADIUS * 0.8
            self._draw_hexagon(self.screen, self.TILE_COLORS[color_idx], center_px, radius)
            if i == self.selected_tile_idx:
                self._draw_hexagon(self.screen, self.COLOR_CURSOR, center_px, radius, width=3)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _axial_to_pixel(self, q, r):
        x = self.HEX_RADIUS * (3/2 * q)
        y = self.HEX_RADIUS * (math.sqrt(3)/2 * q + math.sqrt(3) * r)
        return int(x + self.SCREEN_WIDTH / 2), int(y + self.SCREEN_HEIGHT / 2)

    def _get_neighbors(self, pos):
        q, r = pos
        axial_directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        return [(q + dq, r + dr) for dq, dr in axial_directions if (q + dq, r + dr) in self.valid_coords]

    def _draw_hexagon(self, surface, color, center, size, width=0, use_alpha=False):
        points = []
        for i in range(6):
            angle_deg = 60 * i - 30
            angle_rad = math.pi / 180 * angle_deg
            points.append((center[0] + size * math.cos(angle_rad),
                           center[1] + size * math.sin(angle_rad)))
        
        int_points = [(int(p[0]), int(p[1])) for p in points]
        
        if use_alpha and len(color) == 4:
            temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(temp_surf, color, int_points, width)
            surface.blit(temp_surf, (0, 0))
        else:
            if width == 0:
                pygame.gfxdraw.aapolygon(surface, int_points, color)
                pygame.gfxdraw.filled_polygon(surface, int_points, color)
            else:
                pygame.draw.aalines(surface, color, True, points, 1)
                if width > 1:
                    pygame.draw.lines(surface, color, True, int_points, width)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the environment directly for testing.
    # It will create a window and let you control the game manually.
    
    # Un-dummy the video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("HexaGrid Strategy - Manual Control")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    key_map = { pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4 }
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("----------------------\n")

    while running:
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # For turn-based games, we can advance on key presses
            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                for key_code, move_action in key_map.items():
                    if keys[key_code]:
                        movement = move_action
                        break
                
                if keys[pygame.K_SPACE]: space_held = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
                    
                action = [movement, space_held, shift_held]
                
                if not terminated:
                    obs, reward, terminated, truncated, info = env.step(action)
                    if reward > 0:
                        print(f"Step: {info['steps']}, Reward: {reward:.1f}, New Score: {info['score']:.1f}")
                    if terminated:
                        print(f"\n--- GAME OVER ---")
                        print(f"Final Score: {info['score']:.1f} in {info['steps']} steps.")
        
        # Since auto_advance is False, we only render after a step
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS to avoid busy-waiting

    env.close()