import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press space to place a crystal. Hold shift to reset the cursor."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Redirect lasers to their targets by strategically placing crystals in an isometric 2D cavern."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Screen dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game constants
        self.GRID_WIDTH = 15
        self.GRID_HEIGHT = 12
        self.TILE_WIDTH_HALF = 20
        self.TILE_HEIGHT_HALF = 10
        self.ISO_ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ISO_ORIGIN_Y = 80
        self.MAX_CRYSTALS = 20
        self.MAX_STEPS = 1000
        self.MAX_LASER_LENGTH = 50

        # Grid element types
        self.EMPTY = 0
        self.WALL = 1
        self.CRYSTAL = 2
        self.EMITTER_R, self.TARGET_R = 10, 11
        self.EMITTER_G, self.TARGET_G = 20, 21
        self.EMITTER_B, self.TARGET_B = 30, 31
        self.EMITTER_Y, self.TARGET_Y = 40, 41
        self.EMITTER_P, self.TARGET_P = 50, 51

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (30, 35, 60)
        self.COLOR_WALL = (60, 70, 100)
        self.COLOR_CRYSTAL = (220, 255, 255)
        self.COLOR_CRYSTAL_GLOW = (150, 220, 255)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)

        self.LASER_COLORS = {
            self.EMITTER_R: (255, 50, 50), self.TARGET_R: (255, 50, 50),
            self.EMITTER_G: (50, 255, 50), self.TARGET_G: (50, 255, 50),
            self.EMITTER_B: (80, 80, 255), self.TARGET_B: (80, 80, 255),
            self.EMITTER_Y: (255, 255, 50), self.TARGET_Y: (255, 255, 50),
            self.EMITTER_P: (255, 50, 255), self.TARGET_P: (255, 50, 255),
        }

        # Initialize state variables
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.emitters = []
        self.targets = {}
        self.laser_paths = []
        self.cursor_pos = [0, 0]
        self.crystals_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.previous_targets_hit = set()
        self.current_targets_hit = set()

        # self.reset() is called in the official API, not in __init__
        # self.validate_implementation() is for debugging, not part of final env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.crystals_remaining = self.MAX_CRYSTALS
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]

        self._initialize_level()
        self.previous_targets_hit = set()
        self._calculate_all_laser_paths()

        return self._get_observation(), self._get_info()

    def _initialize_level(self):
        self.grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), self.EMPTY, dtype=int)

        # Define emitters (pos, direction, type)
        self.emitters = [
            ([0, 2], (1, 0), self.EMITTER_R),
            ([self.GRID_WIDTH - 1, 4], (-1, 0), self.EMITTER_G),
            ([3, 0], (0, 1), self.EMITTER_B),
            ([self.GRID_WIDTH - 4, self.GRID_HEIGHT - 1], (0, -1), self.EMITTER_Y),
            ([7, 0], (0, 1), self.EMITTER_P),
        ]

        # Define targets (pos, type)
        raw_targets = [
            ([self.GRID_WIDTH - 1, 8], self.TARGET_R),
            ([0, 10], self.TARGET_G),
            ([10, self.GRID_HEIGHT - 1], self.TARGET_B),
            ([2, 5], self.TARGET_Y),
            ([12, 3], self.TARGET_P),
        ]

        self.targets = {}
        # FIX: Unpack 3 values (pos, direction, type) from self.emitters, ignoring direction.
        for pos, _, type in self.emitters:
            self.grid[pos[0], pos[1]] = type
        for pos, type in raw_targets:
            self.grid[pos[0], pos[1]] = type
            self.targets[type] = tuple(pos)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        crystal_placed = self._handle_actions(movement, space_held, shift_held)

        if crystal_placed:
            self._calculate_all_laser_paths()

        reward = self._calculate_reward()
        self.score += reward

        terminated = self._check_termination()

        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, movement, space_held, shift_held):
        # Shift to reset cursor
        if shift_held:
            self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]

        # Movement
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # Place crystal
        if space_held and self.crystals_remaining > 0:
            x, y = self.cursor_pos
            if self.grid[x, y] == self.EMPTY:
                self.grid[x, y] = self.CRYSTAL
                self.crystals_remaining -= 1
                return True
        return False

    def _calculate_reward(self):
        reward = 0

        newly_hit = self.current_targets_hit - self.previous_targets_hit
        reward += len(newly_hit) * 5.0

        reward += len(self.current_targets_hit) * 0.1

        self.previous_targets_hit = self.current_targets_hit.copy()

        if len(self.current_targets_hit) == len(self.targets):
            reward += 100.0
            self.win = True
        elif self.crystals_remaining <= 0 and not self.win:
            reward -= 100.0

        return reward

    def _check_termination(self):
        if self.win:
            return True
        if self.crystals_remaining <= 0 and len(self.current_targets_hit) < len(self.targets):
            return True
        return False

    def _calculate_all_laser_paths(self):
        self.laser_paths = []
        self.current_targets_hit = set()
        for pos, direction, type in self.emitters:
            self._trace_laser(pos, direction, type)

    def _trace_laser(self, start_pos, start_dir, emitter_type):
        pos = list(start_pos)
        direction = list(start_dir)
        target_type = emitter_type + 1

        path_points = [self._grid_to_iso_center(pos[0], pos[1])]

        for _ in range(self.MAX_LASER_LENGTH):
            next_pos = [pos[0] + direction[0], pos[1] + direction[1]]

            if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                # Hit boundary wall
                break

            pos = next_pos
            tile_content = self.grid[pos[0], pos[1]]
            path_points.append(self._grid_to_iso_center(pos[0], pos[1]))

            if tile_content == self.CRYSTAL:
                # Reflect 90 degrees
                direction[0], direction[1] = direction[1], direction[0]
            elif tile_content == target_type:
                self.current_targets_hit.add(target_type)
                break
            elif tile_content != self.EMPTY:
                # Hit an obstacle (wrong target, another emitter)
                break

        self.laser_paths.append((self.LASER_COLORS[emitter_type], path_points))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_walls()
        self._render_emitters_and_targets()
        self._render_crystals()
        self._render_lasers()
        self._render_cursor()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "crystals": self.crystals_remaining, "win": self.win}

    def _grid_to_iso(self, x, y):
        px = self.ISO_ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        py = self.ISO_ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(px), int(py)

    def _grid_to_iso_center(self, x, y):
        px, py = self._grid_to_iso(x, y)
        return px, py + self.TILE_HEIGHT_HALF

    def _render_grid_and_walls(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                px, py = self._grid_to_iso(x, y)
                points = [
                    (px, py + self.TILE_HEIGHT_HALF),
                    (px + self.TILE_WIDTH_HALF, py + 2 * self.TILE_HEIGHT_HALF),
                    (px, py + 3 * self.TILE_HEIGHT_HALF),
                    (px - self.TILE_WIDTH_HALF, py + 2 * self.TILE_HEIGHT_HALF)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

    def _render_emitters_and_targets(self):
        for item_type, pos in self.targets.items():
            is_hit = item_type in self.current_targets_hit
            self._render_iso_object(pos[0], pos[1], self.LASER_COLORS[item_type], "target", is_hit)

        for pos, _, emitter_type in self.emitters:
            self._render_iso_object(pos[0], pos[1], self.LASER_COLORS[emitter_type], "emitter")

    def _render_iso_object(self, x, y, color, shape, is_active=False):
        center_x, center_y = self._grid_to_iso_center(x, y)

        if shape == "target":
            radius = self.TILE_HEIGHT_HALF
            if is_active:
                glow_radius = int(radius * (1.5 + 0.2 * math.sin(self.steps * 0.2)))
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, (*color, 60))
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, glow_radius, (*color, 120))

            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_BG)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(radius * 0.8), color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_GRID)

        elif shape == "emitter":
            px, py = self._grid_to_iso(x, y)
            points = [
                (px, py + self.TILE_HEIGHT_HALF),
                (px + self.TILE_WIDTH_HALF, py + 2 * self.TILE_HEIGHT_HALF),
                (px, py + 3 * self.TILE_HEIGHT_HALF),
                (px - self.TILE_WIDTH_HALF, py + 2 * self.TILE_HEIGHT_HALF)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

    def _render_crystals(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == self.CRYSTAL:
                    center_x, center_y = self._grid_to_iso_center(x, y)

                    # Glow
                    glow_radius = self.TILE_WIDTH_HALF
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, (*self.COLOR_CRYSTAL_GLOW, 40))

                    # Crystal shape
                    size = self.TILE_HEIGHT_HALF
                    points = [
                        (center_x, center_y - size),
                        (center_x + size * 0.7, center_y),
                        (center_x, center_y + size),
                        (center_x - size * 0.7, center_y)
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)
                    pygame.gfxdraw.aapolygon(self.screen, points, (255, 255, 255))

    def _render_lasers(self):
        for color, path in self.laser_paths:
            if len(path) > 1:
                # Glow effect
                pygame.draw.lines(self.screen, (*color, 80), False, path, width=7)
                # Core beam
                pygame.draw.aalines(self.screen, color, False, path, 1)

    def _render_cursor(self):
        x, y = self.cursor_pos
        px, py = self._grid_to_iso(x, y)
        points = [
            (px, py + self.TILE_HEIGHT_HALF),
            (px + self.TILE_WIDTH_HALF, py + 2 * self.TILE_HEIGHT_HALF),
            (px, py + 3 * self.TILE_HEIGHT_HALF),
            (px - self.TILE_WIDTH_HALF, py + 2 * self.TILE_HEIGHT_HALF)
        ]
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 2)

    def _render_ui(self):
        # Crystal count
        crystal_text = self.font_small.render(f"Crystals: {self.crystals_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(crystal_text, (10, 10))

        # Score
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            message = "VICTORY!" if self.win else "OUT OF CRYSTALS"
            color = (100, 255, 100) if self.win else (255, 100, 100)

            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be run when the environment is loaded by the framework
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display for human interaction
    pygame.display.set_caption("Laser Cavern")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    running = True
    while running:
        # Pygame event handling
        action = [0, 0, 0]  # Default no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Wait 2 seconds
            obs, info = env.reset()

        # The game is not auto-advancing, so we need a delay
        # to make it playable by a human.
        pygame.time.wait(100) # 10 steps per second

    env.close()