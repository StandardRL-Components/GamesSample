import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:43:49.715921
# Source Brief: brief_02339.md
# Brief Index: 2339
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Nanite Infiltration: A turn-based stealth puzzle game.

    The player controls a green nanite on a grid, aiming to reach blue
    objectives while avoiding red enemy patrols. The game is turn-based,
    with enemies moving along predefined paths at set intervals.
    Completing objectives makes enemies faster. The episode ends upon
    completing all objectives (win), being caught (loss), or reaching
    the maximum turn limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Control a nanite to reach objectives while avoiding patrols in this turn-based stealth puzzle."
    user_guide = "Use the arrow keys (↑↓←→) to move one square at a time. Press space to wait a turn."
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 12
    CELL_WIDTH = SCREEN_WIDTH // GRID_WIDTH
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT

    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (40, 50, 70)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_PATH = (128, 25, 25)
    COLOR_OBJECTIVE = (50, 150, 255)
    COLOR_OBJECTIVE_CURRENT = (100, 200, 255)
    COLOR_TEXT = (220, 220, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20)
            self.font_title = pygame.font.SysFont("Consolas", 32, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_title = pygame.font.Font(None, 40)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.max_steps = 1000

        self.player_pos = (0, 0)
        self.objectives = []
        self.current_objective_idx = 0

        self.enemies = []
        self.initial_enemy_move_interval = 25
        self.enemy_move_interval = self.initial_enemy_move_interval
        self.enemy_speed_increase_factor = 0.8 # Patrols become 20% faster

        # Initialize state
        # self.reset() # reset is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.enemy_move_interval = self.initial_enemy_move_interval

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Procedurally generates the game level layout."""
        # Place player in a safe starting corner
        self.player_pos = (
            self.np_random.integers(0, self.GRID_WIDTH // 4),
            self.np_random.integers(0, self.GRID_HEIGHT)
        )

        used_positions = {self.player_pos}

        # Generate objectives
        self.objectives = []
        for _ in range(3):
            pos = self._get_random_empty_pos(used_positions)
            self.objectives.append(pos)
            used_positions.add(pos)
        self.current_objective_idx = 0

        # Generate enemies and their patrol paths
        self.enemies = []
        for _ in range(3):
            path_start_pos = self._get_random_empty_pos(used_positions)
            used_positions.add(path_start_pos)
            path = self._generate_patrol_path(path_start_pos)
            if path:
                self.enemies.append({
                    'pos': path[0],
                    'path_idx': 0,
                    'path': path
                })

    def _get_random_empty_pos(self, used_positions):
        """Finds a random grid cell not in the used_positions set."""
        while True:
            pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if pos not in used_positions:
                return pos

    def _generate_patrol_path(self, start_pos):
        """Creates a simple, cyclical patrol path for an enemy."""
        path = [start_pos]
        x, y = start_pos
        w = self.np_random.integers(3, 8)
        h = self.np_random.integers(3, 8)

        p1 = (min(x + w, self.GRID_WIDTH - 1), y)
        p2 = (min(x + w, self.GRID_WIDTH - 1), min(y + h, self.GRID_HEIGHT - 1))
        p3 = (x, min(y + h, self.GRID_HEIGHT - 1))

        path.extend(self._get_line_points(start_pos, p1))
        path.extend(self._get_line_points(p1, p2))
        path.extend(self._get_line_points(p2, p3))
        path.extend(self._get_line_points(p3, start_pos))

        unique_path = []
        [unique_path.append(item) for item in path if item not in unique_path]
        return unique_path

    def _get_line_points(self, p1, p2):
        """Generates all grid cells for a line between two points."""
        points = []
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2: break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return points[1:]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = 0.0
        terminated = False

        # --- Player Logic ---
        old_pos = self.player_pos
        px, py = self.player_pos
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right
        self.player_pos = (max(0, min(self.GRID_WIDTH - 1, px)), max(0, min(self.GRID_HEIGHT - 1, py)))

        # --- Enemy Logic ---
        if self.steps > 0 and self.steps % self.enemy_move_interval == 0:
            for enemy in self.enemies:
                if enemy['path']:
                    enemy['path_idx'] = (enemy['path_idx'] + 1) % len(enemy['path'])
                    enemy['pos'] = enemy['path'][enemy['path_idx']]
                    # Placeholder: # SFX: enemy_swoosh.wav

        # --- State Checks & Reward Calculation ---
        # 1. Detection (loss condition)
        if any(self.player_pos == enemy['pos'] for enemy in self.enemies):
            reward = -10.0
            self.game_over = True
            terminated = True
            # Placeholder: # SFX: detection_alarm.wav
        else:
            # 2. Objective completion
            if self.current_objective_idx < len(self.objectives):
                current_objective_pos = self.objectives[self.current_objective_idx]
                if self.player_pos == current_objective_pos:
                    reward += 1.0
                    self.current_objective_idx += 1
                    # Placeholder: # SFX: objective_complete.wav

                    if self.current_objective_idx >= len(self.objectives):
                        reward += 100.0
                        self.game_over = True
                        terminated = True
                        # Placeholder: # SFX: win_fanfare.wav
                    else:
                        self.enemy_move_interval = max(5, int(self.enemy_move_interval * self.enemy_speed_increase_factor))
                
                # 3. Distance-based reward
                elif self.player_pos != old_pos and not self.game_over:
                    next_obj_pos = self.objectives[self.current_objective_idx]
                    old_dist = math.hypot(old_pos[0] - next_obj_pos[0], old_pos[1] - next_obj_pos[1])
                    new_dist = math.hypot(self.player_pos[0] - next_obj_pos[0], self.player_pos[1] - next_obj_pos[1])
                    if new_dist < old_dist: reward += 0.1
                    elif new_dist > old_dist: reward -= 0.1

        # 4. Max steps termination
        if self.steps >= self.max_steps:
            self.game_over = True
            terminated = True

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "objectives_completed": self.current_objective_idx,
            "total_objectives": len(self.objectives),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

        # Draw enemy paths
        for enemy in self.enemies:
            if len(enemy['path']) > 1:
                path_pixels = [(p[0] * self.CELL_WIDTH + self.CELL_WIDTH // 2, p[1] * self.CELL_HEIGHT + self.CELL_HEIGHT // 2) for p in enemy['path']]
                pygame.draw.lines(self.screen, self.COLOR_ENEMY_PATH, True, path_pixels, 1)

        # Draw objectives
        for i, pos in enumerate(self.objectives):
            rect = pygame.Rect(pos[0] * self.CELL_WIDTH + 4, pos[1] * self.CELL_HEIGHT + 4, self.CELL_WIDTH - 8, self.CELL_HEIGHT - 8)
            if i < self.current_objective_idx:
                pygame.draw.rect(self.screen, (50, 80, 100), rect, 2)
            elif i == self.current_objective_idx:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                color = tuple(int(c1 + (c2 - c1) * pulse) for c1, c2 in zip(self.COLOR_OBJECTIVE, self.COLOR_OBJECTIVE_CURRENT))
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 2)
            else:
                pygame.draw.rect(self.screen, self.COLOR_OBJECTIVE, rect, 2)

        # Draw enemies
        for enemy in self.enemies:
            center_x = int(enemy['pos'][0] * self.CELL_WIDTH + self.CELL_WIDTH / 2)
            center_y = int(enemy['pos'][1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2)
            radius = int(self.CELL_WIDTH / 3)
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, (center_x, center_y), radius)

        # Draw player
        is_caught = any(self.player_pos == enemy['pos'] for enemy in self.enemies)
        if not (self.game_over and is_caught):
            center_x = int(self.player_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH / 2)
            center_y = int(self.player_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2)
            radius = int(self.CELL_WIDTH / 2.5)
            glow_radius = int(radius * (1.2 + 0.2 * math.sin(self.steps * 0.3)))
            self._draw_glowing_circle(self.screen, (center_x, center_y), glow_radius, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)

    def _draw_glowing_circle(self, surface, pos, radius, color):
        """Draws a circle with a soft, additive glow effect."""
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for i in range(radius, 0, -2):
            alpha = int(255 * (1 - (i / radius))**2) // 4
            pygame.gfxdraw.filled_circle(surf, radius, radius, i, color + (alpha,))
        surface.blit(surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_ui.render(f"TURN: {self.steps}/{self.max_steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 35))
        obj_text = self.font_ui.render(f"OBJECTIVES: {self.current_objective_idx}/{len(self.objectives)}", True, self.COLOR_TEXT)
        self.screen.blit(obj_text, (self.SCREEN_WIDTH - obj_text.get_width() - 10, 10))

        if self.game_over:
            is_win = self.current_objective_idx >= len(self.objectives)
            msg = "MISSION COMPLETE" if is_win else "DETECTION"
            color = self.COLOR_PLAYER if is_win else self.COLOR_ENEMY
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            title_text = self.font_title.render(msg, True, color)
            title_rect = title_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(title_text, title_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    env.validate_implementation()
    obs, info = env.reset()
    done = False
    
    # For human play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    pygame.font.init()

    pygame.display.set_caption("Nanite Infiltration")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        action_taken = False
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif done and event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                elif not done:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    elif event.key == pygame.K_SPACE: action[0] = 0 # Wait
                    # Only consider movement keys as taking a turn
                    if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                        action_taken = True

        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")

        # Render the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(15)

    env.close()