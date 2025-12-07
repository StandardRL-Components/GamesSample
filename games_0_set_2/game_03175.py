import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the robot. Avoid the red traps and reach the green exit."
    )

    game_description = (
        "Navigate a robot through a trap-laden grid to reach the exit in the fewest steps. "
        "The number of traps increases after each successful run."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 500
        self.MIN_TRAPS = 10
        self.MAX_TRAPS = 25

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_ROBOT = (50, 150, 255)
        self.COLOR_ROBOT_GLOW = (150, 200, 255)
        self.COLOR_EXIT = (40, 220, 120)
        self.COLOR_EXIT_GLOW = (140, 255, 200)
        self.COLOR_TRAP = (255, 80, 80)
        self.COLOR_TRAP_TRIGGERED = (255, 200, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # --- Game State (Persistent across resets) ---
        self.num_traps_to_generate = self.MIN_TRAPS
        self.last_episode_won = False

        # --- Game State (Reset every episode) ---
        self.robot_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.trap_locations = set()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""  # "EXIT", "TRAP", "TIME"

        self.reset()
        # self.validate_implementation() # Commented out as it's for debug

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Difficulty Progression ---
        if self.last_episode_won:
            self.num_traps_to_generate = min(self.MAX_TRAPS, self.num_traps_to_generate + 1)

        # --- Reset State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_episode_won = False
        self.termination_reason = ""

        # --- Generate a Solvable Level ---
        path_found = False
        generation_attempts = 0
        while not path_found and generation_attempts < 100:
            # Place exit
            self.exit_pos = tuple(self.np_random.integers(0, [self.GRID_WIDTH, self.GRID_HEIGHT]))

            # Place robot, ensuring it's not on the exit
            while True:
                self.robot_pos = tuple(self.np_random.integers(0, [self.GRID_WIDTH, self.GRID_HEIGHT]))
                if self.robot_pos != self.exit_pos:
                    break

            # Place traps
            self.trap_locations = set()
            while len(self.trap_locations) < self.num_traps_to_generate:
                trap_pos = tuple(self.np_random.integers(0, [self.GRID_WIDTH, self.GRID_HEIGHT]))
                if trap_pos != self.robot_pos and trap_pos != self.exit_pos:
                    self.trap_locations.add(trap_pos)

            path_found = self._is_path_possible(self.robot_pos, self.exit_pos, self.trap_locations)
            generation_attempts += 1

        if not path_found:
            # Fallback if generation fails repeatedly
            # print("Warning: Failed to generate a solvable level. Creating a simple one.")
            self.trap_locations = set()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right

        # --- Update Robot Position ---
        new_pos = list(self.robot_pos)
        if movement == 1: new_pos[1] -= 1  # Up
        elif movement == 2: new_pos[1] += 1  # Down
        elif movement == 3: new_pos[0] -= 1  # Left
        elif movement == 4: new_pos[0] += 1  # Right

        # Check boundaries
        if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
            self.robot_pos = tuple(new_pos)

        self.steps += 1
        reward = -0.1  # Cost per step
        terminated = False

        # --- Check Termination Conditions ---
        if self.robot_pos == self.exit_pos:
            reward = 100.0
            terminated = True
            self.game_over = True
            self.last_episode_won = True
            self.termination_reason = "EXIT"
        elif self.robot_pos in self.trap_locations:
            reward = -50.0
            terminated = True
            self.game_over = True
            self.termination_reason = "TRAP"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.termination_reason = "TIME"

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _is_path_possible(self, start, end, obstacles):
        """Checks for a path using Breadth-First Search."""
        queue = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()

            if current == end:
                return True

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if (0 <= neighbor[0] < self.GRID_WIDTH and
                        0 <= neighbor[1] < self.GRID_HEIGHT and
                        neighbor not in visited and
                        neighbor not in obstacles):
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "traps": self.num_traps_to_generate,
        }

    def _render_game(self):
        # --- Draw Grid Lines ---
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # --- Draw Traps ---
        for tx, ty in self.trap_locations:
            if self.termination_reason == "TRAP" and (tx, ty) == self.robot_pos:
                continue  # Skip drawing trap under triggered robot
            rect = pygame.Rect(tx * self.CELL_SIZE, ty * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_TRAP, rect.inflate(-4, -4))

        # --- Draw Exit ---
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        if self.termination_reason == "EXIT":
            pygame.gfxdraw.box(self.screen, exit_rect, (*self.COLOR_EXIT_GLOW, 150))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect.inflate(-4, -4))

        # --- Draw Robot ---
        rx, ry = self.robot_pos
        robot_rect = pygame.Rect(rx * self.CELL_SIZE, ry * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)

        # Glow effect
        glow_rect = robot_rect.inflate(self.CELL_SIZE * 0.4, self.CELL_SIZE * 0.4)
        pygame.gfxdraw.box(self.screen, glow_rect, (*self.COLOR_ROBOT_GLOW, 80))

        # Main body
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect.inflate(-4, -4))

        # --- Draw Termination Effects ---
        if self.termination_reason == "TRAP":
            center_x = robot_rect.centerx
            center_y = robot_rect.centery
            size = self.CELL_SIZE // 2
            # Draw a yellow 'X' for explosion
            pygame.draw.line(self.screen, self.COLOR_TRAP_TRIGGERED, (center_x - size, center_y - size),
                             (center_x + size, center_y + size), 5)
            pygame.draw.line(self.screen, self.COLOR_TRAP_TRIGGERED, (center_x - size, center_y + size),
                             (center_x + size, center_y - size), 5)

    def _render_ui(self):
        # --- UI Background Panel ---
        ui_panel = pygame.Surface((self.WIDTH, 35), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # --- Render Text ---
        steps_text = self.font_ui.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        traps_text = self.font_ui.render(f"Traps: {self.num_traps_to_generate}", True, self.COLOR_TEXT)

        self.screen.blit(steps_text, (10, 5))
        self.screen.blit(score_text, (200, 5))
        self.screen.blit(traps_text, (400, 5))

        # --- Game Over Message ---
        if self.game_over:
            msg = ""
            if self.termination_reason == "EXIT":
                msg = "SUCCESS!"
                color = self.COLOR_EXIT
            elif self.termination_reason == "TRAP":
                msg = "TRAP ACTIVATED"
                color = self.COLOR_TRAP
            elif self.termination_reason == "TIME":
                msg = "OUT OF TIME"
                color = self.COLOR_TEXT

            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))

            # Text shadow for readability
            shadow_text = self.font_game_over.render(msg, True, (0, 0, 0))
            self.screen.blit(shadow_text, text_rect.move(3, 3))

            self.screen.blit(over_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Validating implementation...")
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
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False

    # Use a dummy screen for display if running as a script
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Grid Runner")

    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset on 'R' key
                    obs, info = env.reset()
                    done = False

        if not env.game_over:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = np.array([movement, space_held, shift_held])

            # Since auto_advance is False, we only step if there's an action or key press
            if movement > 0:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated

        # --- Rendering to the display ---
        # The environment returns the frame as a numpy array, so we convert it back to a surface
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        # Since this is a turn-based game, we can use a small delay to prevent high CPU usage
        pygame.time.wait(30)

    env.close()