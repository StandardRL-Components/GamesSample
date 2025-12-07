import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:12:22.789970
# Source Brief: brief_01549.md
# Brief Index: 1549
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete


class GameEnv(gym.Env):
    """
    GameEnv: Color Lock Puzzle
    The agent must manipulate the intensity of four colored lights to match five
    secret combinations, opening all doors before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manipulate the intensity of four colored lights to match secret combinations. "
        "Unlock all five doors before time runs out to win."
    )
    user_guide = "Controls: Use ←→ to select a light and ↑↓ to change its intensity."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_WALL = (40, 50, 60)
    COLOR_DOOR_FRAME = (70, 80, 90)
    COLOR_DOOR_CLOSED_BG = (50, 30, 30)
    COLOR_DOOR_OPEN_BG = (30, 60, 30)
    COLOR_DOOR_STATUS_CLOSED = (200, 50, 50)
    COLOR_DOOR_STATUS_OPEN = (50, 255, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SELECTOR = (255, 255, 255)

    BASE_LIGHT_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80)   # Yellow
    ]

    # Game parameters
    NUM_LIGHTS = 4
    NUM_DOORS = 5
    MAX_INTENSITY = 100
    INTENSITY_STEP = 10
    MAX_STEPS = 1200 # 120 seconds at 10 logic steps/sec

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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_intensity = pygame.font.Font(None, 24)

        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.lights = []
        self.door_solutions = []
        self.door_states = []
        self.door_openness = []
        self.selected_light_idx = 0
        self.min_dist_to_solution = 0

        # Initialize state for validation check
        self._initialize_state()

        # Critical self-check
        # self.validate_implementation() # Commented out for submission

    def _initialize_state(self):
        """Initializes all game state variables for a new episode."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS

        self.lights = [{'intensity': self.np_random.integers(0, 11) * 10} for _ in range(self.NUM_LIGHTS)]

        self._generate_solutions()
        self.door_states = [False] * self.NUM_DOORS
        self.door_openness = [0.0] * self.NUM_DOORS

        self.selected_light_idx = 0
        self.min_dist_to_solution = self._calculate_min_dist_to_solution()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def _generate_solutions(self):
        """Generates target intensity combinations for the doors."""
        self.door_solutions = []
        # First solution is easy as per brief
        self.door_solutions.append(tuple([50] * self.NUM_LIGHTS))

        # Generate other unique solutions
        while len(self.door_solutions) < self.NUM_DOORS:
            solution = tuple(self.np_random.integers(0, 11, size=self.NUM_LIGHTS) * 10)
            if solution not in self.door_solutions:
                self.door_solutions.append(solution)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1 # Unused per brief
        # shift_held = action[2] == 1 # Unused per brief

        self._update_player_state(movement)
        self._update_world_state()

        reward = self._calculate_reward()
        self.score += reward

        terminated = self._check_termination()
        truncated = False # This game has a time limit, but termination handles it.
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player_state(self, movement):
        """Updates light intensities and selections based on player action."""
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Up: Increase intensity
            # sfx: adjust_intensity_up
            self.lights[self.selected_light_idx]['intensity'] = min(
                self.MAX_INTENSITY,
                self.lights[self.selected_light_idx]['intensity'] + self.INTENSITY_STEP
            )
        elif movement == 2: # Down: Decrease intensity
            # sfx: adjust_intensity_down
            self.lights[self.selected_light_idx]['intensity'] = max(
                0,
                self.lights[self.selected_light_idx]['intensity'] - self.INTENSITY_STEP
            )
        elif movement == 3: # Left: Select previous light
            # sfx: selector_move
            self.selected_light_idx = (self.selected_light_idx - 1) % self.NUM_LIGHTS
        elif movement == 4: # Right: Select next light
            # sfx: selector_move
            self.selected_light_idx = (self.selected_light_idx + 1) % self.NUM_LIGHTS

    def _update_world_state(self):
        """Updates doors, timer, and other non-player elements."""
        self.steps += 1
        self.time_remaining -= 1

        current_intensities = tuple(light['intensity'] for light in self.lights)

        for i, solution in enumerate(self.door_solutions):
            if not self.door_states[i] and current_intensities == solution:
                self.door_states[i] = True
                # sfx: door_open

        # Animate door opening via interpolation for a smooth visual effect
        for i in range(self.NUM_DOORS):
            target_openness = 1.0 if self.door_states[i] else 0.0
            self.door_openness[i] += (target_openness - self.door_openness[i]) * 0.1

    def _calculate_min_dist_to_solution(self):
        """Calculates the sum of absolute differences to the closest unsolved solution."""
        current_intensities = np.array([light['intensity'] for light in self.lights])
        min_dist = float('inf')

        found_unsolved = False
        for i, solution in enumerate(self.door_solutions):
            if not self.door_states[i]:
                found_unsolved = True
                solution_intensities = np.array(solution)
                dist = np.sum(np.abs(current_intensities - solution_intensities))
                if dist < min_dist:
                    min_dist = dist

        return min_dist if found_unsolved else 0

    def _calculate_reward(self):
        """Calculates the reward for the current step based on game events."""
        reward = 0.0

        # Continuous reward for getting closer to a solution
        new_min_dist = self._calculate_min_dist_to_solution()
        dist_change = self.min_dist_to_solution - new_min_dist
        reward += (dist_change / 10.0) * 0.1 # Scaled reward for progress
        self.min_dist_to_solution = new_min_dist

        # Event-based reward for opening a door
        current_intensities = tuple(light['intensity'] for light in self.lights)
        for i, solution in enumerate(self.door_solutions):
             # Check if this door was just opened (openness is low but state is true)
             if self.door_states[i] and self.door_openness[i] < 0.2 and current_intensities == solution:
                 reward += 5.0

        # Terminal rewards
        if all(self.door_states):
            reward += 100.0 # Win
            # sfx: puzzle_complete_win
        elif self.time_remaining <= 0:
            reward -= 100.0 # Lose
            # sfx: puzzle_failed_lose

        return reward

    def _check_termination(self):
        """Checks if the episode should end."""
        win = all(self.door_states)
        lose = self.time_remaining <= 0
        return win or lose

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
            "time_remaining": self.time_remaining,
            "doors_opened": sum(self.door_states)
        }

    def _render_game(self):
        """Renders all primary game elements."""
        pygame.draw.rect(self.screen, self.COLOR_WALL, (20, 20, self.SCREEN_WIDTH - 40, self.SCREEN_HEIGHT - 40))
        self._render_lights()
        self._render_doors()

    def _render_lights(self):
        """Renders the four adjustable lights with glow and selector effects."""
        light_radius = 30
        light_y = self.SCREEN_HEIGHT * 0.65

        for i in range(self.NUM_LIGHTS):
            light_x = self.SCREEN_WIDTH * (i + 1) / (self.NUM_LIGHTS + 1)
            intensity_ratio = self.lights[i]['intensity'] / self.MAX_INTENSITY
            base_color = self.BASE_LIGHT_COLORS[i]
            current_color = tuple(int(c * intensity_ratio) for c in base_color)

            # Draw glow effect using layered transparent circles
            for j in range(light_radius, 0, -2):
                alpha = 60 * (1 - j / light_radius)**2
                glow_color = (*base_color, alpha * intensity_ratio)
                pygame.gfxdraw.filled_circle(self.screen, int(light_x), int(light_y), j + 5, glow_color)

            # Draw main light circle
            pygame.gfxdraw.aacircle(self.screen, int(light_x), int(light_y), light_radius, current_color)
            pygame.gfxdraw.filled_circle(self.screen, int(light_x), int(light_y), light_radius, current_color)

            # Draw pulsating selector ring
            if i == self.selected_light_idx:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                radius = light_radius + 10 + pulse * 4
                alpha = 150 + pulse * 105
                pygame.gfxdraw.aacircle(self.screen, int(light_x), int(light_y), int(radius), (*self.COLOR_SELECTOR, alpha))
                pygame.gfxdraw.aacircle(self.screen, int(light_x), int(light_y), int(radius-1), (*self.COLOR_SELECTOR, alpha))

    def _render_doors(self):
        """Renders the five doors with sliding animation and status lights."""
        door_width = 80
        door_height = 100
        door_y = 40

        for i in range(self.NUM_DOORS):
            door_x = self.SCREEN_WIDTH * (i + 1) / (self.NUM_DOORS + 1) - door_width / 2
            pygame.draw.rect(self.screen, self.COLOR_DOOR_FRAME, (door_x, door_y, door_width, door_height), 3)

            bg_color = self.COLOR_DOOR_OPEN_BG if self.door_states[i] else self.COLOR_DOOR_CLOSED_BG
            pygame.draw.rect(self.screen, bg_color, (door_x+3, door_y+3, door_width-6, door_height-6))

            # Sliding panel for opening animation
            panel_height = door_height - 6
            open_offset = panel_height * self.door_openness[i]
            pygame.draw.rect(self.screen, self.COLOR_WALL, (door_x+3, door_y+3, door_width-6, max(0, panel_height - open_offset)))

            # Status light
            status_color = self.COLOR_DOOR_STATUS_OPEN if self.door_states[i] else self.COLOR_DOOR_STATUS_CLOSED
            status_x, status_y = door_x + door_width / 2, door_y + door_height + 15
            pygame.gfxdraw.filled_circle(self.screen, int(status_x), int(status_y), 5, status_color)
            pygame.gfxdraw.aacircle(self.screen, int(status_x), int(status_y), 5, status_color)

    def _render_ui(self):
        """Renders text information like timer, score, and intensities."""
        # Light intensity values
        light_y = self.SCREEN_HEIGHT * 0.65 - 60
        for i in range(self.NUM_LIGHTS):
            light_x = self.SCREEN_WIDTH * (i + 1) / (self.NUM_LIGHTS + 1)
            text_surf = self.font_intensity.render(str(self.lights[i]['intensity']), True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(int(light_x), int(light_y)))
            self.screen.blit(text_surf, text_rect)

        # Timer
        time_sec = math.ceil(self.time_remaining / 10) if self.time_remaining > 0 else 0
        timer_surf = self.font_ui.render(f"TIME: {time_sec}", True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (30, self.SCREEN_HEIGHT - 40))

        # Score
        score_surf = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 30, self.SCREEN_HEIGHT - 40))
        self.screen.blit(score_surf, score_rect)

        # Doors Opened
        doors_surf = self.font_ui.render(f"DOORS: {sum(self.door_states)}/{self.NUM_DOORS}", True, self.COLOR_TEXT)
        doors_rect = doors_surf.get_rect(topright=(self.SCREEN_WIDTH - 30, 30))
        self.screen.blit(doors_surf, doors_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "SYSTEM UNLOCKED" if all(self.door_states) else "SYSTEM LOCKDOWN"
            msg_surf = self.font_ui.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Un-dummy the video driver for local play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Color Lock Puzzle")
    clock = pygame.time.Clock()

    running = True
    terminated = False

    while running:
        action = [0, 0, 0] # Default action: [no-op, space_released, shift_released]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    terminated = False
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        # Pygame uses a different coordinate system for blitting numpy arrays
        # Transpose back from (H, W, C) to (W, H, C) for make_surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate for human playability

    env.close()