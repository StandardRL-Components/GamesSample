import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:49:46.857870
# Source Brief: brief_01980.md
# Brief Index: 1980
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player guides a falling ball across three
    vertically oscillating platforms.

    The goal is to complete five levels by making the ball touch all three platforms
    and then exit the bottom of the screen within a time limit. Difficulty
    increases with each level as the platforms oscillate faster.

    Visuals are minimalist and geometric, with a focus on clarity and "game feel".
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Guide a falling ball to touch three oscillating platforms before time runs out. "
        "Select and stop platforms strategically to control the ball's descent through multiple levels."
    )
    user_guide = (
        "Controls: Use ← and → to select the left and right platforms. "
        "Hold Shift to select the middle platform. Press Space to temporarily stop the selected platform."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60 # Internal simulation rate

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (200, 200, 255)
    COLOR_PLATFORM_R = (255, 80, 80)
    COLOR_PLATFORM_G = (80, 255, 80)
    COLOR_PLATFORM_B = (80, 80, 255)
    COLOR_FLASH = (255, 255, 255)
    COLOR_SELECTED = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_VALUE = (255, 255, 255)

    # Physics
    GRAVITY = 250.0 # pixels/sec^2
    BALL_BOUNCE_FACTOR = -0.95
    BALL_WALL_BOUNCE_FACTOR = -0.8

    # Ball
    BALL_RADIUS = 10

    # Platforms
    PLATFORM_WIDTH = 120
    PLATFORM_HEIGHT = 15
    PLATFORM_Y_CENTER = 250
    PLATFORM_AMPLITUDE = 80
    BASE_OSC_SPEED = 1.0 # rad/sec
    LEVEL_OSC_SPEED_INCREASE = 0.5 # rad/sec per level

    # Game Flow
    MAX_LEVELS = 5
    LEVEL_TIME_LIMIT = 15.0 # seconds
    PLATFORM_FREEZE_DURATION = 0.2 # seconds
    PLATFORM_PLAYER_STOP_DURATION = 0.25 # seconds
    PLATFORM_FLASH_DURATION = 0.15 # seconds
    LEVEL_COMPLETE_Y_THRESH = SCREEN_HEIGHT - 30

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_level = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Internal State ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.timer = 0.0
        self.ball_pos = None
        self.ball_vel = None
        self.platforms = []
        self.selected_platform_idx = 0
        self.touched_platforms = set()
        self.dt = 1 / self.FPS

        # self.reset() # Removed to follow Gymnasium API, reset is called by the user
        # self.validate_implementation() # Removed as it's for dev, not part of the env

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self._start_new_level(1)

        return self._get_observation(), self._get_info()

    def _start_new_level(self, level):
        """Resets state for the beginning of a new level."""
        self.level = level
        self.timer = self.LEVEL_TIME_LIMIT
        self.touched_platforms.clear()
        self.selected_platform_idx = 0

        # Reset ball
        self.ball_pos = np.array([self.SCREEN_WIDTH / 2, self.BALL_RADIUS * 2], dtype=float)
        self.ball_vel = np.array([self.np_random.uniform(-50, 50), 0.0], dtype=float)

        # Reset platforms
        self.platforms = []
        platform_colors = [self.COLOR_PLATFORM_R, self.COLOR_PLATFORM_G, self.COLOR_PLATFORM_B]
        platform_x_positions = [
            self.SCREEN_WIDTH * 0.25,
            self.SCREEN_WIDTH * 0.50,
            self.SCREEN_WIDTH * 0.75,
        ]
        # Random phase shifts for variety
        phases = [0, math.pi * 2/3, math.pi * 4/3]
        self.np_random.shuffle(phases)

        for i in range(3):
            self.platforms.append({
                'x': platform_x_positions[i],
                'y': self.PLATFORM_Y_CENTER,
                'color': platform_colors[i],
                'phase': phases[i],
                'freeze_timer': 0.0,
                'stop_timer': 0.0,
                'flash_timer': 0.0,
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        self._handle_input(action)
        self._update_physics()
        reward += self._handle_collisions()
        reward += self._check_game_state()

        # Small penalty for time passing
        reward -= 0.01
        # Small reward for ball staying on screen
        if not self.game_over:
            reward += 0.02

        self.score += reward
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        """Process the MultiDiscrete action from the agent."""
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Platform selection
        if shift_held:
            self.selected_platform_idx = 1  # Middle (Green)
        elif movement == 3:  # Left
            self.selected_platform_idx = 0  # Left (Red)
        elif movement == 4:  # Right
            self.selected_platform_idx = 2  # Right (Blue)

        # Platform stop action
        if space_held:
            # SFX: player_stop_platform.wav
            self.platforms[self.selected_platform_idx]['stop_timer'] = self.PLATFORM_PLAYER_STOP_DURATION

    def _update_physics(self):
        """Update positions and velocities of all game objects."""
        # Update timers
        self.timer = max(0, self.timer - self.dt)
        for p in self.platforms:
            p['freeze_timer'] = max(0, p['freeze_timer'] - self.dt)
            p['stop_timer'] = max(0, p['stop_timer'] - self.dt)
            p['flash_timer'] = max(0, p['flash_timer'] - self.dt)

        # Update platforms
        osc_speed = self.BASE_OSC_SPEED + self.LEVEL_OSC_SPEED_INCREASE * (self.level - 1)
        for p in self.platforms:
            if p['freeze_timer'] <= 0 and p['stop_timer'] <= 0:
                time_component = self.steps * self.dt * osc_speed
                p['y'] = self.PLATFORM_Y_CENTER + self.PLATFORM_AMPLITUDE * math.sin(time_component + p['phase'])

        # Update ball
        self.ball_vel[1] += self.GRAVITY * self.dt
        self.ball_pos += self.ball_vel * self.dt

    def _handle_collisions(self):
        """Check and resolve collisions for the ball."""
        reward = 0.0

        # Ball vs. side walls
        if self.ball_pos[0] < self.BALL_RADIUS or self.ball_pos[0] > self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
            self.ball_vel[0] *= self.BALL_WALL_BOUNCE_FACTOR
            # SFX: wall_bounce.wav

        # Ball vs. top wall
        if self.ball_pos[1] < self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= self.BALL_WALL_BOUNCE_FACTOR
            # SFX: wall_bounce.wav

        # Ball vs. platforms
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )

        for i, p in enumerate(self.platforms):
            platform_rect = pygame.Rect(
                p['x'] - self.PLATFORM_WIDTH / 2,
                p['y'] - self.PLATFORM_HEIGHT / 2,
                self.PLATFORM_WIDTH,
                self.PLATFORM_HEIGHT
            )
            # Check for collision only if ball is falling and above the platform
            if self.ball_vel[1] > 0 and ball_rect.colliderect(platform_rect):
                # Resolve position to prevent sinking
                self.ball_pos[1] = p['y'] - self.PLATFORM_HEIGHT / 2 - self.BALL_RADIUS
                self.ball_vel[1] *= self.BALL_BOUNCE_FACTOR
                # SFX: platform_bounce.wav

                # Add to touched platforms and grant reward if new
                if i not in self.touched_platforms:
                    self.touched_platforms.add(i)
                    reward += 1.0

                # Trigger effects
                p['flash_timer'] = self.PLATFORM_FLASH_DURATION
                for j in range(3):
                    if i != j:
                        self.platforms[j]['freeze_timer'] = self.PLATFORM_FREEZE_DURATION
                break # Only collide with one platform per frame
        return reward

    def _check_game_state(self):
        """Check for win/loss/level-up conditions."""
        reward = 0.0

        # Loss Condition: Ball falls off bottom
        if self.ball_pos[1] > self.SCREEN_HEIGHT + self.BALL_RADIUS:
            self.game_over = True
            reward -= 100.0
            # SFX: game_over_fall.wav
            return reward

        # Loss Condition: Timer runs out
        if self.timer <= 0:
            self.game_over = True
            reward -= 100.0
            # SFX: game_over_timeout.wav
            return reward

        # Level Complete Condition
        if len(self.touched_platforms) == 3 and self.ball_pos[1] > self.LEVEL_COMPLETE_Y_THRESH:
            if self.level == self.MAX_LEVELS:
                # VICTORY!
                self.game_over = True
                reward += 100.0
                # SFX: game_win.wav
            else:
                # Level Up
                reward += 5.0
                self._start_new_level(self.level + 1)
                # SFX: level_up.wav

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the main game elements (ball, platforms)."""
        # Draw platforms
        for i, p in enumerate(self.platforms):
            color = p['flash_timer'] > 0 and self.COLOR_FLASH or p['color']
            rect = pygame.Rect(
                p['x'] - self.PLATFORM_WIDTH / 2,
                p['y'] - self.PLATFORM_HEIGHT / 2,
                self.PLATFORM_WIDTH,
                self.PLATFORM_HEIGHT
            )
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

            # Draw selection indicator
            if i == self.selected_platform_idx:
                pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, width=3, border_radius=4)

        # Draw ball with glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y,
                                     self.BALL_RADIUS + 3, (*self.COLOR_BALL_GLOW, 80))
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y,
                                     self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y,
                                self.BALL_RADIUS, self.COLOR_BALL)


    def _render_ui(self):
        """Renders the UI overlay (timer, level, score)."""
        # Level Display
        level_text = self.font_level.render(f"Level", True, self.COLOR_UI_TEXT)
        level_val = self.font_level.render(f"{self.level}/{self.MAX_LEVELS}", True, self.COLOR_UI_VALUE)
        self.screen.blit(level_text, (15, 10))
        self.screen.blit(level_val, (15, 35))

        # Timer Display
        timer_text = self.font_ui.render("Time", True, self.COLOR_UI_TEXT)
        timer_val = self.font_ui.render(f"{self.timer:.1f}", True, self.COLOR_UI_VALUE)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_val.get_width() - 15, 10))
        self.screen.blit(timer_val, (self.SCREEN_WIDTH - timer_val.get_width() - 15, 30))

        # Score Display
        score_text = self.font_ui.render("Score", True, self.COLOR_UI_TEXT)
        score_val = self.font_ui.render(f"{int(self.score)}", True, self.COLOR_UI_VALUE)
        self.screen.blit(score_text, (self.SCREEN_WIDTH / 2 - score_text.get_width()/2, 10))
        self.screen.blit(score_val, (self.SCREEN_WIDTH / 2 - score_val.get_width()/2, 30))

        # Touched platforms indicator
        for i in range(3):
            is_touched = i in self.touched_platforms
            color = self.platforms[i]['color'] if is_touched else (50, 50, 60)
            x_pos = self.SCREEN_WIDTH / 2 - 30 + (i * 30)
            pygame.draw.circle(self.screen, color, (int(x_pos), 70), 10)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "timer": self.timer,
            "ball_pos": self.ball_pos.tolist(),
            "touched_platforms": len(self.touched_platforms)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Example Usage & Manual Play ---
    # The main loop needs a visible display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Pygame setup for manual play
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Platform Fall")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    while running:
        # --- Action Mapping for Human Player ---
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Environment ---")
                obs, info = env.reset()
                total_reward = 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
            print("Press 'R' to restart.")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()