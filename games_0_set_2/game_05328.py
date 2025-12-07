
# Generated: 2025-08-28T04:41:03.672973
# Source Brief: brief_05328.md
# Brief Index: 5328

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Space to cycle color forwards (R→G→B), Shift to cycle backwards."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match your paddle's color to the falling ball. Score points for correct matches, but lose lives for misses or mismatches. First to 10 points wins!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 20
        self.BALL_RADIUS = 12
        self.PADDLE_SPEED = 12
        self.INITIAL_BALL_SPEED_Y = 3.0
        self.BALL_SPEED_INCREMENT = 0.5
        self.MAX_SCORE = 10
        self.MAX_MISSES = 5
        self.MAX_STEPS = 1000
        self.BALL_COLOR_CHANGE_INTERVAL = 90  # 3 seconds at 30fps

        # Colors
        self.COLOR_BG = (26, 26, 46)
        self.COLOR_UI = (230, 230, 255)
        self.COLOR_PADDLE_BORDER = (200, 200, 220)
        self.COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 80, 255)     # Blue
        ]
        self.COLOR_NAMES = ["RED", "GREEN", "BLUE"]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.missed_balls = 0
        self.game_over = False
        self.paddle_x = 0
        self.paddle_color_index = 0
        self.ball = {}
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.ball_color_timer = 0
        self.current_ball_speed_y = self.INITIAL_BALL_SPEED_Y
        self.rng = None

        self.reset()
        
        self.validate_implementation()

    def _reset_ball(self):
        """Resets the ball to the top of the screen."""
        self.ball = {
            "x": self.rng.integers(self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS),
            "y": self.BALL_RADIUS + 1,
            "vy": self.current_ball_speed_y,
            "color_index": self.rng.integers(0, len(self.COLORS))
        }
        self.ball_color_timer = self.BALL_COLOR_CHANGE_INTERVAL

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.missed_balls = 0
        self.game_over = False

        self.paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle_color_index = self.rng.integers(0, len(self.COLORS))
        
        self.current_ball_speed_y = self.INITIAL_BALL_SPEED_Y
        self._reset_ball()

        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Player Input & Continuous Rewards ---
        dist_before = abs((self.paddle_x + self.PADDLE_WIDTH / 2) - self.ball["x"])
        if movement == 3:  # Left
            self.paddle_x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_x += self.PADDLE_SPEED
        self.paddle_x = np.clip(self.paddle_x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)
        dist_after = abs((self.paddle_x + self.PADDLE_WIDTH / 2) - self.ball["x"])
        
        if dist_after < dist_before:
            reward += 0.1
        elif dist_after > dist_before:
            reward -= 0.1

        if space_held and not self.last_space_held:
            self.paddle_color_index = (self.paddle_color_index + 1) % len(self.COLORS)
            # sfx: color_swap_up.wav
        if shift_held and not self.last_shift_held:
            self.paddle_color_index = (self.paddle_color_index - 1 + len(self.COLORS)) % len(self.COLORS)
            # sfx: color_swap_down.wav
        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        if self.paddle_color_index == self.ball["color_index"]:
            reward += 0.5
        
        # --- Update Game State ---
        self.ball["y"] += self.ball["vy"]
        
        self.ball_color_timer -= 1
        if self.ball_color_timer <= 0:
            self.ball["color_index"] = (self.ball["color_index"] + 1) % len(self.COLORS)
            self.ball_color_timer = self.BALL_COLOR_CHANGE_INTERVAL
            # sfx: ball_color_change.wav

        self._update_particles()
        
        # --- Handle Events & Event Rewards ---
        paddle_rect = pygame.Rect(self.paddle_x, self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball["x"] - self.BALL_RADIUS, self.ball["y"] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        if paddle_rect.colliderect(ball_rect):
            if self.paddle_color_index == self.ball["color_index"]:
                self.score += 1
                reward += 1
                self.current_ball_speed_y += self.BALL_SPEED_INCREMENT
                self._create_particles(self.ball["x"], paddle_rect.top, self.COLORS[self.ball["color_index"]])
                # sfx: success_hit.wav
            else:
                self.missed_balls += 1
                reward -= 1
                self._create_particles(self.ball["x"], paddle_rect.top, (128, 128, 128), 10)
                # sfx: fail_hit.wav
            self._reset_ball()
        elif self.ball["y"] > self.SCREEN_HEIGHT:
            self.missed_balls += 1
            reward -= 1
            self._reset_ball()
            # sfx: miss.wav
            
        # --- Check for Termination ---
        if self.score >= self.MAX_SCORE:
            reward += 10
            terminated = True
            self.game_over = True
        elif self.missed_balls >= self.MAX_MISSES:
            reward -= 10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS - 1:
            terminated = True
            self.game_over = True

        self.steps += 1

        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "missed_balls": self.missed_balls,
        }
        
    def _render_game(self):
        for p in self.particles:
            alpha = int(p["life"] / p["max_life"] * 255)
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), int(p["size"]), color)

        ball_color = self.COLORS[self.ball["color_index"]]
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_color = (*ball_color, 80)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball["x"]), int(self.ball["y"]), glow_radius, glow_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball["x"]), int(self.ball["y"]), glow_radius, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball["x"]), int(self.ball["y"]), self.BALL_RADIUS, ball_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball["x"]), int(self.ball["y"]), self.BALL_RADIUS, ball_color)
        
        paddle_color = self.COLORS[self.paddle_color_index]
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10
        paddle_rect = pygame.Rect(int(self.paddle_x), int(paddle_y), self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, paddle_color, paddle_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_BORDER, paddle_rect, width=2, border_radius=5)

    def _render_ui(self):
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, 30))
        self.screen.blit(score_text, score_rect)

        misses_text = self.font_medium.render(f"LIVES: {self.MAX_MISSES - self.missed_balls}", True, self.COLOR_UI)
        misses_rect = misses_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(misses_text, misses_rect)
        
        if self.game_over:
            message, color = ("YOU WIN!", (150, 255, 150)) if self.score >= self.MAX_SCORE else ("GAME OVER", (255, 150, 150))
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _create_particles(self, x, y, color, count=30):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 3 + 1
            self.particles.append({
                "x": x, "y": y, "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "size": self.rng.random() * 4 + 2, "color": color,
                "life": self.rng.integers(20, 40), "max_life": 40
            })

    def _update_particles(self):
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    import time

    # To run in a window, comment out the next line
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    if is_headless:
        print("Running in headless mode. No window will be displayed.")
        start_time = time.time()
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if (i+1) % 100 == 0:
                print(f"Step {i+1}: Reward={reward:.2f}, Info={info}")
            if terminated:
                print(f"Episode finished after {i+1} steps.")
                obs, info = env.reset()
        end_time = time.time()
        print(f"1000 steps took {end_time - start_time:.2f} seconds.")
    else:
        print("\n--- Running with Pygame window for visualization ---")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Color Pong")
        clock = pygame.time.Clock()
        
        running = True
        terminated = False
        
        while running:
            if terminated:
                time.sleep(1) # Pause on game over
                obs, info = env.reset()
                terminated = False

            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            action = [movement, 1 if keys[pygame.K_SPACE] else 0, 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            clock.tick(30)
            
    env.close()
    pygame.quit()