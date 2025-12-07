
# Generated: 2025-08-27T22:57:08.067792
# Source Brief: brief_03300.md
# Brief Index: 3300

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro neon block breaker. Use the paddle to clear all blocks. Risky edge-hits and fast combos are rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 8
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 58, 20
        self.MAX_STEPS = 1000
        self.COMBO_WINDOW = 30  # in steps/frames

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 30, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (150, 150, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (255, 0, 128),  # Magenta
            (0, 255, 255),  # Cyan
            (0, 255, 0),    # Green
            (255, 128, 0),  # Orange
        ]
        
        # Game state variables (initialized in reset)
        self.paddle = None
        self.ball = None
        self.blocks = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.balls_left = 0
        self.combo_count = 0
        self.last_block_hit_time = 0
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize paddle
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )

        # Initialize blocks
        self.blocks = []
        num_rows = 5
        num_cols = 10
        y_offset = 50
        for i in range(num_rows):
            for j in range(num_cols):
                block_rect = pygame.Rect(
                    j * (self.BLOCK_WIDTH + 6) + 28,
                    i * (self.BLOCK_HEIGHT + 6) + y_offset,
                    self.BLOCK_WIDTH,
                    self.BLOCK_HEIGHT,
                )
                color_index = (i + j) % len(self.BLOCK_COLORS)
                self.blocks.append({"rect": block_rect, "color": self.BLOCK_COLORS[color_index]})

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.balls_left = 3
        self.particles = []
        self.combo_count = 0
        self.last_block_hit_time = -self.COMBO_WINDOW

        self._reset_ball()
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball = {
            "pos": np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=float),
            "vel": np.array([0.0, 0.0]),
            "state": "ready", # 'ready', 'active'
            "trail": []
        }
    
    def step(self, action):
        reward = -0.02  # Per-step penalty to encourage speed
        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # 1. Handle Input
        if not self.game_over:
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            self.paddle.clamp_ip(self.screen.get_rect())

            if self.ball["state"] == "ready" and space_held:
                # sfx: launch_ball.wav
                self.ball["state"] = "active"
                launch_angle = (self.np_random.random() - 0.5) * (math.pi / 4) # +/- 22.5 degrees
                self.ball["vel"] = np.array([math.sin(launch_angle), -math.cos(launch_angle)]) * self.BALL_SPEED

        # 2. Update Game State
        if self.ball["state"] == "active":
            self.ball["trail"].append(self.ball["pos"].copy())
            if len(self.ball["trail"]) > 5: self.ball["trail"].pop(0)

            self.ball["pos"] += self.ball["vel"]
            ball_rect = pygame.Rect(self.ball["pos"][0] - self.BALL_RADIUS, self.ball["pos"][1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

            # Ball-wall collision
            if ball_rect.left < 0 or ball_rect.right > self.WIDTH:
                self.ball["vel"][0] *= -1; self.ball["pos"][0] = np.clip(self.ball["pos"][0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                # sfx: wall_bounce.wav
            if ball_rect.top < 0:
                self.ball["vel"][1] *= -1; self.ball["pos"][1] = np.clip(self.ball["pos"][1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
                # sfx: wall_bounce.wav

            # Ball-paddle collision
            if self.paddle.colliderect(ball_rect) and self.ball["vel"][1] > 0:
                # sfx: paddle_hit.wav
                self.ball["vel"][1] *= -1; self.ball["pos"][1] = self.paddle.top - self.BALL_RADIUS
                dist_from_center = self.ball["pos"][0] - self.paddle.centerx
                normalized_dist = dist_from_center / (self.PADDLE_WIDTH / 2)
                
                if abs(normalized_dist) > 0.8: reward += 0.1 # Risky edge hit
                elif abs(normalized_dist) < 0.2: reward -= 0.2 # Safe center hit
                
                self.ball["vel"][0] += normalized_dist * 2.0
                norm = np.linalg.norm(self.ball["vel"])
                if norm > 0: self.ball["vel"] = self.ball["vel"] / norm * self.BALL_SPEED

            # Ball-block collision
            hit_block_idx = ball_rect.collidelist([b["rect"] for b in self.blocks])
            if hit_block_idx != -1:
                # sfx: block_break.wav
                block_data = self.blocks.pop(hit_block_idx)
                self.score += 1; reward += 1.0

                if self.steps - self.last_block_hit_time < self.COMBO_WINDOW:
                    self.combo_count += 1; reward += 5.0
                else: self.combo_count = 1
                self.last_block_hit_time = self.steps

                self._spawn_particles(block_data["rect"].center, block_data["color"])

                to_collision = np.array(ball_rect.center) - np.array(block_data["rect"].center)
                overlap_x = (self.BLOCK_WIDTH/2 + self.BALL_RADIUS) - abs(to_collision[0])
                overlap_y = (self.BLOCK_HEIGHT/2 + self.BALL_RADIUS) - abs(to_collision[1])
                if overlap_x < overlap_y: self.ball["vel"][0] *= -1
                else: self.ball["vel"][1] *= -1

            # Ball out of bounds
            if ball_rect.top > self.HEIGHT:
                # sfx: lose_ball.wav
                self.balls_left -= 1; self.combo_count = 0
                if self.balls_left > 0: self._reset_ball()
                else: self.game_over = True; self.win = False

        else: # ball is 'ready'
            self.ball["pos"][0] = self.paddle.centerx
            self.ball["trail"] = []

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles: p["pos"] += p["vel"]; p["life"] -= 1

        # 3. Check Termination
        terminated = False
        if not self.game_over and len(self.blocks) == 0:
            self.game_over = True; self.win = True; reward += 100; self.score += 100
        
        if self.game_over and self.win is False: reward -= 100; self.score -= 100

        if self.game_over or self.steps >= self.MAX_STEPS:
            terminated = True
            if self.game_over: self.ball["vel"] = np.array([0.0, 0.0])
        
        return (self._get_observation(), reward, terminated, False, self._get_info())
    
    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                "pos": np.array(pos, dtype=float),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "life": self.np_random.integers(15, 30),
                "color": color,
                "radius": self.np_random.random() * 2 + 1
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        for x in range(0, self.WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            highlight_rect = block["rect"].inflate(-6, -6)
            highlight_color = tuple(min(255, c + 40) for c in block["color"])
            pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=2)

        for p in self.particles:
            alpha = max(0, int(255 * (p["life"] / 30.0)))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), p["color"] + (alpha,))

        for i, pos in enumerate(self.ball["trail"]):
            alpha = int(255 * (i + 1) / (len(self.ball["trail"]) + 1) * 0.5)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.BALL_RADIUS, self.COLOR_BALL + (alpha,))

        ball_pos = (int(self.ball["pos"][0]), int(self.ball["pos"][1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos[0], ball_pos[1], self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW + (100,))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos[0], ball_pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        for i in range(self.balls_left):
            pos_x = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            pos_y = 10 + self.BALL_RADIUS
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.BALL_RADIUS, self.COLOR_BALL)

        if self.combo_count > 1:
            combo_text = self.font_large.render(f"x{self.combo_count}", True, self.COLOR_TEXT)
            text_rect = combo_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 50))
            self.screen.blit(combo_text, text_rect)

        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 128) if self.win else (255, 0, 0)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
            "combo": self.combo_count,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls for testing
    # Requires pygame to be installed with display support (pip install pygame)
    # Not using dummy driver here for interactive play
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    terminated = False
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("Playing game. Close the window to exit.")
    print(env.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Space button
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # Shift button (unused but can be mapped)
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over. Final Info: {info}")
    env.close()
    pygame.quit()