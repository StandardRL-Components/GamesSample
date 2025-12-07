# Generated: 2025-08-27T12:48:38.755838
# Source Brief: brief_00168.md
# Brief Index: 168

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
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
        "A retro arcade block-breaker. Clear all blocks to advance through 3 levels. Don't lose all your balls!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (20, 40, 80)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.BLOCK_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 255, 80), (80, 255, 255), (255, 80, 255)
        ]

        # Game constants
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12 # Higher for better responsiveness at 30fps
        self.BALL_RADIUS = 7
        self.MAX_LIVES = 3
        self.MAX_LEVELS = 3
        self.FPS = 30

        # State variables (will be initialized in reset)
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.level = 0
        self.level_timer = 0
        self.game_over = False
        self.win = False
        self.prev_space_held = False
        self.np_random = None
        
        # Initialize state variables
        self.reset()
        # self.validate_implementation() # Removed for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        
        self.score = 0
        self.level = 1
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.win = False
        self.steps = 0
        self.prev_space_held = True # Prevent launch on first frame after reset
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        # Reset paddle
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        # Reset ball
        self.ball_launched = False
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS], dtype=np.float32)
        base_speed = 5 + self.level * 0.5 # Ball speed increases per level
        angle = self.np_random.uniform(-math.pi / 6, math.pi / 6) # Narrower start angle
        self.ball_vel = np.array([base_speed * math.sin(angle), -base_speed * math.cos(angle)], dtype=np.float32)

        # Reset timer
        self.level_timer = 60 * self.FPS

        # Generate blocks
        self.blocks = []
        block_width = 50
        block_height = 20
        rows = 3 + self.level # Block rows increase per level
        cols = self.WIDTH // (block_width + 4)
        start_y = 50
        start_x = (self.WIDTH - cols * (block_width + 4)) // 2

        for i in range(rows):
            for j in range(cols):
                # Add some randomness to block placement for variety
                if self.np_random.random() > 0.05 * (4 - self.level):
                    block_rect = pygame.Rect(
                        start_x + j * (block_width + 4),
                        start_y + i * (block_height + 4),
                        block_width,
                        block_height
                    )
                    color_idx = self.np_random.integers(len(self.BLOCK_COLORS))
                    self.blocks.append({"rect": block_rect, "color": self.BLOCK_COLORS[color_idx]})
        
        # Clear particles
        self.particles = []

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02  # Time penalty from brief
        
        self.steps += 1
        self.level_timer -= 1

        # 1. Handle Input
        movement = action[0]
        space_held = action[1] == 1
        
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if space_held and not self.prev_space_held and not self.ball_launched:
            self.ball_launched = True
            # sfx: launch_sound.play()
        self.prev_space_held = space_held

        # 2. Update Game State
        if self.ball_launched:
            self.ball_pos += self.ball_vel
        else:
            self.ball_pos[0] = self.paddle.centerx
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

        # 3. Collision Detection
        ball_rect = pygame.Rect(int(self.ball_pos[0] - self.BALL_RADIUS), int(self.ball_pos[1] - self.BALL_RADIUS), self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Ball vs Walls
        if ball_rect.left <= 0:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = self.BALL_RADIUS
            # sfx: wall_bounce.play()
        if ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            # sfx: wall_bounce.play()
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce.play()

        # Ball vs Paddle
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.5
            self.ball_vel[0] = np.clip(self.ball_vel[0], -abs(self.ball_vel[1]) * 1.5, abs(self.ball_vel[1]) * 1.5)
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            # sfx: paddle_bounce.play()

        # Ball vs Blocks
        hit_block_idx = ball_rect.collidelist([b["rect"] for b in self.blocks])
        if hit_block_idx != -1:
            hit_block = self.blocks.pop(hit_block_idx)
            reward += 1.0
            self.score += 10
            self.ball_vel[1] *= -1
            # sfx: block_break.play()
            self._create_particles(hit_block["rect"].center, hit_block["color"])

        # 4. Update Particles
        self._update_particles()
        
        # 5. Check Game State Changes
        terminated = False
        if ball_rect.top > self.HEIGHT: # Lose a life
            self.lives -= 1
            reward += -1.0 # Penalty from brief
            self.ball_launched = False
            if self.lives <= 0:
                self.game_over = True
                terminated = True
                reward += -10.0 # Terminal penalty for losing
                # sfx: game_over.play()
            else:
                self._setup_level() # Reset ball and paddle, but keep score/level
                # sfx: lose_life.play()
        
        if not self.blocks and not self.game_over: # Level complete
            if self.level >= self.MAX_LEVELS:
                self.game_over = True
                self.win = True
                terminated = True
                reward += 100.0 # Win reward from brief
                self.score += 1000
                # sfx: game_win.play()
            else:
                reward += 10.0 # Level complete reward from brief
                self.score += 200
                self.level += 1
                self._setup_level()
                # sfx: level_complete.play()
                
        if self.level_timer <= 0 and not self.game_over: # Time out
            self.game_over = True
            terminated = True
            reward += -10.0 # Terminal penalty for losing
            # sfx: time_out.play()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = self.np_random.uniform(-2.5, 2.5, size=2)
            life = self.np_random.integers(15, 25)
            self.particles.append({"pos": np.array(pos, dtype=np.float32), "vel": vel, "life": life, "max_life": life, "color": color})

    def _update_particles(self):
        if not self.particles: return
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["life"] -= 1

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(int(c1 * (1 - interp) + c2 * interp) for c1, c2 in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        # Draw blocks with a 3D effect
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            brighter = tuple(min(255, c + 40) for c in block["color"])
            darker = tuple(max(0, c - 40) for c in block["color"])
            pts = block["rect"].topleft, block["rect"].topright, block["rect"].bottomright, block["rect"].bottomleft
            pygame.draw.line(self.screen, brighter, pts[0], pts[1], 2)
            pygame.draw.line(self.screen, brighter, pts[0], pts[3], 2)
            pygame.draw.line(self.screen, darker, pts[1], pts[2], 2)
            pygame.draw.line(self.screen, darker, pts[3], pts[2], 2)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)

        # Draw ball with glow
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        glow_radius = self.BALL_RADIUS * 2.5
        for i in range(int(glow_radius), 0, -2):
            alpha = 40 * (1 - (i / glow_radius))
            temp_surface = pygame.Surface((i*2, i*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, (*self.COLOR_BALL, alpha), (i, i), i)
            self.screen.blit(temp_surface, (x - i, y - i), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = p["life"] / p["max_life"]
            color = p["color"]
            size = max(1, int(alpha * 6))
            rect = pygame.Rect(int(p["pos"][0] - size/2), int(p["pos"][1] - size/2), size, size)
            temp_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            temp_surface.fill((*color, int(alpha * 255)))
            self.screen.blit(temp_surface, rect.topleft)

        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Score
        score_surf = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Lives
        lives_text_surf = self.font_medium.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text_surf, (self.WIDTH - 150, 10))
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 60 + i * 20, 25, 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 60 + i * 20, 25, 6, self.COLOR_BALL)

        # Level and Timer
        time_left = max(0, self.level_timer // self.FPS)
        info_text = f"LEVEL {self.level}  |  TIME: {time_left}"
        info_surf = self.font_small.render(info_text, True, self.COLOR_TEXT)
        info_rect = info_surf.get_rect(center=(self.WIDTH // 2, 20))
        self.screen.blit(info_surf, info_rect)

        # Game State Messages
        if not self.ball_launched and not self.game_over:
            msg_surf = self.font_medium.render("PRESS SPACE TO LAUNCH", True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            self.screen.blit(msg_surf, msg_rect)
            
            final_score_surf = self.font_medium.render(f"FINAL SCORE: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 30))
            self.screen.blit(final_score_surf, final_score_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "lives": self.lives,
        }

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")