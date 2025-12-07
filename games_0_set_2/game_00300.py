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
        "Controls: Use ← and → to move the paddle. Press Space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Clear all blocks by bouncing the ball with your paddle. Clear 3 stages to win, but lose all 3 lives and it's game over."
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
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # Colors (vibrant retro)
        self.COLOR_BG = (15, 20, 41) # Dark Blue
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0) # Bright Yellow
        self.COLOR_BALL_GLOW = (255, 255, 0, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = [
            (255, 69, 0),   # OrangeRed
            (0, 206, 209),  # DarkTurquoise
            (255, 20, 147), # DeepPink
            (50, 205, 50),  # LimeGreen
            (148, 0, 211),  # DarkViolet
        ]

        # Game objects properties
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 10
        self.BALL_RADIUS = 7
        self.BALL_SPEED = 7

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)

        # State variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.stars = []
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.steps = 0
        self.game_state = "PLAYING" # States: BALL_HELD, PLAYING, STAGE_CLEAR, GAME_OVER, GAME_WON
        self.game_over = False

        # RNG
        self.np_random = None
        
        # self.reset() is called by the environment wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Reset game state
        self.score = 0
        self.lives = 3
        self.stage = 1
        self.steps = 0
        self.game_over = False
        self.particles = []
        self._setup_stage(self.stage)
        self._reset_paddle_and_ball()
        
        # Generate background stars
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
            for _ in range(100)
        ]
        
        return self._get_observation(), self._get_info()

    def _reset_paddle_and_ball(self):
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        self.game_state = "BALL_HELD"

    def _setup_stage(self, stage_num):
        self.blocks = []
        block_width = 50
        block_height = 20
        margin_top = 50
        margin_x = (self.WIDTH - 11 * block_width) / 2 + 10 # Centered

        if stage_num == 1:
            for row in range(5):
                for col in range(11):
                    color = self.BLOCK_COLORS[row % len(self.BLOCK_COLORS)]
                    self.blocks.append({
                        "rect": pygame.Rect(margin_x + col * block_width, margin_top + row * block_height, block_width - 2, block_height - 2),
                        "color": color
                    })
        elif stage_num == 2:
            for row in range(6):
                for col in range(row, 11 - row):
                    color = self.BLOCK_COLORS[row % len(self.BLOCK_COLORS)]
                    self.blocks.append({
                        "rect": pygame.Rect(margin_x + col * block_width, margin_top + row * block_height, block_width - 2, block_height - 2),
                        "color": color
                    })
        elif stage_num == 3:
            for row in range(8):
                for col in range(11):
                    if (col + row) % 2 == 0:
                        color = self.BLOCK_COLORS[row % len(self.BLOCK_COLORS)]
                        self.blocks.append({
                            "rect": pygame.Rect(margin_x + col * block_width, margin_top + row * block_height, block_width - 2, block_height - 2),
                            "color": color
                        })

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        terminated = self.game_over
        if terminated:
            return self._get_observation(), 0, terminated, False, self._get_info()
        
        self.steps += 1
        reward = 0

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # 1. Handle paddle movement
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.02
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.02
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        # 2. Handle ball launch
        if self.game_state == "BALL_HELD":
            self.ball_pos.x = self.paddle.centerx
            if space_held:
                self.game_state = "PLAYING"
                # FIX: Swapped arguments to ensure low < high
                angle = self.np_random.uniform(-math.pi * 0.6, -math.pi * 0.4)
                self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED
                # sfx: ball_launch

        # 3. Update game logic if ball is in play
        if self.game_state == "PLAYING":
            reward += 0.1 # Survival reward
            self.ball_pos += self.ball_vel
            ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # Wall collisions
            if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.WIDTH - self.BALL_RADIUS:
                self.ball_vel.x *= -1
                self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                # sfx: wall_bounce
            if self.ball_pos.y <= self.BALL_RADIUS:
                self.ball_vel.y *= -1
                self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
                # sfx: wall_bounce

            # Paddle collision
            if self.ball_vel.y > 0 and ball_rect.colliderect(self.paddle):
                offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                offset = np.clip(offset, -1, 1)
                
                hit_dist_from_edge = min(abs(self.ball_pos.x - self.paddle.left), abs(self.paddle.right - self.ball_pos.x))
                if hit_dist_from_edge < self.PADDLE_WIDTH * 0.15:
                    reward += 5.0
                    # sfx: risky_save
                else:
                    reward -= 0.2
                    # sfx: paddle_bounce

                angle = math.pi * 0.5 * offset - math.pi * 0.5
                self.ball_vel.x = self.BALL_SPEED * math.cos(angle)
                self.ball_vel.y = -abs(self.BALL_SPEED * math.sin(angle))
                self.ball_pos.y = self.paddle.top - self.BALL_RADIUS

            # Block collisions
            hit_block_idx = ball_rect.collidelist([b["rect"] for b in self.blocks])
            if hit_block_idx != -1:
                block_data = self.blocks.pop(hit_block_idx)
                block_rect = block_data["rect"]
                
                # sfx: block_break
                reward += 1.0
                self.score += 10
                self._create_particles(block_rect.center, block_data["color"])

                prev_ball_pos = self.ball_pos - self.ball_vel
                if (prev_ball_pos.y + self.BALL_RADIUS <= block_rect.top or prev_ball_pos.y - self.BALL_RADIUS >= block_rect.bottom):
                    self.ball_vel.y *= -1
                else:
                    self.ball_vel.x *= -1

            # Lose life
            if self.ball_pos.y >= self.HEIGHT - self.BALL_RADIUS:
                self.lives -= 1
                reward -= 100.0
                # sfx: lose_life
                if self.lives > 0:
                    self._reset_paddle_and_ball()
                else:
                    self.game_state = "GAME_OVER"
                    self.game_over = True
        
        # 4. Check for stage clear / win
        if not self.blocks and self.game_state == "PLAYING":
            reward += 100.0
            self.score += 1000
            self.stage += 1
            if self.stage > 3:
                self.game_state = "GAME_WON"
                self.game_over = True
            else:
                self.game_state = "STAGE_CLEAR"
                # sfx: stage_clear
                self._setup_stage(self.stage)
                self._reset_paddle_and_ball()
        
        self._update_particles()
        terminated = self._check_termination()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": pygame.Vector2(pos), "vel": vel, "lifespan": lifespan, "color": color})
    
    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= 5000:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 220, 50), (x, y), size)
            
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            darker_color = tuple(max(0, c - 40) for c in block["color"])
            pygame.draw.rect(self.screen, darker_color, block["rect"].inflate(-6, -6), border_radius=2)

        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        for p in self.particles:
            p_pos_int = (int(p["pos"].x), int(p["pos"].y))
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            size = int(max(1, self.BALL_RADIUS * 0.5 * (p["lifespan"] / 30)))
            pygame.gfxdraw.filled_circle(self.screen, p_pos_int[0], p_pos_int[1], size, color)

    def _render_ui(self):
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        for i in range(self.lives):
            life_paddle = pygame.Rect(self.WIDTH - 10 - (i + 1) * 40, 15, 30, 8)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_paddle, border_radius=3)

        stage_text = self.font_medium.render(f"STAGE: {self.stage}", True, self.COLOR_UI_TEXT)
        stage_rect = stage_text.get_rect(centerx=self.WIDTH / 2, y=10)
        self.screen.blit(stage_text, stage_rect)

        if self.game_state in ["GAME_OVER", "GAME_WON"]:
            msg = "GAME OVER" if self.game_state == "GAME_OVER" else "YOU WIN!"
            color = (255, 50, 50) if self.game_state == "GAME_OVER" else (50, 255, 50)
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
            "blocks_left": len(self.blocks),
        }

    def close(self):
        pygame.quit()
        super().close()
        
    def _validate_implementation(self):
        # This is an internal validation method and can be removed if not needed.
        print("Running internal validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        obs, _ = self.reset(seed=1)
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=1)
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

# The original code had a `validate_implementation` call in __init__ which is not ideal.
# It also had a `reset` call in __init__. Both are removed as the environment wrapper
# handles initialization and reset.
# The original code also had a `validate_implementation` method that is not part of the
# standard Gym API. It has been renamed to `_validate_implementation` to indicate it is
# an internal helper, though it is not called automatically.