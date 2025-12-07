
# Generated: 2025-08-28T05:43:53.334620
# Source Brief: brief_02715.md
# Brief Index: 2715

        
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
        "A retro arcade block breaker. Clear all the blocks without losing your balls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (15, 15, 30)
    COLOR_GRID = (30, 30, 60)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (200, 200, 220)
    BLOCK_COLORS = [(220, 50, 50), (50, 220, 50), (50, 50, 220), (220, 120, 50)]
    BLOCK_VALUES = [10, 20, 30, 40] # Corresponds to BLOCK_COLORS

    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 8
    BALL_SPEED = 7
    MAX_BOUNCE_ANGLE_SCALE = 1.2

    MAX_STEPS = 2500
    INITIAL_LIVES = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_on_paddle = None
        self.blocks = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.combo_hits = None
        self.combo_timer = None
        self.stuck_counter = 0
        self.np_random = None
        
        # Initialize state variables via reset
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()
        
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_on_paddle = True
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

        self.blocks = []
        num_cols, num_rows = 10, 5
        block_width = self.WIDTH // num_cols
        block_height = 20
        for i in range(num_rows):
            for j in range(num_cols):
                color_index = i % len(self.BLOCK_COLORS)
                block_rect = pygame.Rect(
                    j * block_width,
                    i * block_height + 50,
                    block_width,
                    block_height
                )
                self.blocks.append({
                    "rect": block_rect,
                    "color": self.BLOCK_COLORS[color_index],
                    "value": self.BLOCK_VALUES[color_index]
                })

        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.combo_hits = 0
        self.combo_timer = 0
        self.stuck_counter = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return (
                self._get_observation(),
                0,
                True,
                False,
                self._get_info(),
            )

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = -0.02  # Time penalty

        # --- Handle player input ---
        paddle_move = 0
        if movement == 3:  # Left
            paddle_move = -self.PADDLE_SPEED
        elif movement == 4:  # Right
            paddle_move = self.PADDLE_SPEED
        
        self.paddle.x += paddle_move
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH))

        if self.ball_on_paddle and space_held:
            # sound: launch_ball.wav
            self.ball_on_paddle = False
            initial_angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            self.ball_vel = pygame.Vector2(
                self.BALL_SPEED * math.sin(initial_angle),
                -self.BALL_SPEED * math.cos(initial_angle)
            )

        # --- Update game state ---
        self._update_particles()
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            self.combo_hits = 0

        if self.ball_on_paddle:
            self.ball_pos.x = self.paddle.centerx
        else:
            reward = self._update_ball(reward)

        # Reward for keeping paddle under ball
        if not self.ball_on_paddle and self.paddle.left < self.ball_pos.x < self.paddle.right:
            reward += 0.1

        self.steps += 1
        terminated = (self.lives <= 0) or (len(self.blocks) == 0) or (self.steps >= self.MAX_STEPS)
        if terminated:
            self.game_over = True
            if len(self.blocks) == 0:
                reward += 100 # Win bonus
                self.score += 1000 # Win score bonus
            elif self.lives <= 0:
                reward -= 100 # Lose penalty
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_ball(self, reward):
        self.ball_pos += self.ball_vel
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

        # Wall collisions
        if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.ball_pos.x, self.WIDTH - self.BALL_RADIUS))
            # sound: wall_bounce.wav
        if self.ball_pos.y <= self.BALL_RADIUS:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # sound: wall_bounce.wav

        # Paddle collision
        if self.ball_vel.y > 0 and ball_rect.colliderect(self.paddle):
            # sound: paddle_bounce.wav
            hit_pos_norm = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            hit_pos_norm = max(-1, min(1, hit_pos_norm)) # Clamp
            
            bounce_angle = hit_pos_norm * (math.pi / 2.5) # Max angle ~72 degrees
            self.ball_vel.x = self.BALL_SPEED * math.sin(bounce_angle) * self.MAX_BOUNCE_ANGLE_SCALE
            self.ball_vel.y = -self.BALL_SPEED * math.cos(bounce_angle)
            
            if abs(hit_pos_norm) < 0.1: # Center 10% of paddle
                reward -= 0.5 # Penalty for safe play
            
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            self.combo_hits = 0 # Reset combo on paddle hit

        # Block collisions
        hit_block = None
        for i, block in enumerate(self.blocks):
            if ball_rect.colliderect(block["rect"]):
                hit_block = i
                break
        
        if hit_block is not None:
            # sound: block_break.wav
            block_data = self.blocks.pop(hit_block)
            self._create_particles(block_data["rect"].center, block_data["color"])
            
            dx = self.ball_pos.x - block_data["rect"].centerx
            dy = self.ball_pos.y - block_data["rect"].centery
            if abs(dx / block_data["rect"].width) > abs(dy / block_data["rect"].height):
                self.ball_vel.x *= -1
            else:
                self.ball_vel.y *= -1
            
            reward += 1.0
            self.score += block_data["value"]
            
            self.combo_hits += 1
            self.combo_timer = 20 # 2/3 of a second at 30fps
            if self.combo_hits >= 3:
                reward += 5.0 # Combo bonus
                self.score += 50 # Combo score bonus
        
        # Anti-softlock
        if abs(self.ball_vel.y) < 0.1:
            self.stuck_counter += 1
            if self.stuck_counter > 60: # Stuck for 2 seconds
                self.ball_vel.y += self.np_random.choice([-0.5, 0.5])
                self.stuck_counter = 0
        else:
            self.stuck_counter = 0

        # Ball lost
        if self.ball_pos.y > self.HEIGHT:
            # sound: lose_life.wav
            self.lives -= 1
            self.ball_on_paddle = True
            self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
            self.ball_vel = pygame.Vector2(0, 0)
            self.combo_hits = 0
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        # Render all game elements
        self._render_game()
        # Render UI overlay
        self._render_ui()
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), block["rect"], 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball with glow effect
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        for i in range(4, 0, -1):
            alpha = 100 - i * 20
            color = (*self.COLOR_BALL, alpha)
            pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + i, color)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = p["color"]
            bright_color = (min(255,color[0]+alpha), min(255,color[1]+alpha), min(255,color[2]+alpha))
            pygame.draw.circle(self.screen, bright_color, (int(p["pos"].x), int(p["pos"].y)), 2)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            life_rect = pygame.Rect(self.WIDTH - 30 - i * 25, 15, 20, 5)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_rect, border_radius=2)

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if len(self.blocks) == 0 else "GAME OVER"
            end_text = self.font_main.render(msg, True, self.COLOR_BALL)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks),
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

# This block allows the game to be run directly for testing and visualization
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # The observation is already a rendered frame, so we just need to display it
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset(seed=random.randint(0, 10000))
            terminated = False

        clock.tick(30)
        
    env.close()