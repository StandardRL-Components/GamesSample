
# Generated: 2025-08-27T17:27:03.710695
# Source Brief: brief_01538.md
# Brief Index: 1538

        
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
        "Controls: Use ← and → to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A visually stunning, procedurally generated block-breaking game. "
        "Deflect the ball to destroy all 100 blocks. Destroying blocks "
        "of different colors yields different points. The ball speeds up as you progress!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 12
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 6.0
    MAX_STEPS = 3000 # Increased from 1000 to allow more time for completion

    # --- Colors ---
    COLOR_BG_TOP = (15, 20, 40)
    COLOR_BG_BOTTOM = (30, 40, 60)
    COLOR_PADDLE = (220, 220, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 150)
    COLOR_TEXT = (200, 200, 255)
    BLOCK_COLORS = {
        1: (0, 200, 100),  # Green
        2: (100, 150, 255), # Blue
        3: (255, 100, 100)  # Red
    }
    PARTICLE_COLORS = {
        1: (50, 255, 150),
        2: (150, 200, 255),
        3: (255, 150, 150)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Internal state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.score = 0
        self.lives = 0
        self.steps = 0
        self.blocks_destroyed_count = 0
        self.game_over = False
        
        # This will be initialized in reset()
        self.np_random = None

        self.validate_implementation()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.blocks_destroyed_count = 0
        self.game_over = False
        self.particles = []
        
        # Paddle
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Ball
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=np.float64)
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)  # Start downwards
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * self.INITIAL_BALL_SPEED
        
        # Blocks
        self.blocks = []
        block_width = 58
        block_height = 20
        gap = 6
        num_cols = 10
        num_rows = 5 # Reduced rows to make the game more completable within max_steps
        start_x = (self.SCREEN_WIDTH - (num_cols * (block_width + gap) - gap)) / 2
        start_y = 50
        for i in range(num_rows):
            for j in range(num_cols):
                block_type = self.np_random.choice([1, 1, 1, 2, 2, 3]) # More common green blocks
                block_rect = pygame.Rect(
                    start_x + j * (block_width + gap),
                    start_y + i * (block_height + gap),
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": block_rect, "type": block_type, "color": self.BLOCK_COLORS[block_type]})
        self.total_blocks = len(self.blocks)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.0
        
        # --- Update Paddle ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.01
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.01
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))
        
        # --- Update Ball ---
        self.ball_pos += self.ball_vel

        # --- Ball Collisions ---
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        
        # Wall collision
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.SCREEN_HEIGHT)
            # sfx: wall_bounce

        # Paddle collision
        if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            
            # Add "spin" based on hit location
            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.5
            # sfx: paddle_hit

        # Block collision
        hit_block_idx = ball_rect.collidelist([b["rect"] for b in self.blocks])
        if hit_block_idx != -1:
            block_info = self.blocks.pop(hit_block_idx)
            block = block_info["rect"]
            
            # Determine bounce direction
            prev_ball_pos = self.ball_pos - self.ball_vel
            if (prev_ball_pos[0] < block.left or prev_ball_pos[0] > block.right):
                 self.ball_vel[0] *= -1 # Side collision
            else:
                 self.ball_vel[1] *= -1 # Top/bottom collision

            # Rewards
            reward += 0.1  # Hit reward
            reward += block_info["type"] # Points-based reward
            self.score += block_info["type"]
            self.blocks_destroyed_count += 1

            # Particles
            self._create_particles(block.center, self.PARTICLE_COLORS[block_info["type"]])
            # sfx: block_break

            # Speed up ball
            speed_increase_tiers = self.blocks_destroyed_count // (self.total_blocks // 5) # Speed up 5 times
            current_speed = self.INITIAL_BALL_SPEED * (1 + 0.1 * speed_increase_tiers)
            norm_vel = self.ball_vel / np.linalg.norm(self.ball_vel)
            self.ball_vel = norm_vel * current_speed

        # --- Update Particles ---
        self._update_particles()
        
        # --- Check for Life Lost ---
        if self.ball_pos[1] > self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 10
            # sfx: lose_life
            if self.lives > 0:
                # Reset ball
                self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 5], dtype=np.float64)
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * self.INITIAL_BALL_SPEED
            else:
                self.game_over = True

        # --- Update Game State ---
        self.steps += 1
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.game_over: # Lost all lives
            reward -= 50
            terminated = True
        elif not self.blocks: # Won
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS: # Time out
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            size = self.np_random.uniform(2, 5)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color, "size": size})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _get_observation(self):
        # Clear screen with background gradient
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), block["rect"], width=2, border_radius=3)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)
        
        # Draw ball with glow
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW + (50,))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p["lifespan"] / 40))
            color_with_alpha = p["color"] + (alpha,)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = int(p["size"])
            # Create a temporary surface for the particle to handle alpha
            particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color_with_alpha, (size, size), size)
            self.screen.blit(particle_surf, (pos[0]-size, pos[1]-size))

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            msg = "YOU WIN!" if not self.blocks else "GAME OVER"
            color = (100, 255, 100) if not self.blocks else (255, 100, 100)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,128))
            self.screen.blit(overlay, (0,0))
            
            game_over_font = pygame.font.SysFont("monospace", 72, bold=True)
            game_over_text = game_over_font.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }
    
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # Space and shift are not used

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        # --- Game Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        # Pygame uses (width, height) but numpy uses (height, width, channels)
        # We need to transpose it back for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(60) # Run at 60 FPS for smooth human play

    env.close()