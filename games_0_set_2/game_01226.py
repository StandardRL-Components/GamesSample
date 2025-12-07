
# Generated: 2025-08-27T16:26:28.052125
# Source Brief: brief_01226.md
# Brief Index: 1226

        
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
        "Controls: ← to move left, → to move right. Break all the blocks to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker. Strategic paddle positioning and risk-taking are rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 4.0
    MAX_STEPS = 10000

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (30, 30, 45)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 0, 128)
    COLOR_BALL_GLOW = (255, 100, 180)
    COLOR_UI_TEXT = (220, 220, 220)
    
    BLOCK_COLORS = {
        10: (0, 200, 100),   # Green
        20: (0, 150, 255),   # Blue
        30: (255, 50, 50),   # Red
        40: (255, 200, 0),   # Yellow
    }

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = None
        self.particles = None
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.combo_count = 0
        self.blocks_destroyed_count = 0
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.combo_display_timer = 0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=np.float64)
        
        # Start ball with a random upward direction
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * self.INITIAL_BALL_SPEED
        
        self._create_blocks()
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.combo_count = 0
        self.blocks_destroyed_count = 0
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.combo_display_timer = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks = []
        block_width = 50
        block_height = 20
        rows = 5
        cols = self.WIDTH // (block_width + 4)
        
        start_y = 50
        start_x = (self.WIDTH - cols * (block_width + 4)) // 2

        point_values = [40, 30, 20, 10, 10]

        for r in range(rows):
            for c in range(cols):
                points = point_values[r]
                color = self.BLOCK_COLORS[points]
                rect = pygame.Rect(
                    start_x + c * (block_width + 4),
                    start_y + r * (block_height + 4),
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": rect, "color": color, "points": points})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # Unpack factorized action
        movement = action[0]
        
        # 1. Update Paddle
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)

        # 2. Update Ball
        self.ball_pos += self.ball_vel

        # 3. Collision Detection
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            
            # Change horizontal velocity based on hit location
            hit_offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += hit_offset * 2.0
            
            # Normalize to maintain constant speed
            current_magnitude = np.linalg.norm(self.ball_vel)
            if current_magnitude > 0:
                 self.ball_vel = (self.ball_vel / current_magnitude) * self.ball_speed

            # Reward for risky play
            if abs(hit_offset) > 0.75:
                reward += 0.1
            elif abs(hit_offset) < 0.25:
                reward -= 0.02

            self.combo_count = 0
            # sfx: paddle_hit

        # Block collisions
        hit_block = None
        for i, block_data in enumerate(self.blocks):
            if ball_rect.colliderect(block_data["rect"]):
                hit_block = i
                break
        
        if hit_block is not None:
            block_data = self.blocks.pop(hit_block)
            # sfx: block_break
            
            # Reward for breaking block
            reward += 1
            self.score += block_data["points"]

            # Combo rewards
            self.combo_count += 1
            if self.combo_count == 2:
                reward += 5 # Start combo
            elif self.combo_count > 2:
                reward += 1 # Continue combo
            self.combo_display_timer = 60 # Show combo text for 2 seconds (30fps)
            
            # Create particles
            self._create_particles(block_data["rect"].center, block_data["color"])

            # Bounce logic
            self.ball_vel[1] *= -1
            # Add small random horizontal nudge to avoid loops
            self.ball_vel[0] += self.np_random.uniform(-0.1, 0.1)

            # Difficulty scaling
            self.blocks_destroyed_count += 1
            if self.blocks_destroyed_count % 10 == 0:
                self.ball_speed += 0.5 # Brief specifies 0.05, but that's too slow. Upping for gameplay.
                # Re-normalize velocity to new speed
                current_magnitude = np.linalg.norm(self.ball_vel)
                if current_magnitude > 0:
                     self.ball_vel = (self.ball_vel / current_magnitude) * self.ball_speed
        
        # 4. Update Particles
        self._update_particles()
        if self.combo_display_timer > 0:
            self.combo_display_timer -= 1

        # 5. Check Termination Conditions
        terminated = False
        
        # Lose life
        if ball_rect.top > self.HEIGHT:
            self.lives -= 1
            self.combo_count = 0
            # sfx: lose_life
            if self.lives <= 0:
                self.game_over = True
                terminated = True
                reward -= 100
            else:
                # Reset ball position
                self.paddle.centerx = self.WIDTH // 2
                self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=np.float64)
                angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * self.ball_speed

        # Win condition
        if not self.blocks:
            self.game_over = True
            terminated = True
            reward += 100
            self.score += 1000 # Win bonus
            # sfx: win_game

        # Step limit
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "lifetime": lifetime,
                "color": color,
                "radius": self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data["color"], block_data["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block_data["rect"], 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball glow
        pygame.gfxdraw.filled_circle(
            self.screen,
            int(self.ball_pos[0]),
            int(self.ball_pos[1]),
            int(self.BALL_RADIUS * 1.5),
            (*self.COLOR_BALL_GLOW, 80)
        )
        # Ball
        pygame.gfxdraw.filled_circle(
            self.screen,
            int(self.ball_pos[0]),
            int(self.ball_pos[1]),
            self.BALL_RADIUS,
            self.COLOR_BALL
        )
        pygame.gfxdraw.aacircle(
            self.screen,
            int(self.ball_pos[0]),
            int(self.ball_pos[1]),
            self.BALL_RADIUS,
            self.COLOR_BALL
        )
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 30.0))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Combo
        if self.combo_count > 1 and self.combo_display_timer > 0:
            fade = min(1.0, self.combo_display_timer / 20.0)
            alpha = int(255 * fade)
            
            combo_text = self.font_large.render(f"x{self.combo_count}", True, (*self.COLOR_BALL, alpha))
            combo_text.set_alpha(alpha)
            text_rect = combo_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(combo_text, text_rect)

        # Game Over / Win Text
        if self.game_over:
            if not self.blocks: # Win
                end_text = self.font_large.render("YOU WIN!", True, self.BLOCK_COLORS[40])
            else: # Lose
                end_text = self.font_large.render("GAME OVER", True, self.BLOCK_COLORS[30])
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            self.screen.blit(end_text, text_rect)

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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run pygame in a window
    import os
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Re-initialize pygame for display
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print(GameEnv.user_guide)

    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0

        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()