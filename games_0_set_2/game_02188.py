
# Generated: 2025-08-27T19:33:42.095099
# Source Brief: brief_02188.md
# Brief Index: 2188

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use ← and → to move the paddle. Break all the blocks to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro-neon block breaker. Clear stages by breaking all blocks, but don't lose the ball!"
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 40
    GAME_AREA_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_PADDLE = (0, 255, 255)
    COLOR_PADDLE_GLOW = (0, 128, 128)
    COLOR_BALL = (255, 0, 128)
    COLOR_BALL_GLOW = (128, 0, 64)
    COLOR_GRID = (20, 40, 70)
    COLOR_UI_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (0, 255, 0), (255, 255, 0), (255, 128, 0),
        (0, 128, 255), (128, 0, 255)
    ]
    BONUS_COLOR_1 = (255, 215, 0) # Gold
    BONUS_COLOR_2 = (255, 255, 100) # Light Yellow

    # Game Parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 12
    PADDLE_SPEED = 12
    BALL_RADIUS = 6
    MAX_LIVES = 3
    TOTAL_STAGES = 3
    TIME_PER_STAGE = 60  # seconds
    FPS = 30

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_main = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Initialize state variables
        self.paddle = None
        self.ball_x, self.ball_y = 0.0, 0.0
        self.ball_vel_x, self.ball_vel_y = 0.0, 0.0
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.stage = 0
        self.time_left = 0
        self.ball_base_speed = 0

        self.reset()
        self.validate_implementation()
    
    def _generate_stage(self):
        self.blocks.clear()
        
        # Increase ball speed per stage
        self.ball_base_speed = 6 + (self.stage - 1) * 0.5
        
        num_cols = 14
        num_rows = 5 + self.stage # More rows for later stages
        block_width = self.SCREEN_WIDTH // num_cols
        block_height = 20
        
        # Use a seeded random generator for deterministic layouts
        stage_rng = random.Random(self.stage)

        for r in range(num_rows):
            for c in range(num_cols):
                # Simple pattern that changes with stage
                if (r + c + self.stage) % 2 == 0:
                    x = c * block_width
                    y = r * block_height + self.UI_HEIGHT + 20
                    rect = pygame.Rect(x, y, block_width - 2, block_height - 2)
                    
                    is_bonus = stage_rng.random() < 0.15 # 15% chance for a bonus block
                    color = self.BLOCK_COLORS[stage_rng.randint(0, len(self.BLOCK_COLORS) - 1)]
                    
                    self.blocks.append({'rect': rect, 'color': color, 'bonus': is_bonus})
                    
        # Ensure at least 50 blocks are generated for the brief's requirement
        while len(self.blocks) < 50 and len(self.blocks) < num_cols * num_rows:
            r = stage_rng.randint(0, num_rows - 1)
            c = stage_rng.randint(0, num_cols - 1)
            x = c * block_width
            y = r * block_height + self.UI_HEIGHT + 20
            rect = pygame.Rect(x, y, block_width - 2, block_height - 2)
            
            # Avoid placing on top of existing block
            if not any(b['rect'].colliderect(rect) for b in self.blocks):
                is_bonus = stage_rng.random() < 0.15
                color = self.BLOCK_COLORS[stage_rng.randint(0, len(self.BLOCK_COLORS) - 1)]
                self.blocks.append({'rect': rect, 'color': color, 'bonus': is_bonus})

    def _reset_ball_and_paddle(self):
        self.paddle = pygame.Rect(
            self.SCREEN_WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.SCREEN_HEIGHT - 30,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball_x = self.paddle.centerx
        self.ball_y = self.paddle.top - self.BALL_RADIUS - 1
        
        # Start with a random upward angle
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel_x = self.ball_base_speed * math.cos(angle)
        self.ball_vel_y = -self.ball_base_speed * math.sin(angle)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.stage = 1
        self.time_left = self.TIME_PER_STAGE * self.FPS
        self.game_over = False
        self.particles.clear()
        
        self._generate_stage()
        self._reset_ball_and_paddle()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.01  # Small reward for surviving a step
        
        # --- Action Handling ---
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Keep paddle within bounds
        self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # --- Game Logic ---
        self.steps += 1
        self.time_left -= 1
        
        # Ball movement
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y
        ball_rect = pygame.Rect(self.ball_x - self.BALL_RADIUS, self.ball_y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # --- Collisions ---
        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel_x *= -1
            self.ball_x = max(self.BALL_RADIUS, min(self.SCREEN_WIDTH - self.BALL_RADIUS, self.ball_x))
            # sfx: wall_bounce

        if ball_rect.top <= self.UI_HEIGHT:
            self.ball_vel_y *= -1
            self.ball_y = self.UI_HEIGHT + self.BALL_RADIUS
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel_y > 0:
            self.ball_vel_y *= -1
            self.ball_y = self.paddle.top - self.BALL_RADIUS

            # "Game Feel" - angle depends on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel_x = self.ball_base_speed * offset * 1.5
            # Clamp horizontal velocity to prevent extreme angles
            self.ball_vel_x = max(-self.ball_base_speed, min(self.ball_base_speed, self.ball_vel_x))
            
            # Reward risky vs safe plays
            if abs(offset) > 0.5:
                reward += 0.05 # Risky play bonus
            else:
                reward -= 0.02 # Safe play penalty
            # sfx: paddle_hit

        # Block collisions
        hit_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_idx != -1:
            hit_block = self.blocks.pop(hit_idx)
            # sfx: block_break
            
            # Reward
            if hit_block['bonus']:
                self.score += 25
                reward += 2.0
            else:
                self.score += 10
                reward += 1.0

            # Create particles
            for _ in range(15):
                self.particles.append(self._create_particle(hit_block['rect'].center, hit_block['color']))

            # Bounce logic
            self.ball_vel_y *= -1

        # --- Termination Checks ---
        terminated = False
        
        # 1. Lose a life
        if ball_rect.top > self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 50
            # sfx: lose_life
            if self.lives <= 0:
                self.game_over = True
                terminated = True
            else:
                self._reset_ball_and_paddle()

        # 2. Stage Clear
        if not self.blocks:
            reward += 50
            self.stage += 1
            # sfx: stage_clear
            if self.stage > self.TOTAL_STAGES:
                self.game_over = True
                terminated = True
                reward += 100 # Bonus for winning the game
            else:
                self.time_left = self.TIME_PER_STAGE * self.FPS
                self._generate_stage()
                self._reset_ball_and_paddle()

        # 3. Time out
        if self.time_left <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
            # sfx: game_over
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _create_particle(self, pos, color):
        return {
            'x': pos[0], 'y': pos[1],
            'vx': self.np_random.uniform(-2, 2), 'vy': self.np_random.uniform(-2, 2),
            'life': self.np_random.integers(15, 30),
            'color': color,
            'radius': self.np_random.uniform(2, 5)
        }

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] > 0 and p['radius'] > 0:
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['radius']), color)
                active_particles.append(p)
        self.particles = active_particles

    def _render_game(self):
        # Draw subtle grid
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, self.UI_HEIGHT), (i, self.SCREEN_HEIGHT))
        for i in range(self.UI_HEIGHT, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw blocks
        for block in self.blocks:
            color = block['color']
            if block['bonus']:
                # Flashing effect for bonus blocks
                t = (self.steps % 20) / 20.0
                color = tuple(int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(self.BONUS_COLOR_1, self.BONUS_COLOR_2))
            pygame.draw.rect(self.screen, color, block['rect'], border_radius=3)

        # Draw paddle with glow
        glow_rect = self.paddle.inflate(8, 8)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_GLOW, glow_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=6)
        
        # Draw ball trail
        trail_pos_x, trail_pos_y = self.ball_x, self.ball_y
        for i in range(4, 0, -1):
            trail_pos_x -= self.ball_vel_x * 0.5
            trail_pos_y -= self.ball_vel_y * 0.5
            alpha = 100 - i * 20
            color = (*self.COLOR_BALL, alpha)
            radius = self.BALL_RADIUS * (1 - i / 5.0)
            pygame.gfxdraw.filled_circle(self.screen, int(trail_pos_x), int(trail_pos_y), int(radius), color)

        # Draw ball with glow
        ball_pos = (int(self.ball_x), int(self.ball_y))
        pygame.gfxdraw.filled_circle(self.screen, *ball_pos, self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, *ball_pos, self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, *ball_pos, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, *ball_pos, self.BALL_RADIUS, self.COLOR_BALL)

        # Draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # UI background
        pygame.draw.rect(self.screen, (5, 10, 20), (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.UI_HEIGHT-1), (self.SCREEN_WIDTH, self.UI_HEIGHT-1))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_str = f"{int(self.time_left / self.FPS):02d}"
        timer_text = self.font_ui.render(f"TIME: {time_str}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 10))
        
        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))
        
        # Stage
        stage_text = self.font_main.render(f"STAGE {self.stage}", True, self.COLOR_GRID)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH // 2 - stage_text.get_width() // 2, self.SCREEN_HEIGHT - 35))

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
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
            "time_left": self.time_left,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # space and shift are not used
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()