
# Generated: 2025-08-27T21:42:40.089556
# Source Brief: brief_02880.md
# Brief Index: 2880

        
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
        "A retro-arcade block breaker. Clear all blocks to advance. Passing the ball through the yellow multiplier zone doubles your score for the next hit. You have 3 balls per game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (220, 220, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (200, 200, 220)
    COLOR_MULTIPLIER_ZONE = (255, 220, 50, 50) # RGBA for transparency

    BLOCK_COLORS = {
        10: (200, 50, 50),   # Red
        20: (50, 200, 50),   # Green
        30: (50, 100, 200),  # Blue
    }
    
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8.0
    
    BALL_RADIUS = 7
    BASE_BALL_SPEED = 4.0
    
    MAX_STAGES = 3
    MAX_STEPS = 3000 # Increased from 1000 to allow for longer games

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # State variables are initialized in reset()
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.multiplier_zone = None
        self.ball_in_multiplier_zone = None
        self.particles = None
        self.steps = None
        self.score = None
        self.balls_left = None
        self.stage = None
        self.game_over = None
        self.prev_space_held = None
        self.current_ball_speed = None
        
        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def _setup_stage(self):
        self.blocks = []
        self.particles = []
        self.ball_in_multiplier_zone = False

        # Reset ball to paddle
        self._reset_ball()

        # Difficulty scaling
        self.current_ball_speed = self.BASE_BALL_SPEED * (1 + (self.stage - 1) * 0.1)

        # Multiplier Zone
        zone_height = 80
        if self.stage == 1:
            self.multiplier_zone = pygame.Rect(0, 150, self.SCREEN_WIDTH, zone_height)
        elif self.stage == 2:
            self.multiplier_zone = pygame.Rect(100, 130, self.SCREEN_WIDTH - 200, zone_height - 20)
        else: # Stage 3
            self.multiplier_zone = pygame.Rect(self.SCREEN_WIDTH / 2 - 150, 180, 300, zone_height - 40)

        # Block layout
        block_width = 58
        block_height = 20
        num_cols = 10
        num_rows = 5 + (self.stage - 1) # 5, 6, 7 rows
        
        for r in range(num_rows):
            for c in range(num_cols):
                point_val = list(self.BLOCK_COLORS.keys())[r % len(self.BLOCK_COLORS)]
                color = self.BLOCK_COLORS[point_val]
                
                x = c * (block_width + 2) + 20
                y = r * (block_height + 2) + 50
                
                # Create some gaps for stage variety
                if (self.stage == 2 and (c == 4 or c == 5)) or \
                   (self.stage == 3 and (r % 2 == 0 and c % 2 == 0)):
                    continue

                self.blocks.append({
                    "rect": pygame.Rect(x, y, block_width, block_height),
                    "color": color,
                    "points": point_val
                })

    def _reset_ball(self):
        self.ball_launched = False
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top - 5
        self.ball_vel = [0, 0]
        self.ball_in_multiplier_zone = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.balls_left = 3
        self.stage = 1
        self.game_over = False
        self.prev_space_held = False
        
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1
        
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward -= 0.02 # Small penalty for movement to encourage efficiency
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            reward -= 0.02

        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.SCREEN_WIDTH, self.paddle.right)

        # Launch ball on space press (rising edge)
        if space_held and not self.prev_space_held and not self.ball_launched:
            self.ball_launched = True
            # sfx: ball_launch.wav
            initial_dx = self.np_random.uniform(-0.5, 0.5)
            self.ball_vel = [initial_dx, -1]
            # Normalize velocity
            norm = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel = [v / norm * self.current_ball_speed for v in self.ball_vel]

        self.prev_space_held = space_held

        # --- Game Logic ---
        if self.ball_launched:
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]
        else: # Ball attached to paddle
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top - 5

        # Ball collisions
        # Walls
        if self.ball.left <= 0 or self.ball.right >= self.SCREEN_WIDTH:
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.SCREEN_WIDTH, self.ball.right)
            self.ball_vel[0] *= -1
            # sfx: wall_bounce.wav
        if self.ball.top <= 0:
            self.ball.top = 0
            self.ball_vel[1] *= -1
            # sfx: wall_bounce.wav

        # Paddle
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top - 1
            self.ball_vel[1] *= -1
            
            # Add "spin" based on hit location
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.0
            
            # Normalize and re-apply speed
            norm = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            self.ball_vel = [v / norm * self.current_ball_speed for v in self.ball_vel]
            
            self.ball_in_multiplier_zone = False # Multiplier is consumed on paddle hit
            # sfx: paddle_hit.wav

        # Blocks
        hit_block = None
        for block in self.blocks:
            if self.ball.colliderect(block["rect"]):
                hit_block = block
                # sfx: block_break.wav
                
                # Determine bounce direction
                overlap = self.ball.clip(block["rect"])
                if overlap.width < overlap.height:
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[1] *= -1
                
                break
        
        if hit_block:
            points = hit_block["points"]
            base_reward = 1.0
            if self.ball_in_multiplier_zone:
                points *= 2
                base_reward = 2.0
                self.ball_in_multiplier_zone = False
            
            self.score += points
            reward += base_reward
            self._create_particles(hit_block["rect"].center, hit_block["color"])
            self.blocks.remove(hit_block)

        # Multiplier Zone Check
        if self.ball.colliderect(self.multiplier_zone):
            if not self.ball_in_multiplier_zone:
                # sfx: enter_multiplier_zone.wav
                self.ball_in_multiplier_zone = True

        # Update particles
        self._update_particles()

        # --- State Changes & Termination ---
        terminated = False
        
        # Ball lost
        if self.ball.top >= self.SCREEN_HEIGHT:
            self.balls_left -= 1
            self._reset_ball()
            # sfx: lose_ball.wav
            if self.balls_left <= 0:
                self.game_over = True
                terminated = True
                reward -= 100 # Large penalty for losing
        
        # Stage cleared
        if not self.blocks and not self.game_over:
            self.stage += 1
            reward += 10 # Bonus for clearing a stage
            if self.stage > self.MAX_STAGES:
                self.game_over = True
                terminated = True
                reward += 100 # Large bonus for winning
            else:
                self._setup_stage()
                # sfx: stage_clear.wav
        
        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # Drag
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render multiplier zone
        zone_surf = pygame.Surface(self.multiplier_zone.size, pygame.SRCALPHA)
        zone_surf.fill(self.COLOR_MULTIPLIER_ZONE)
        self.screen.blit(zone_surf, self.multiplier_zone.topleft)
        pygame.draw.rect(self.screen, (255, 220, 50), self.multiplier_zone, 1)

        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], 1)

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 20.0))))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            # Using gfxdraw for antialiasing
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)
        
        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Render ball (with glow)
        ball_pos = (int(self.ball.centerx), int(self.ball.centery))
        glow_color = self.COLOR_MULTIPLIER_ZONE if self.ball_in_multiplier_zone else (180, 180, 255)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos[0], ball_pos[1], self.BALL_RADIUS + 3, (*glow_color[:3], 80))
        pygame.gfxdraw.aacircle(self.screen, ball_pos[0], ball_pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos[0], ball_pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))
        
        # Balls left
        ball_icon_radius = 8
        for i in range(self.balls_left):
            x = self.SCREEN_WIDTH / 2 - (self.balls_left - 1) * 15 / 2 + i * 25
            y = self.SCREEN_HEIGHT - 15
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), ball_icon_radius, self.COLOR_PADDLE)
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), ball_icon_radius, self.COLOR_PADDLE)

        # Game Over / Win Text
        if self.game_over:
            if self.stage > self.MAX_STAGES:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "balls_left": self.balls_left,
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