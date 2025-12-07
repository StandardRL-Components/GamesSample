
# Generated: 2025-08-27T17:33:04.447712
# Source Brief: brief_01566.md
# Brief Index: 1566

        
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
        "Controls: Use ← and → to move the paddle. Break all the blocks to win!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro block-breaker game. Clear the screen of blocks by bouncing the ball off your paddle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 10, 40) # Dark Blue
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    
    BLOCK_COLORS = {
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "red": (255, 0, 0),
    }
    BLOCK_VALUES = {
        "green": 10,
        "yellow": 20,
        "orange": 30,
        "red": 40,
    }
    
    # Game parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 8
    BALL_SPEED = 6.0
    MAX_STEPS = 2000
    INITIAL_BALLS = 3

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
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Etc...        
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        
        self.score = 0
        self.steps = 0
        self.balls_left = 0
        self.consecutive_hits = 0
        self.game_over = False
        self.game_won = False
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.balls_left = self.INITIAL_BALLS
        self.game_over = False
        self.game_won = False
        self.consecutive_hits = 0
        
        # Create paddle
        paddle_y = self.SCREEN_HEIGHT - 40
        paddle_x = (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        # Create blocks
        self._initialize_blocks()
        
        # Reset ball
        self._reset_ball()
        
        # Clear particles
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01 # Small penalty per step to encourage speed
        
        # 1. Handle player input
        self._handle_input(movement)
        
        # 2. Update ball position and handle collisions
        reward += self._update_ball()
        
        # 3. Update particles
        self._update_particles()
        
        # 4. Check for termination conditions
        self.steps += 1
        terminated = False
        
        if self.balls_left <= 0:
            self.game_over = True
            self.game_won = False
            terminated = True
            reward -= 100 # Penalty for losing
        
        if not self.blocks:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward += 100 # Bonus for winning
            
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_blocks()
        self._render_particles()
        self._render_paddle()
        self._render_ball()
        
        # Render UI overlay
        self._render_ui()

        # Render Game Over screen if applicable
        if self.game_over:
            self._render_game_over()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    # --- Helper methods ---

    def _handle_input(self, movement):
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
            
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.paddle.x, self.SCREEN_WIDTH - self.PADDLE_WIDTH))

    def _update_ball(self):
        reward = 0
        prev_ball_pos = self.ball_pos[:]
        
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1; ball_rect.clamp_ip(self.screen.get_rect()); self.ball_pos[0] = ball_rect.centerx
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1; ball_rect.clamp_ip(self.screen.get_rect()); self.ball_pos[1] = ball_rect.centery
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            hit_pos_norm = (self.paddle.centerx - self.ball_pos[0]) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] -= hit_pos_norm * 2.5
            self._normalize_ball_velocity()
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            self.consecutive_hits = 0 # Reset combo
            # sfx: paddle_hit

        # Block collisions
        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            if ball_rect.colliderect(block['rect']):
                # sfx: block_break
                prev_ball_rect = pygame.Rect(prev_ball_pos[0] - self.BALL_RADIUS, prev_ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
                if prev_ball_rect.centery < block['rect'].top or prev_ball_rect.centery > block['rect'].bottom: self.ball_vel[1] *= -1
                else: self.ball_vel[0] *= -1

                block_value = self.BLOCK_VALUES[block['type']]; reward += block_value; self.score += block_value
                self.consecutive_hits += 1
                if self.consecutive_hits >= 2: combo_bonus = 50; reward += combo_bonus; self.score += combo_bonus
                
                self._create_particles(block['rect'].center, self.BLOCK_COLORS[block['type']])
                self.blocks.pop(i)
                break

        # Out of bounds
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.balls_left -= 1
            self.consecutive_hits = 0
            if self.balls_left > 0: self._reset_ball() # sfx: lose_ball
            else: pass # sfx: game_over
        return reward

    def _initialize_blocks(self):
        self.blocks = []
        block_width, block_height, cols, rows = 58, 18, 10, 10
        start_x = (self.SCREEN_WIDTH - (cols * (block_width + 2))) / 2
        start_y = 50
        for r in range(rows):
            for c in range(cols):
                if r < 2: t = "red"
                elif r < 4: t = "orange"
                elif r < 6: t = "yellow"
                else: t = "green"
                self.blocks.append({
                    "rect": pygame.Rect(start_x + c * (block_width + 2), start_y + r * (block_height + 2), block_width, block_height),
                    "type": t,
                })

    def _reset_ball(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        self.ball_vel = [math.cos(angle) * self.BALL_SPEED, math.sin(angle) * self.BALL_SPEED]
        self._normalize_ball_velocity()

    def _normalize_ball_velocity(self):
        speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
        if speed == 0: self.ball_vel = [0, -self.BALL_SPEED]; return
        self.ball_vel[0] = (self.ball_vel[0] / speed) * self.BALL_SPEED
        self.ball_vel[1] = (self.ball_vel[1] / speed) * self.BALL_SPEED

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi); speed = random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "color": color, "life": random.randint(15, 30)
            })

    def _update_particles(self):
        for p in self.particles: p['pos'][0] += p['vel'][0]; p['pos'][1] += p['vel'][1]; p['vel'][1] += 0.1; p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_paddle(self): pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
    def _render_ball(self): pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL); pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
    def _render_blocks(self):
        for block in self.blocks: pygame.draw.rect(self.screen, self.BLOCK_COLORS[block['type']], block['rect'], border_radius=2)
    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0)))); size = max(1, int(p['life'] / 10))
            particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA); pygame.draw.circle(particle_surf, (*p['color'], alpha), (size, size), size)
            self.screen.blit(particle_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))
    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT); self.screen.blit(score_text, (10, 10))
        balls_text = self.font_ui.render(f"BALLS: {self.balls_left}", True, self.COLOR_UI_TEXT); self.screen.blit(balls_text, (self.SCREEN_WIDTH - balls_text.get_width() - 10, 10))
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); self.screen.blit(overlay, (0, 0))
        message = "YOU WIN!" if self.game_won else "GAME OVER"; text_color = (0, 255, 0) if self.game_won else (255, 0, 0)
        game_over_text = self.font_game_over.render(message, True, text_color); text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)); self.screen.blit(game_over_text, text_rect)
    def close(self): pygame.quit()