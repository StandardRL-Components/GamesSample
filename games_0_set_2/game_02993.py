
# Generated: 2025-08-28T06:39:51.982375
# Source Brief: brief_02993.md
# Brief Index: 2993

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
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
        "A visually vibrant top-down Breakout clone where risk-taking is rewarded. Clear all the neon blocks to win, but don't lose the ball!"
    )

    # Frames auto-advance at a fixed rate for smooth real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Visual Design ---
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_GRID = (30, 20, 70)
        self.COLOR_PADDLE = (0, 255, 255) # Cyan
        self.COLOR_BALL = (255, 255, 255) # White
        self.COLOR_BALL_GLOW = (200, 200, 255)
        self.BLOCK_COLORS = [
            (255, 0, 128),   # Hot Pink
            (0, 255, 0),     # Lime Green
            (255, 128, 0),   # Bright Orange
            (0, 128, 255),   # Sky Blue
            (255, 255, 0),   # Yellow
            (128, 0, 255),   # Purple
            (255, 50, 50)    # Red
        ]
        self.FONT_UI = pygame.font.Font(None, 28)
        self.FONT_MSG = pygame.font.Font(None, 50)

        # --- Game Mechanics ---
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 7.0
        self.PADDLE_BOUNCE_FACTOR = 1.2 # How much paddle edges affect ball angle
        
        self.BLOCK_ROWS, self.BLOCK_COLS = 7, 10
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 58, 20
        self.BLOCK_SPACING = 6
        self.BLOCK_AREA_TOP = 50

        self.MAX_LIVES = 2
        self.MAX_STEPS = 2000
        
        # --- State Variables ---
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = False
        self.blocks = []
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.ball_pos_history = deque(maxlen=100)
        
        # Initialize state variables
        self.reset()

        # Validate implementation after full initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = self.MAX_LIVES
        
        # Paddle
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle_rect = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Ball
        self.ball_launched = False
        self.ball_pos = pygame.Vector2(self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

        # Blocks
        self.blocks = []
        total_block_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - total_block_width) / 2
        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_AREA_TOP + r * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                color = self.BLOCK_COLORS[r]
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({'rect': block_rect, 'color': color})
        
        # Effects and tracking
        self.particles = []
        self.ball_pos_history.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1
        
        if movement == 3: # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle_rect.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle_rect.x))

        # Launch ball
        if space_held and not self.ball_launched:
            self.ball_launched = True
            angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upward cone
            self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED
            # sound: ball_launch.wav

        # --- Game Logic Update ---
        reward = -0.02 # Time penalty
        
        if not self.ball_launched:
            # Ball follows paddle
            self.ball_pos.x = self.paddle_rect.centerx
        else:
            # Move ball
            self.ball_pos += self.ball_vel

        # --- Collision Detection ---
        blocks_hit_this_frame = self._handle_collisions()
        
        # Reward for breaking blocks
        if blocks_hit_this_frame > 0:
            reward += blocks_hit_this_frame * 1.0
            if blocks_hit_this_frame >= 3:
                reward += 5.0 # Multi-hit bonus
            self.score += blocks_hit_this_frame

        # --- Ball Out of Bounds ---
        if self.ball_pos.y > self.HEIGHT:
            self.lives -= 1
            reward -= 5.0
            self.ball_launched = False
            # sound: lose_life.wav
            if self.lives < 0:
                self.game_over = True
                self.win = False
                reward -= 50.0
            else:
                # Reset ball on paddle
                self.ball_pos = pygame.Vector2(self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS)
                self.ball_vel = pygame.Vector2(0, 0)

        # --- Update Particles ---
        self._update_particles()
        
        # --- Anti-softlock ---
        self._check_softlock()

        # --- Termination Conditions ---
        self.steps += 1
        terminated = False
        if not self.blocks: # Win condition
            self.game_over = True
            self.win = True
            terminated = True
            reward += 50.0
        elif self.game_over: # Lose condition
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_collisions(self):
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            # sound: wall_bounce.wav
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = max(self.BALL_RADIUS, self.ball_pos.y)
            # sound: wall_bounce.wav

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect):
            # Ensure ball is above paddle to prevent it getting stuck
            if self.ball_pos.y < self.paddle_rect.centery:
                self.ball_pos.y = self.paddle_rect.top - self.BALL_RADIUS
                
                # Calculate bounce angle based on hit position
                offset = self.ball_pos.x - self.paddle_rect.centerx
                normalized_offset = offset / (self.PADDLE_WIDTH / 2)
                
                new_vel_x = normalized_offset * self.BALL_SPEED * self.PADDLE_BOUNCE_FACTOR
                
                # Ensure speed is constant
                if abs(new_vel_x) >= self.BALL_SPEED:
                    new_vel_x = math.copysign(self.BALL_SPEED * 0.9, new_vel_x)
                
                new_vel_y = -math.sqrt(max(0, self.BALL_SPEED**2 - new_vel_x**2))
                
                self.ball_vel = pygame.Vector2(new_vel_x, new_vel_y)
                # sound: paddle_bounce.wav

        # Block collisions
        blocks_hit_this_frame = 0
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                self._create_particles(block['rect'].center, block['color'])
                self.blocks.remove(block)
                blocks_hit_this_frame += 1
                
                # Determine bounce direction
                # A simple method: check which side of the block the ball is closest to
                dx = self.ball_pos.x - block['rect'].centerx
                dy = self.ball_pos.y - block['rect'].centery
                
                if abs(dx / block['rect'].width) > abs(dy / block['rect'].height):
                    self.ball_vel.x *= -1
                else:
                    self.ball_vel.y *= -1
                # sound: block_break.wav
                break # Only handle one block collision per frame for simplicity
        
        return blocks_hit_this_frame

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            particle = {
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 5),
                'color': color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _check_softlock(self):
        if self.ball_launched:
            self.ball_pos_history.append(self.ball_pos.copy())
            if len(self.ball_pos_history) == self.ball_pos_history.maxlen:
                positions = np.array(self.ball_pos_history)
                std_dev = np.std(positions, axis=0)
                if np.all(std_dev < 1.0): # If ball is stuck in a small area
                    nudge = pygame.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5))
                    self.ball_vel = (self.ball_vel + nudge).normalize() * self.BALL_SPEED

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in block['color']), block['rect'], 2) # Highlight

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)

        # Ball with glow
        ball_x, ball_y = int(self.ball_pos.x), int(self.ball_pos.y)
        glow_radius = int(self.BALL_RADIUS * 1.8)
        
        # Use gfxdraw for antialiased circles
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, glow_radius, (*self.COLOR_BALL_GLOW, 50))
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

        # Game Over / Win Message
        if self.game_over:
            msg_text = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 128) if self.win else (255, 0, 100)
            text_surf = self.FONT_MSG.render(msg_text, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.FONT_UI.render(f"LIVES: {max(0, self.lives)}", True, (255, 255, 255))
        lives_rect = lives_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(lives_text, lives_rect)

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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Setup a window to display the game
    pygame.display.set_caption("Breakout Neon")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Not used in this game
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over. Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()