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


# Set the SDL video driver to "dummy" for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced isometric block breaker. Clear all the blocks to win, but lose a life if the ball misses your paddle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.PADDLE_SPEED = 8
        self.BALL_SPEED = 5
        self.BLOCK_ROWS = 5
        self.BLOCK_COLS = 10
        self.INITIAL_LIVES = 3

        # Color palette
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (200, 200, 255, 60)
        self.COLOR_SHADOW = (10, 15, 25, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            "base": [(50, 200, 50), (50, 100, 200), (200, 50, 50)], # Green, Blue, Red
            "light": [(100, 220, 100), (100, 150, 220), (220, 100, 100)],
            "dark": [(30, 150, 30), (30, 70, 150), (150, 30, 30)],
        }
        self.BLOCK_VALUES = [1, 2, 3] # Corresponds to color index

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        # FIX: Initialize a display mode, even for headless, so that .convert_alpha() works.
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Isometric projection settings
        self.iso_scale = 20
        self.iso_angle = math.pi / 6
        self.iso_cos = math.cos(self.iso_angle)
        self.iso_sin = math.sin(self.iso_angle)
        self.world_offset_x = self.WIDTH / 2
        self.world_offset_y = self.HEIGHT / 4

        # Initialize state variables
        self.paddle_pos = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_on_paddle = None
        self.blocks = []
        self.particles = []
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        # The original code called reset() here, which is not standard.
        # State is initialized in reset(), so we don't need to call it twice.

    def _to_iso(self, x, y, z=0):
        """Projects 3D world coordinates to 2D screen coordinates."""
        iso_x = (x - y) * self.iso_cos
        iso_y = (x + y) * self.iso_sin - z
        return int(iso_x * self.iso_scale + self.world_offset_x), int(iso_y * self.iso_scale + self.world_offset_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.particles = []
        
        # Paddle state
        self.paddle_pos = {'x': 0, 'y': 8, 'z': 0, 'w': 3, 'd': 0.5}
        
        # Ball state
        self._reset_ball()
        
        # Block layout
        self.blocks = []
        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                color_index = (r % len(self.BLOCK_VALUES))
                block = {
                    'x': c - self.BLOCK_COLS / 2 + 0.5,
                    'y': r - 3,
                    'z': 2,
                    'w': 0.9, 'd': 0.9, 'h': 0.5,
                    'alive': True,
                    'color_index': color_index,
                    'value': self.BLOCK_VALUES[color_index]
                }
                self.blocks.append(block)

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        """Resets the ball to be on the paddle."""
        self.ball_on_paddle = True
        self.ball_pos = {'x': self.paddle_pos['x'], 'y': self.paddle_pos['y'] - 0.5, 'z': 0.5, 'r': 0.3}
        self.ball_vel = {'x': 0, 'y': 0}

    def step(self, action):
        reward = -0.02  # Small penalty per step to encourage speed
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle_pos['x'] -= self.PADDLE_SPEED / 30.0
        elif movement == 4:  # Right
            self.paddle_pos['x'] += self.PADDLE_SPEED / 30.0
        
        # Clamp paddle position
        max_x = self.BLOCK_COLS / 2 - self.paddle_pos['w'] / 2
        self.paddle_pos['x'] = max(min(self.paddle_pos['x'], max_x), -max_x)
        
        # Launch ball
        if self.ball_on_paddle and space_held:
            # sfx: ball_launch.wav
            self.ball_on_paddle = False
            initial_angle = self.np_random.uniform(-0.2, 0.2)
            self.ball_vel = {'x': math.sin(initial_angle) * self.BALL_SPEED, 'y': -math.cos(initial_angle) * self.BALL_SPEED}
            reward += 0.1 # Small reward for launching

        # --- Update Game Logic ---
        self._update_ball()
        self._update_particles()
        
        # --- Calculate Reward & Termination ---
        reward += self.frame_reward # Add rewards accumulated during ball update
        terminated = self._check_termination()
        truncated = False

        if terminated and not self.game_over:
            if self.lives <= 0:
                reward -= 100 # Penalty for losing
            elif sum(1 for b in self.blocks if b['alive']) == 0:
                reward += 100 # Bonus for winning
            self.game_over = True
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Per Gymnasium API, if truncated is True, terminated should also be True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_ball(self):
        self.frame_reward = 0
        if self.ball_on_paddle:
            self.ball_pos['x'] = self.paddle_pos['x']
            return

        # Move ball
        dt = 1.0 / 30.0
        self.ball_pos['x'] += self.ball_vel['x'] * dt
        self.ball_pos['y'] += self.ball_vel['y'] * dt

        # Wall collisions
        half_width = self.BLOCK_COLS / 2
        if self.ball_pos['x'] < -half_width or self.ball_pos['x'] > half_width:
            self.ball_vel['x'] *= -1
            self.ball_pos['x'] = max(min(self.ball_pos['x'], half_width), -half_width)
            # sfx: wall_bounce.wav
        if self.ball_pos['y'] < -4: # Top wall
            self.ball_vel['y'] *= -1
            self.ball_pos['y'] = -4
            # sfx: wall_bounce.wav

        # Paddle collision
        paddle_y_check = self.paddle_pos['y'] - self.paddle_pos['d'] / 2
        if self.ball_vel['y'] > 0 and self.ball_pos['y'] >= paddle_y_check:
            px, pw = self.paddle_pos['x'], self.paddle_pos['w']
            bx = self.ball_pos['x']
            if px - pw / 2 < bx < px + pw / 2:
                # sfx: paddle_hit.wav
                self.frame_reward += 0.1
                self.ball_vel['y'] *= -1
                
                # Influence angle based on hit position
                hit_offset = (bx - px) / (pw / 2)
                self.ball_vel['x'] += hit_offset * 2.0
                
                # Normalize speed
                speed = math.sqrt(self.ball_vel['x']**2 + self.ball_vel['y']**2)
                if speed > 0:
                    self.ball_vel['x'] = (self.ball_vel['x'] / speed) * self.BALL_SPEED
                    self.ball_vel['y'] = (self.ball_vel['y'] / speed) * self.BALL_SPEED
                
                self.ball_pos['y'] = paddle_y_check - 0.01 # Prevent sticking
            
        # Lose life
        if self.ball_pos['y'] > self.paddle_pos['y'] + 2:
            # sfx: lose_life.wav
            self.lives -= 1
            self.frame_reward -= 10
            if self.lives > 0:
                self._reset_ball()
        
        # Block collisions
        for block in self.blocks:
            if not block['alive']:
                continue
            
            # Simple AABB collision in 3D world space
            b_min_x, b_max_x = block['x'] - block['w']/2, block['x'] + block['w']/2
            b_min_y, b_max_y = block['y'] - block['d']/2, block['y'] + block['d']/2
            
            if (b_min_x < self.ball_pos['x'] < b_max_x and
                b_min_y < self.ball_pos['y'] < b_max_y):
                
                block['alive'] = False
                self.frame_reward += block['value']
                self.score += block['value'] * 10
                # sfx: block_break.wav
                self._create_particles(block)

                # Determine collision side to reflect velocity
                prev_x = self.ball_pos['x'] - self.ball_vel['x'] * (1/30.0)
                prev_y = self.ball_pos['y'] - self.ball_vel['y'] * (1/30.0)
                
                if prev_y >= b_max_y or prev_y <= b_min_y:
                    self.ball_vel['y'] *= -1
                if prev_x >= b_max_x or prev_x <= b_min_x:
                    self.ball_vel['x'] *= -1
                break

    def _create_particles(self, block):
        px, py = self._to_iso(block['x'], block['y'], block['z'])
        color = self.BLOCK_COLORS['base'][block['color_index']]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [px, py],
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color
            })
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        if self.lives <= 0:
            return True
        if all(not b['alive'] for b in self.blocks):
            return True
        return False

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
        self._draw_grid()
        self._draw_blocks()
        self._draw_paddle()
        self._draw_ball_and_shadow()
        self._draw_particles()

    def _draw_iso_cube(self, x, y, z, w, d, h, base_color, light_color, dark_color):
        """Draws a 3D cube in isometric view."""
        corners = [
            (x - w/2, y - d/2, z), (x + w/2, y - d/2, z),
            (x + w/2, y + d/2, z), (x - w/2, y + d/2, z),
            (x - w/2, y - d/2, z + h), (x + w/2, y - d/2, z + h),
            (x + w/2, y + d/2, z + h), (x - w/2, y + d/2, z + h)
        ]
        iso_corners = [self._to_iso(cx, cy, cz) for cx, cy, cz in corners]

        # Draw faces (back to front)
        # Top face
        pygame.draw.polygon(self.screen, light_color, [iso_corners[4], iso_corners[5], iso_corners[6], iso_corners[7]])
        # Left face
        pygame.draw.polygon(self.screen, dark_color, [iso_corners[0], iso_corners[3], iso_corners[7], iso_corners[4]])
        # Right face
        pygame.draw.polygon(self.screen, base_color, [iso_corners[0], iso_corners[1], iso_corners[5], iso_corners[4]])
        
        # Draw outlines for clarity
        pygame.draw.aalines(self.screen, self.COLOR_SHADOW, True, [iso_corners[4], iso_corners[5], iso_corners[6], iso_corners[7]], 0)
        pygame.draw.aalines(self.screen, self.COLOR_SHADOW, True, [iso_corners[0], iso_corners[3], iso_corners[7], iso_corners[4]], 0)
        pygame.draw.aalines(self.screen, self.COLOR_SHADOW, True, [iso_corners[0], iso_corners[1], iso_corners[5], iso_corners[4]], 0)


    def _draw_grid(self):
        for i in range(-10, 11):
            # Lines along one axis
            p1 = self._to_iso(i, -10, 0)
            p2 = self._to_iso(i, 10, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
            # Lines along other axis
            p1 = self._to_iso(-10, i, 0)
            p2 = self._to_iso(10, i, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

    def _draw_blocks(self):
        for block in sorted(self.blocks, key=lambda b: b['y'], reverse=True):
            if block['alive']:
                ci = block['color_index']
                self._draw_iso_cube(block['x'], block['y'], block['z'], block['w'], block['d'], block['h'],
                                    self.BLOCK_COLORS['base'][ci], self.BLOCK_COLORS['light'][ci], self.BLOCK_COLORS['dark'][ci])

    def _draw_paddle(self):
        p = self.paddle_pos
        self._draw_iso_cube(p['x'], p['y'], p['z'], p['w'], p['d'], 0.2,
                            (150, 150, 180), self.COLOR_PADDLE, (120, 120, 150))

    def _draw_ball_and_shadow(self):
        # Shadow
        shadow_pos = self._to_iso(self.ball_pos['x'], self.ball_pos['y'], 0)
        shadow_r = self.ball_pos['r'] * self.iso_scale
        shadow_rect = pygame.Rect(shadow_pos[0] - shadow_r, shadow_pos[1] - shadow_r / 2, shadow_r * 2, shadow_r)
        
        shadow_surf = pygame.Surface((shadow_r * 2, shadow_r)).convert_alpha()
        shadow_surf.fill((0,0,0,0))
        pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, (0, 0, shadow_r*2, shadow_r))
        self.screen.blit(shadow_surf, shadow_rect.topleft)

        # Ball
        ball_screen_pos = self._to_iso(self.ball_pos['x'], self.ball_pos['y'], self.ball_pos['z'])
        ball_r = int(self.ball_pos['r'] * self.iso_scale)
        
        # Glow effect
        glow_surf = pygame.Surface((ball_r * 4, ball_r * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (ball_r * 2, ball_r * 2), ball_r * 2)
        self.screen.blit(glow_surf, (ball_screen_pos[0] - ball_r * 2, ball_screen_pos[1] - ball_r * 2))
        
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, ball_screen_pos[0], ball_screen_pos[1], ball_r, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_screen_pos[0], ball_screen_pos[1], ball_r, self.COLOR_BALL)


    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = (*p['color'], alpha)
            size = max(1, int(5 * (p['life'] / 30.0)))
            
            surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (size,size), size)
            self.screen.blit(surf, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_large.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        lives_rect = lives_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(lives_text, lives_rect)

        # Game Over / Win message
        if self.game_over:
            message = "GAME OVER" if self.lives <= 0 else "YOU WIN!"
            
            msg_surf = self.font_large.render(message, True, self.COLOR_PADDLE)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, self.COLOR_BG, msg_rect.inflate(20, 20))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": sum(1 for b in self.blocks if b['alive']),
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly.
    # It will override the headless mode and create a real window.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "cocoa"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The environment's screen is already a display surface, so we can use it directly.
    pygame.display.set_caption("Isometric Block Breaker")
    
    done = False
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    # Game loop
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Get keyboard inputs
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Map keys to MultiDiscrete action space
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(env.screen, frame)
        pygame.display.flip()
        
        # Cap the frame rate
        env.clock.tick(30)
        
    env.close()