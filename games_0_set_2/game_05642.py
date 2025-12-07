import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ['SDL_VIDEODRIVER'] = 'dummy'

class GameEnv(gym.Env):
    """
    A fast-paced, procedurally generated block-breaking game where risk-taking is rewarded.
    The player controls a paddle to bounce a ball, clear blocks, and score points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Short, user-facing description of the game
    game_description = (
        "A retro arcade block-breaker. Clear the screen of blocks by bouncing the ball. "
        "More vibrant blocks are tougher but give more points. Don't lose the ball!"
    )

    # Frames auto-advance at 30fps for smooth, real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed frame rate for game logic
        self.MAX_STEPS = 10000
        self.INITIAL_LIVES = 3

        # --- Colors ---
        self.COLOR_BG_TOP = (15, 20, 40)
        self.COLOR_BG_BOTTOM = (40, 30, 60)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_PADDLE_RISK = (255, 100, 100)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        self.bg_surface = self._create_gradient_background()

        # --- Game State Attributes (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.level = 0
        self.game_over = False
        
        self.paddle = None
        self.paddle_speed = 12
        self.paddle_risk_timer = 0

        self.ball_pos = None
        self.ball_vel = None
        self.ball_radius = 6
        self.base_ball_speed = 6.0
        self.ball_on_paddle = True

        self.blocks = []
        self.max_block_health = 1
        
        self.particles = []

        # Initialize state variables
        # self.reset() is called by the environment wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.level = 1
        self.game_over = False
        
        self.paddle = pygame.Rect(self.WIDTH // 2 - 50, self.HEIGHT - 30, 100, 12)
        
        self.particles = []
        
        self._reset_ball()
        self._generate_blocks()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement in [3, 4]: # Left or Right
            reward -= 0.02 # Small penalty for movement to encourage efficiency
            if movement == 3: # Left
                self.paddle.x -= self.paddle_speed
            elif movement == 4: # Right
                self.paddle.x += self.paddle_speed
            self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.paddle.width)

        # --- Launch Ball ---
        if self.ball_on_paddle and space_held:
            self.ball_on_paddle = False
            initial_angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            speed = self.base_ball_speed + (self.level - 1) * 0.05
            self.ball_vel = [math.cos(initial_angle) * speed, math.sin(initial_angle) * speed]
            
        # --- Update Game Logic ---
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        if self.paddle_risk_timer > 0:
            self.paddle_risk_timer -= 1

        # --- Check for Level Clear ---
        if not self.blocks:
            reward += 100  # Goal-oriented reward for clearing level
            self.level += 1
            self._generate_blocks()
            self._reset_ball()

        # --- Check for Termination ---
        if self.lives <= 0:
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _reset_ball(self):
        self.ball_on_paddle = True
        speed = self.base_ball_speed + (self.level - 1) * 0.05
        self.ball_vel = [0, -speed]
        self._update_ball_pos_on_paddle()

    def _update_ball_pos_on_paddle(self):
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.ball_radius]

    def _update_ball(self):
        if self.ball_on_paddle:
            self._update_ball_pos_on_paddle()
        else:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]

    def _handle_collisions(self):
        if self.ball_on_paddle:
            return 0
        
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_radius, self.ball_pos[1] - self.ball_radius, self.ball_radius * 2, self.ball_radius * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.ball_radius, self.WIDTH - self.ball_radius)
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.ball_radius, self.HEIGHT - self.ball_radius)

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.ball_radius
            
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            speed = math.hypot(*self.ball_vel)
            angle = math.acos(self.ball_vel[0] / speed) if self.ball_vel[1] < 0 else -math.acos(self.ball_vel[0] / speed)
            new_angle = angle + offset * 0.8
            new_angle = np.clip(new_angle, -math.pi * 0.9, -math.pi * 0.1)
            
            self.ball_vel[0] = math.cos(new_angle) * speed
            self.ball_vel[1] = math.sin(new_angle) * speed

            if abs(offset) > 0.8:
                reward -= 0.2
                self.paddle_risk_timer = 10

        # Block collisions
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            block = self.blocks[hit_block_idx]
            
            prev_ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_vel[0] - self.ball_radius, 
                                         self.ball_pos[1] - self.ball_vel[1] - self.ball_radius,
                                         self.ball_radius * 2, self.ball_radius * 2)
            
            if prev_ball_rect.bottom <= block['rect'].top or prev_ball_rect.top >= block['rect'].bottom:
                self.ball_vel[1] *= -1
            if prev_ball_rect.right <= block['rect'].left or prev_ball_rect.left >= block['rect'].right:
                self.ball_vel[0] *= -1

            block['health'] -= 1
            reward += 0.1
            
            if block['health'] <= 0:
                reward += 1.0 * (block['initial_health'] / self.max_block_health)
                self.score += 10 * block['initial_health']
                self._create_particles(block['rect'].center, block['initial_color'])
                self.blocks.pop(hit_block_idx)
            else:
                initial_color = pygame.Color(block['initial_color'])
                h, s, v, a = initial_color.hsva
                v = max(20, v * (block['health'] / block['initial_health']))
                block['color'].hsva = (h, s, v, a)
        
        # Ball lost
        if ball_rect.top > self.HEIGHT:
            self.lives -= 1
            reward -= 100
            if self.lives > 0:
                self._reset_ball()
        
        return reward

    def _generate_blocks(self):
        self.blocks.clear()
        density = min(0.8, 0.3 + self.level * 0.05)
        self.max_block_health = min(5, 1 + self.level // 2)
        
        base_hue = self.np_random.integers(0, 360)
        
        block_width, block_height = 40, 20
        rows = 8
        cols = self.WIDTH // block_width
        
        for r in range(rows):
            for c in range(cols):
                if self.np_random.random() < density:
                    health = self.np_random.integers(1, self.max_block_health + 1)
                    
                    color = pygame.Color(0,0,0)
                    saturation = 70 + (health / self.max_block_health) * 30
                    value = 60 + (health / self.max_block_health) * 40
                    color.hsva = (base_hue, saturation, value, 100)
                    
                    self.blocks.append({
                        'rect': pygame.Rect(c * block_width, 50 + r * block_height, block_width - 2, block_height - 2),
                        'health': health,
                        'initial_health': health,
                        'color': color,
                        'initial_color': pygame.Color(color)  # FIX: Create a copy of the color object
                    })

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)

        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 30))
            color = p['color']
            
            s = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (color.r, color.g, color.b, int(alpha)), (p['radius'], p['radius']), p['radius'])
            self.screen.blit(s, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

        # Render paddle
        paddle_color = self.COLOR_PADDLE_RISK if self.paddle_risk_timer > 0 else self.COLOR_PADDLE
        pygame.draw.rect(self.screen, paddle_color, self.paddle, border_radius=4)
        pygame.draw.rect(self.screen, (255,255,255), self.paddle.inflate(-4, -4), border_radius=3)

        # Render ball with glow
        if self.ball_pos:
            ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
            glow_color = (128, 128, 0, 100)
            pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.ball_radius + 2, glow_color)
            pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.ball_radius + 2, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.ball_radius, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.ball_radius, self.COLOR_BALL)

    def _render_ui(self):
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            content = font.render(text, True, color)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(content, pos)

        draw_text(f"SCORE: {self.score}", self.font_small, self.COLOR_TEXT, (10, 10))
        
        lives_text = "LIVES: " + "♥ " * self.lives
        draw_text(lives_text, self.font_small, self.COLOR_TEXT, (self.WIDTH - self.font_small.size(lives_text)[0] - 10, 10))
        
        level_text = f"LEVEL {self.level}"
        draw_text(level_text, self.font_small, self.COLOR_TEXT, (self.WIDTH // 2 - self.font_small.size(level_text)[0] // 2, self.HEIGHT - 25))

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            go_text = "GAME OVER"
            final_score_text = f"FINAL SCORE: {self.score}"
            
            draw_text(go_text, self.font_large, (255, 50, 50), 
                      (self.WIDTH // 2 - self.font_large.size(go_text)[0] // 2, self.HEIGHT // 2 - 50))
            draw_text(final_score_text, self.font_small, self.COLOR_TEXT,
                      (self.WIDTH // 2 - self.font_small.size(final_score_text)[0] // 2, self.HEIGHT // 2 + 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "level": self.level,
        }
        
    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(bg, color, (0, y), (self.WIDTH, y))
        return bg
        
    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # To run with a display, change 'dummy' to 'x11' (on Linux) or 'windows' (on Windows)
    os.environ['SDL_VIDEODRIVER'] = 'x11'
    
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    while not done:
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(env.FPS)

    env.close()
    print(f"Game Over! Final Info: {info}")