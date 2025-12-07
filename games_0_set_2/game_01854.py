import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# --- Helper Classes ---

class Bat:
    """Represents a single bat enemy."""
    def __init__(self, np_random, screen_width, screen_height):
        self.np_random = np_random
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.reset()
        self.wing_flap_angle = self.np_random.uniform(0, 2 * math.pi)

    def reset(self, side="right"):
        """Resets the bat's properties to a new random state."""
        self.y_base = self.np_random.uniform(40, self.screen_height - 40)
        self.amplitude = self.np_random.uniform(20, 80)
        self.frequency = self.np_random.uniform(0.008, 0.02)
        self.speed = self.np_random.uniform(1.5, 3.5)
        self.phase_offset = self.np_random.uniform(0, 2 * math.pi)
        
        if side == "right":
            self.x = self.screen_width + self.np_random.uniform(20, 200)
        else: # Start from left, for initial setup
            self.x = self.np_random.uniform(self.screen_width * 0.4, self.screen_width * 0.8)

        self.y = self.y_base + self.amplitude * math.sin(self.frequency * self.x + self.phase_offset)
        self.rect = pygame.Rect(self.x, self.y, 24, 12)
        self.color = (139, 0, 0) # Dark Red

    def update(self):
        """Updates the bat's position and animation."""
        self.x -= self.speed
        self.y = self.y_base + self.amplitude * math.sin(self.frequency * self.x + self.phase_offset)
        self.rect.center = (int(self.x), int(self.y))
        self.wing_flap_angle += 0.4 # Controls wing flap speed

        if self.x < -30:
            self.reset()

    def draw(self, surface):
        """Draws the bat on the given surface."""
        wing_y_offset = abs(math.sin(self.wing_flap_angle)) * 8
        cx, cy = self.rect.center
        
        # Body
        body_points = [(cx-3, cy-3), (cx+3, cy-3), (cx, cy+4)]
        pygame.gfxdraw.aapolygon(surface, body_points, self.color)
        pygame.gfxdraw.filled_polygon(surface, body_points, self.color)

        # Wings
        wing1 = [(cx - 2, cy), (cx - 12, cy - wing_y_offset), (cx - 10, cy + 5)]
        wing2 = [(cx + 2, cy), (cx + 12, cy - wing_y_offset), (cx + 10, cy + 5)]
        pygame.gfxdraw.aapolygon(surface, wing1, self.color)
        pygame.gfxdraw.filled_polygon(surface, wing1, self.color)
        pygame.gfxdraw.aapolygon(surface, wing2, self.color)
        pygame.gfxdraw.filled_polygon(surface, wing2, self.color)

class Particle:
    """Represents a single particle for effects."""
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.vx = np_random.uniform(-2, 2)
        self.vy = np_random.uniform(-3, 3)
        self.lifespan = np_random.integers(15, 30)
        self.color = color
        self.radius = np_random.uniform(2, 5)

    def update(self):
        """Updates particle position and lifespan."""
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius *= 0.95
        self.vx *= 0.98
        self.vy *= 0.98

    def draw(self, surface):
        """Draws the particle."""
        if self.lifespan > 0 and self.radius > 0.5:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.radius))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑/↓ to move the paddle. Reflect the ball to score. Avoid the bats."
    )

    game_description = (
        "Reflect the spectral pong ball to score points while dodging menacing bats in a haunted mansion."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 12, 80
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 8
        self.MAX_SCORE = 10
        self.MAX_HITS = 3
        self.MAX_STEPS = 1500 # Increased for longer rallies
        
        # --- Colors ---
        self.COLOR_BG = (25, 15, 40)
        self.COLOR_WALL = (45, 35, 60)
        self.COLOR_PADDLE = (173, 216, 230) # Pale Blue
        self.COLOR_BALL = (57, 255, 20) # Neon Green
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_RISKY = (255, 255, 0) # Yellow for risky bonus text
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- Game State Initialization ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.bat_hits = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_rect = None
        self.bats = []
        self.particles = []
        self.background_elements = {}
        self.last_paddle_y = self.HEIGHT / 2
        self.popup_texts = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.bat_hits = 0
        self.game_over = False
        
        # Player paddle
        self.paddle = pygame.Rect(30, self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.last_paddle_y = self.paddle.y
        
        # Ball
        self._reset_ball()
        
        # Bats
        self.bats = [Bat(self.np_random, self.WIDTH, self.HEIGHT) for _ in range(4)]
        for bat in self.bats:
            bat.reset(side="left") # Start bats on screen

        # Effects
        self.particles.clear()
        self.popup_texts.clear()
        
        # Generate static background elements for the episode
        self._generate_background()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _reset_ball(self):
        """Resets the ball's position and velocity."""
        y_start = self.np_random.uniform(self.BALL_RADIUS + 20, self.HEIGHT - self.BALL_RADIUS - 20)
        self.ball_pos = [self.WIDTH * 0.75, y_start]
        angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
        speed = self.np_random.uniform(4.5, 5.5)
        self.ball_vel = [-speed * math.cos(angle), speed * math.sin(angle)]
        self.ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

    def _generate_background(self):
        """Creates a set of static decorations for the haunted mansion backdrop."""
        self.background_elements['paintings'] = []
        for _ in range(self.np_random.integers(2, 5)):
            w, h = self.np_random.integers(30, 80), self.np_random.integers(40, 100)
            x = self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH - w - 20)
            y = self.np_random.uniform(40, self.HEIGHT - h - 40)
            self.background_elements['paintings'].append(pygame.Rect(x, y, w, h))

        self.background_elements['candles'] = []
        for _ in range(self.np_random.integers(5, 10)):
            x = self.np_random.uniform(20, self.WIDTH - 20)
            y = self.np_random.uniform(20, self.HEIGHT - 20)
            self.background_elements['candles'].append({'pos': (x, y), 'base_radius': self.np_random.uniform(2, 4)})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1
        
        # --- Action Handling ---
        movement = action[0]
        self.last_paddle_y = self.paddle.y
        if movement == 1: # Up
            self.paddle.y -= self.PADDLE_SPEED
        elif movement == 2: # Down
            self.paddle.y += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.y = max(10, min(self.paddle.y, self.HEIGHT - 10 - self.PADDLE_HEIGHT))
        
        # --- Reward for paddle movement ---
        is_moving = abs(self.paddle.y - self.last_paddle_y) > 0.1
        is_at_edge = self.paddle.y <= 10 or self.paddle.y >= self.HEIGHT - 10 - self.PADDLE_HEIGHT
        if is_at_edge:
            reward -= 0.02 # Small penalty for hugging edges

        # Penalty for not moving when ball is approaching
        if not is_moving and self.ball_vel[0] < 0 and self.ball_pos[0] < self.WIDTH / 2:
            reward -= 0.02
        
        # --- Game Logic Update ---
        self._update_ball()
        self._update_bats()
        self._update_particles()
        self._update_popups()

        # --- Collision Detection ---
        reward += self._handle_collisions()

        # Continuous reward for keeping ball in play
        reward += 0.01

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.score >= self.MAX_SCORE:
            reward += 10.0
            terminated = True
            self.game_over = True
            self._create_popup("VICTORY!", self.COLOR_BALL, 120)
        elif self.bat_hits >= self.MAX_HITS:
            reward -= 10.0
            terminated = True
            self.game_over = True
            self._create_popup("DEFEATED", (255, 0, 0), 120)
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            self.game_over = True
        
        terminated = terminated or truncated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        self.ball_rect.center = (int(self.ball_pos[0]), int(self.ball_pos[1]))

    def _update_bats(self):
        for bat in self.bats:
            bat.update()

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _update_popups(self):
        self.popup_texts = [p for p in self.popup_texts if p['lifespan'] > 0]
        for p in self.popup_texts:
            p['lifespan'] -= 1
            p['y'] -= 0.5 # Float upwards

    def _handle_collisions(self):
        reward = 0.0
        
        # Ball vs Walls
        if self.ball_pos[1] <= self.BALL_RADIUS or self.ball_pos[1] >= self.HEIGHT - self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = max(self.BALL_RADIUS, min(self.ball_pos[1], self.HEIGHT - self.BALL_RADIUS))
            self._create_particles(self.ball_pos[0], self.ball_pos[1], self.COLOR_BALL, 5) # sfx: wall_thud
        if self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            self._create_particles(self.ball_pos[0], self.ball_pos[1], self.COLOR_BALL, 5) # sfx: wall_thud
        
        # Ball miss
        if self.ball_pos[0] < self.BALL_RADIUS:
            self._reset_ball()
            # No direct reward change, but lack of +1 return reward is the penalty

        # Ball vs Paddle
        if self.paddle.colliderect(self.ball_rect) and self.ball_vel[0] < 0:
            reward += 1.0
            self.score += 1
            
            # Change ball velocity based on impact point
            offset = (self.paddle.centery - self.ball_pos[1]) / (self.PADDLE_HEIGHT / 2)
            self.ball_vel[0] *= -1.05 # Speed up slightly
            self.ball_vel[1] = -offset * 6
            self.ball_vel[0] = min(self.ball_vel[0], 12) # Cap max speed

            # Risky play check
            is_risky = False
            for bat in self.bats:
                dist = math.hypot(bat.rect.centerx - self.ball_rect.centerx, bat.rect.centery - self.ball_rect.centery)
                if dist < 120:
                    is_risky = True
                    break
            
            if is_risky:
                reward += 2.0
                self._create_popup("RISKY! +2", self.COLOR_RISKY, 30, (self.paddle.right + 10, self.paddle.centery))
                self._create_particles(self.paddle.right, self.ball_pos[1], self.COLOR_RISKY, 25) # sfx: risky_hit
            else:
                self._create_particles(self.paddle.right, self.ball_pos[1], self.COLOR_PADDLE, 15) # sfx: paddle_hit

        # Paddle vs Bats
        for bat in self.bats:
            if self.paddle.colliderect(bat.rect):
                reward -= 5.0
                self.bat_hits += 1
                bat.reset()
                self._create_particles(self.paddle.centerx, self.paddle.centery, (255, 0, 0), 30) # sfx: player_hit
                if self.bat_hits < self.MAX_HITS:
                    self._create_popup("-1 LIFE", (255, 0, 0), 60)
        
        return reward

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, self.np_random))

    def _create_popup(self, text, color, lifespan, pos=None):
        if pos is None:
            pos = (self.WIDTH // 2, self.HEIGHT // 2)
        self.popup_texts.append({
            'text': text, 'color': color, 'lifespan': lifespan, 
            'x': pos[0], 'y': pos[1], 'alpha': 255
        })

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "bat_hits": self.bat_hits}

    def _render_frame(self):
        """Main rendering function."""
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()

    def _render_background(self):
        # Wall borders
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, 10))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.HEIGHT - 10, self.WIDTH, 10))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - 10, 0, 10, self.HEIGHT))
        
        # Paintings
        for rect in self.background_elements['paintings']:
            pygame.draw.rect(self.screen, (10, 5, 15), rect) # Dark frame
            pygame.draw.rect(self.screen, (35, 25, 50), rect.inflate(-8, -8)) # Canvas

        # Candles
        for candle in self.background_elements['candles']:
            pos = candle['pos']
            base_radius = candle['base_radius']
            flicker = self.np_random.uniform(0.8, 1.2)
            radius = base_radius * flicker
            
            # Glow effect
            glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            glow_color_val = 150 + self.np_random.integers(-20, 20)
            glow_color = (min(255, glow_color_val + 50), min(255, glow_color_val), 50, 30)
            pygame.draw.circle(glow_surf, glow_color, (radius * 2, radius * 2), radius * 2)
            self.screen.blit(glow_surf, (pos[0] - radius * 2, pos[1] - radius * 2), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Flame
            flame_color = (255, 220, 100)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), flame_color)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(radius), flame_color)

    def _render_game_elements(self):
        # Particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Bats
        for bat in self.bats:
            bat.draw(self.screen)
            
        # Ball glow
        glow_surf = pygame.Surface((self.BALL_RADIUS * 4, self.BALL_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_BALL, 40), (self.BALL_RADIUS * 2, self.BALL_RADIUS * 2), self.BALL_RADIUS * 2)
        self.screen.blit(glow_surf, (self.ball_rect.centerx - self.BALL_RADIUS * 2, self.ball_rect.centery - self.BALL_RADIUS * 2), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, self.ball_rect.centerx, self.ball_rect.centery, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, self.ball_rect.centerx, self.ball_rect.centery, self.BALL_RADIUS, self.COLOR_BALL)

        # Paddle
        paddle_surf = pygame.Surface((self.PADDLE_WIDTH, self.PADDLE_HEIGHT), pygame.SRCALPHA)
        paddle_surf.fill((*self.COLOR_PADDLE, 150))
        # Add a subtle energy core
        core_rect = pygame.Rect(self.PADDLE_WIDTH//2 - 2, 0, 4, self.PADDLE_HEIGHT)
        pygame.draw.rect(paddle_surf, (*self.COLOR_PADDLE, 255), core_rect)
        self.screen.blit(paddle_surf, self.paddle.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Bat hits (lives)
        for i in range(self.MAX_HITS - self.bat_hits):
            self._draw_skull(self.screen, self.WIDTH - 30 - i * 25, 35)

        # Popup texts
        for p in self.popup_texts:
            alpha = max(0, min(255, p['lifespan'] * 5))
            font = self.font_large if "VICTORY" in p['text'] or "DEFEATED" in p['text'] else self.font_small
            text_surf = font.render(p['text'], True, p['color'])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(int(p['x']), int(p['y'])))
            self.screen.blit(text_surf, text_rect)

    def _draw_skull(self, surface, x, y):
        color = (220, 220, 220)
        # Main head
        pygame.gfxdraw.filled_circle(surface, x, y, 8, color)
        pygame.gfxdraw.aacircle(surface, x, y, 8, color)
        # Jaw
        pygame.draw.rect(surface, color, (x - 4, y + 5, 8, 4))
        # Eyes
        pygame.draw.circle(surface, (0, 0, 0), (x - 3, y - 1), 2)
        pygame.draw.circle(surface, (0, 0, 0), (x + 3, y - 1), 2)
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly.
    # You might need to comment out the `os.environ` line at the top
    # of the file for this to work on your system.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The dummy driver does not support display, so we can't create a window.
    # To play, comment out `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")`
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Spectral Pong")
    except pygame.error as e:
        print(f"Could not create display: {e}")
        print("To play interactively, comment out the SDL_VIDEODRIVER line at the top of the file.")
        env.close()
        exit()

    terminated = False
    truncated = False
    clock = pygame.time.Clock()
    
    # --- Human Controls ---
    # 0=none, 1=up, 2=down
    movement_action = 0 
    
    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        else:
            movement_action = 0
            
        # The action space is MultiDiscrete, but we only use the first part for human play
        action = [movement_action, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(60) # Run at 60 FPS
        
    env.close()