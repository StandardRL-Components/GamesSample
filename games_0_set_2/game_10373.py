import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:16:46.333153
# Source Brief: brief_00373.md
# Brief Index: 373
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Keep the ball bouncing to score points. Control the bounce power and horizontal movement to stay in the game as gravity shifts."
    user_guide = "Controls: Use ←→ arrow keys to move horizontally. Use ↑↓ arrow keys to adjust the power of your next bounce."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds
    WIN_SCORE = 1000
    INITIAL_LIVES = 3

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 10)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_GRAVITY_NORMAL = (0, 255, 100)
    COLOR_GRAVITY_STRONG = (255, 50, 50)
    COLOR_GRAVITY_WEAK = (100, 150, 255)
    
    # Physics & Gameplay
    GRAVITY_INTERVAL_STEPS = 5 * FPS # 5 seconds
    GRAVITY_STRENGTHS = [0.25, 0.4, 0.15] # Normal, Strong Down, Weak Down (Upward Force)
    MIN_BOUNCE_POWER = 5.0
    MAX_BOUNCE_POWER = 20.0
    BOUNCE_POWER_INCREMENT = 0.4
    HORIZONTAL_ACCEL = 0.5
    MAX_HORIZONTAL_SPEED = 5.0
    HORIZONTAL_DRAG = 0.97
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # State variables are initialized in reset()
        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]
        self.target_bounce_power = 0
        self.current_bounce_power = 0
        self.max_y_since_bounce = 0
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        self.gravity_mode = 0
        self.gravity_strength = 0
        self.gravity_timer = 0
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False
        
        self.ball_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        self.ball_vel = [random.uniform(-1, 1), 0]
        self.target_bounce_power = (self.MIN_BOUNCE_POWER + self.MAX_BOUNCE_POWER) / 2
        self.current_bounce_power = self.target_bounce_power
        self.max_y_since_bounce = self.HEIGHT

        self.gravity_mode = 0  # 0: Normal, 1: Strong, 2: Weak
        self.gravity_strength = self.GRAVITY_STRENGTHS[self.gravity_mode]
        self.gravity_timer = 0
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        self._handle_input(action)
        
        bounce_event, points_scored = self._update_ball_physics()
        self.score += points_scored
        
        if bounce_event:
            reward += 0.1  # Continuous feedback for bouncing
        if points_scored > 0:
            reward += points_scored # Event-based reward for scoring

        self._update_gravity()
        self._update_particles()
        
        if self._check_life_loss():
            self.lives -= 1
            reward -= 5  # Penalty for losing a life
            if self.lives > 0:
                self._respawn_ball()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.win:
                reward += 100 # Goal-oriented reward for winning
            elif self.steps >= self.MAX_STEPS:
                reward -= 10 # Penalty for timeout

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]

        if movement == 1: # Up
            self.target_bounce_power += self.BOUNCE_POWER_INCREMENT
        elif movement == 2: # Down
            self.target_bounce_power -= self.BOUNCE_POWER_INCREMENT
        self.target_bounce_power = np.clip(self.target_bounce_power, self.MIN_BOUNCE_POWER, self.MAX_BOUNCE_POWER)

        if movement == 3: # Left
            self.ball_vel[0] -= self.HORIZONTAL_ACCEL
        elif movement == 4: # Right
            self.ball_vel[0] += self.HORIZONTAL_ACCEL
        
        self.ball_vel[0] = np.clip(self.ball_vel[0], -self.MAX_HORIZONTAL_SPEED, self.MAX_HORIZONTAL_SPEED)
        
    def _update_ball_physics(self):
        self.ball_vel[1] += self.gravity_strength
        self.ball_vel[0] *= self.HORIZONTAL_DRAG
        
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        self.max_y_since_bounce = min(self.max_y_since_bounce, self.ball_pos[1])

        bounce_event = False
        points_scored = 0
        ball_radius = self._get_ball_radius()

        if self.ball_pos[1] >= self.HEIGHT - ball_radius:
            self.ball_pos[1] = self.HEIGHT - ball_radius
            self.ball_vel[1] = -self.current_bounce_power
            self.current_bounce_power = self.target_bounce_power
            
            # Score is proportional to the height achieved
            achieved_height = self.HEIGHT - self.max_y_since_bounce
            points_scored = max(0, int(achieved_height / 10))
            self.max_y_since_bounce = self.HEIGHT

            self._create_bounce_particles(
                [self.ball_pos[0], self.HEIGHT - ball_radius],
                achieved_height
            )
            bounce_event = True
            
        return bounce_event, points_scored

    def _update_gravity(self):
        self.gravity_timer += 1
        if self.gravity_timer >= self.GRAVITY_INTERVAL_STEPS:
            self.gravity_timer = 0
            self.gravity_mode = (self.gravity_mode + 1) % len(self.GRAVITY_STRENGTHS)
            self.gravity_strength = self.GRAVITY_STRENGTHS[self.gravity_mode]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1

    def _check_life_loss(self):
        ball_radius = self._get_ball_radius()
        return self.ball_pos[0] < -ball_radius or self.ball_pos[0] > self.WIDTH + ball_radius

    def _respawn_ball(self):
        self.ball_pos = [self.WIDTH / 2, self.HEIGHT / 4]
        self.ball_vel = [random.uniform(-1, 1), 0]
        self.max_y_since_bounce = self.HEIGHT

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.win = True
            return True
        if self.lives <= 0:
            return True
        return False

    def _get_ball_radius(self):
        return 10 + (self.target_bounce_power - self.MIN_BOUNCE_POWER) * 0.8

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # Render particles
        for p in self.particles:
            if p['radius'] > 0:
                alpha = int(255 * (p['life'] / p['initial_life']))
                color = (*p['color'], alpha)
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Render ball
        ball_radius = int(self._get_ball_radius())
        pos = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        
        # Glow effect
        glow_radius = int(ball_radius * 1.8)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Main ball
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], ball_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], ball_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (15, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.WIDTH / 2 - timer_text.get_width() / 2, 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 10))

        # Gravity Indicator
        if self.gravity_mode == 0:
            grav_text, grav_color = "NORMAL", self.COLOR_GRAVITY_NORMAL
        elif self.gravity_mode == 1:
            grav_text, grav_color = "STRONG", self.COLOR_GRAVITY_STRONG
        else:
            grav_text, grav_color = "WEAK", self.COLOR_GRAVITY_WEAK
        
        grav_label = self.font_small.render("GRAVITY:", True, self.COLOR_UI_TEXT)
        grav_value = self.font_small.render(grav_text, True, grav_color)
        self.screen.blit(grav_label, (15, self.HEIGHT - 30))
        self.screen.blit(grav_value, (15 + grav_label.get_width() + 5, self.HEIGHT - 30))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_PLAYER if self.win else self.COLOR_GRAVITY_STRONG)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_bounce_particles(self, pos, height):
        num_particles = min(30, 5 + int(height / 20))
        for _ in range(num_particles):
            angle = random.uniform(math.pi * 1.1, math.pi * 1.9)
            speed = random.uniform(1, 3 + height / 100)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.uniform(2, 5),
                'life': random.randint(20, 40),
                'initial_life': 40,
                'color': random.choice([self.COLOR_PLAYER, (255, 255, 150), (200, 200, 0)])
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "ball_pos": self.ball_pos,
            "ball_vel": self.ball_vel,
            "gravity": self.gravity_strength
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You might need to comment out os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # at the top of the file to see the window.
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    # For manual play, we need a display screen
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Bouncing Ball Environment")
    
    target_bounce_power = env.target_bounce_power
    
    while running:
        # --- Event Handling ---
        action = [0, 0, 0] # Default action: no-op
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        # --- Step the environment ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()