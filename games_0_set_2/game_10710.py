import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:54:55.841374
# Source Brief: brief_00710.md
# Brief Index: 710
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball to hit moving targets.
    The goal is to achieve a score of 500 within 60 seconds.

    Visual Style: Clean, minimalist, geometric shapes with vibrant colors and effects.
    Gameplay: Skill-based timing and trajectory prediction.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    - action[1]: Space button (unused)
    - action[2]: Shift button (unused)

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Control a bouncing ball to hit moving targets and score points before time runs out."
    user_guide = "Use the ←→ arrow keys to move, ↑ to jump, and ↓ to fall faster."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3600  # 60 seconds at 60 FPS (brief says 6000, but 3600 is standard for 60s)
    WIN_SCORE = 500

    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (40, 80, 120)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (200, 200, 255)
    COLOR_PLATFORM = (0, 191, 255)
    COLOR_PLATFORM_TRACK = (60, 100, 140)
    COLOR_TARGET_LOW = (0, 255, 128)
    COLOR_TARGET_HIGH = (255, 223, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # Physics
    GRAVITY = pygame.Vector2(0, 0.25)
    FRICTION = 0.99
    BALL_ACCEL = 0.5
    BALL_MAX_SPEED_X = 5
    JUMP_IMPULSE = -7.5
    PLATFORM_BOOST = -8.5
    FAST_FALL_ACCEL = 0.3
    BOUNCE_DAMPENING = 0.8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_radius = 12
        self.ball_trail = []
        self.on_ground = False
        self.targets = []
        self.platforms = []
        self.particles = []
        self.consecutive_misses = 0
        self.last_target_spawn_time = 0

        # This is not used in the final version, but we'll reset it properly
        self.prev_ball_pos = pygame.Vector2(0, 0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 10 # Start with a small score to avoid immediate game over from penalty
        self.game_over = False

        self.ball_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - self.ball_radius - 1)
        self.prev_ball_pos = self.ball_pos.copy()
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_trail = []
        self.on_ground = True
        
        self.targets = []
        self.platforms = []
        self.particles = []
        self.consecutive_misses = 0
        self.last_target_spawn_time = 0

        self._create_platforms()
        for _ in range(5):
            self._spawn_target()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, _, _ = action
        reward = 0
        
        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_ball()
        self._update_platforms()
        self._update_targets()
        self._update_particles()
        
        # --- Collisions and Scoring ---
        hit_reward = self._handle_collisions()
        reward += hit_reward

        # --- Target Management ---
        miss_penalty = self._manage_targets()
        reward += miss_penalty

        self.steps += 1
        
        # --- Termination Conditions ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            # // Win sound effect
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            # // Time's up sound effect
        elif self.score <= 0:
            reward -= 100
            self.score = 0
            terminated = True
            # // Lose sound effect
        
        if terminated:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Internal Logic Methods ---
    
    def _create_platforms(self):
        self.platforms.clear()
        self.platforms.append({
            'rect': pygame.Rect(100, 280, 120, 15),
            'start': 100, 'end': 420, 'speed': 1.5, 'axis': 'x'
        })
        self.platforms.append({
            'rect': pygame.Rect(260, 150, 120, 15),
            'start': 150, 'end': 420, 'speed': -1.0, 'axis': 'x'
        })
        self.platforms.append({
            'rect': pygame.Rect(500, 100, 15, 100),
            'start': 100, 'end': 250, 'speed': 1.2, 'axis': 'y'
        })

    def _spawn_target(self):
        if len(self.targets) >= 10:
            return
        
        is_high_value = self.np_random.random() < 0.1
        value = 100 if is_high_value else self.np_random.integers(1, 11)
        color = self.COLOR_TARGET_HIGH if is_high_value else self.COLOR_TARGET_LOW
        radius = 15 if is_high_value else 10
        
        # Ensure targets don't spawn on top of each other
        for _ in range(10): # Try 10 times to find a spot
            pos = pygame.Vector2(
                self.np_random.integers(radius, self.SCREEN_WIDTH - radius),
                self.np_random.integers(radius, self.SCREEN_HEIGHT - 100) # Keep them in upper area
            )
            if not any(pos.distance_to(t['pos']) < t['radius'] + radius + 20 for t in self.targets):
                break
        else: # If no spot found, don't spawn
            return
        
        speed = self.np_random.uniform(0.5, 1.5)
        angle = self.np_random.uniform(0, 2 * math.pi)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed

        self.targets.append({
            'pos': pos, 'vel': vel, 'radius': radius, 'value': value, 'color': color, 'spawn_time': self.steps
        })

    def _handle_input(self, movement):
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 3: # Left
            self.ball_vel.x -= self.BALL_ACCEL
        elif movement == 4: # Right
            self.ball_vel.x += self.BALL_ACCEL
        
        if movement == 1 and self.on_ground: # Up (Jump)
            self.ball_vel.y = self.JUMP_IMPULSE
            self.on_ground = False
            self._create_particles(self.ball_pos + pygame.Vector2(0, self.ball_radius), self.COLOR_BALL, 10, -self.ball_vel)
            # // Jump sound effect
        
        if movement == 2: # Down (Fast Fall)
            if not self.on_ground:
                self.ball_vel.y += self.FAST_FALL_ACCEL

    def _update_ball(self):
        # Apply physics
        self.ball_vel += self.GRAVITY
        self.ball_vel.x *= self.FRICTION
        self.ball_vel.x = max(-self.BALL_MAX_SPEED_X, min(self.BALL_MAX_SPEED_X, self.ball_vel.x))
        
        # Store previous position for collision detection
        self.prev_ball_pos = self.ball_pos.copy()
        self.ball_pos += self.ball_vel

        # Update trail
        self.ball_trail.append(self.ball_pos.copy())
        if len(self.ball_trail) > 10:
            self.ball_trail.pop(0)

        # Boundary collisions
        self.on_ground = False
        if self.ball_pos.x - self.ball_radius < 0:
            self.ball_pos.x = self.ball_radius
            self.ball_vel.x *= -self.BOUNCE_DAMPENING
        elif self.ball_pos.x + self.ball_radius > self.SCREEN_WIDTH:
            self.ball_pos.x = self.SCREEN_WIDTH - self.ball_radius
            self.ball_vel.x *= -self.BOUNCE_DAMPENING
        
        if self.ball_pos.y + self.ball_radius > self.SCREEN_HEIGHT:
            self.ball_pos.y = self.SCREEN_HEIGHT - self.ball_radius
            self.ball_vel.y *= -self.BOUNCE_DAMPENING
            self.on_ground = True
            # // Bounce sound effect
        
        if self.ball_pos.y - self.ball_radius < 0:
            self.ball_pos.y = self.ball_radius
            self.ball_vel.y *= -self.BOUNCE_DAMPENING

    def _update_platforms(self):
        for p in self.platforms:
            if p['axis'] == 'x':
                p['rect'].x += p['speed']
                if p['rect'].left <= p['start'] or p['rect'].right >= p['end']:
                    p['speed'] *= -1
                    p['rect'].x = max(p['rect'].x, p['start'])
                    p['rect'].x = min(p['rect'].x, p['end'] - p['rect'].width)
            elif p['axis'] == 'y':
                p['rect'].y += p['speed']
                if p['rect'].top <= p['start'] or p['rect'].bottom >= p['end']:
                    p['speed'] *= -1
                    p['rect'].y = max(p['rect'].y, p['start'])
                    p['rect'].y = min(p['rect'].y, p['end'] - p['rect'].height)

    def _update_targets(self):
        for t in self.targets:
            t['pos'] += t['vel']
            if t['pos'].x - t['radius'] <= 0 or t['pos'].x + t['radius'] >= self.SCREEN_WIDTH:
                t['vel'].x *= -1
            if t['pos'].y - t['radius'] <= 0 or t['pos'].y + t['radius'] >= self.SCREEN_HEIGHT - 80:
                t['vel'].y *= -1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= 1
    
    def _handle_collisions(self):
        reward = 0
        # Ball vs Platforms
        for p in self.platforms:
            # Check if ball was above platform and is now intersecting
            if (self.prev_ball_pos.y + self.ball_radius <= p['rect'].top and
                self.ball_pos.y + self.ball_radius >= p['rect'].top and
                self.ball_pos.x > p['rect'].left and
                self.ball_pos.x < p['rect'].right and
                self.ball_vel.y > 0):
                
                self.ball_pos.y = p['rect'].top - self.ball_radius
                self.ball_vel.y = self.PLATFORM_BOOST
                self.on_ground = True
                self._create_particles(self.ball_pos + pygame.Vector2(0, self.ball_radius), self.COLOR_PLATFORM, 20)
                # // Platform boost sound effect

        # Ball vs Targets
        for t in self.targets[:]:
            if self.ball_pos.distance_to(t['pos']) < self.ball_radius + t['radius']:
                self.score += t['value']
                reward += t['value'] / 10.0 # Scale reward
                self.targets.remove(t)
                self.consecutive_misses = 0
                self._create_particles(t['pos'], t['color'], 50)
                # // Target hit sound effect
        return reward

    def _manage_targets(self):
        reward = 0
        # Remove targets that live too long (missed)
        for t in self.targets[:]:
            if self.steps - t['spawn_time'] > 400: # ~6.6 seconds
                self.targets.remove(t)
                self.consecutive_misses += 1
                # // Target miss sound effect

        # Apply penalty for consecutive misses
        if self.consecutive_misses >= 3:
            penalty = min(10, int(self.score * 0.2))
            self.score -= penalty
            reward -= penalty / 10.0 # Scale reward
            self.consecutive_misses = 0
        
        # Spawn new targets periodically
        if self.steps - self.last_target_spawn_time > 60: # Every second
            self._spawn_target()
            self.last_target_spawn_time = self.steps
            
        return reward
        
    def _create_particles(self, pos, color, count, base_vel=None):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            if base_vel:
                vel += base_vel * 0.3
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    # --- Rendering Methods ---

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Platform tracks
        for p in self.platforms:
            if p['axis'] == 'x':
                pygame.draw.line(self.screen, self.COLOR_PLATFORM_TRACK, (p['start'], p['rect'].centery), (p['end'], p['rect'].centery), 1)
            else:
                pygame.draw.line(self.screen, self.COLOR_PLATFORM_TRACK, (p['rect'].centerx, p['start']), (p['rect'].centerx, p['end']), 1)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']),
                    (*p['color'], alpha)
                )

        # Ball Trail
        if self.ball_trail:
            for i, pos in enumerate(self.ball_trail):
                alpha = int(255 * (i / len(self.ball_trail)) * 0.5)
                radius = int(self.ball_radius * (i / len(self.ball_trail)))
                if radius > 1:
                    pygame.gfxdraw.filled_circle(
                        self.screen, int(pos.x), int(pos.y), radius,
                        (*self.COLOR_BALL_GLOW, alpha)
                    )

        # Platforms
        for p in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, p['rect'], border_radius=4)
        
        # Targets
        for t in self.targets:
            pygame.gfxdraw.filled_circle(self.screen, int(t['pos'].x), int(t['pos'].y), t['radius'], t['color'])
            pygame.gfxdraw.aacircle(self.screen, int(t['pos'].x), int(t['pos'].y), t['radius'], t['color'])

        # Ball Glow
        for i in range(self.ball_radius, 0, -2):
            alpha = int(80 * (1 - i / self.ball_radius))
            pygame.gfxdraw.filled_circle(
                self.screen, int(self.ball_pos.x), int(self.ball_pos.y), i,
                (*self.COLOR_BALL_GLOW, alpha)
            )
        
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.ball_radius, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (20, 20), self.font_large)

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 60)
        timer_text = f"TIME: {time_left:.1f}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH - 150, 20), self.font_large)

        # Game Over / Win message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = self.COLOR_TARGET_HIGH
            else:
                msg = "GAME OVER"
                color = (255, 50, 50)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            self._draw_text(msg, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20), self.font_large, color=color, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, center=False):
        text_surface = font.render(text, True, color)
        text_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(text_shadow, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # To run with display, remove the os.environ line
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bouncy Ball Target Shooter")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1 # up
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2 # down
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3 # left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4 # right
        
        action = [movement, 0, 0] # space and shift are unused

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # The environment handles the game over screen, so we just wait for 'r' or quit
            
        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Run at 60 FPS
        
    env.close()