import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:38:50.815781
# Source Brief: brief_00546.md
# Brief Index: 546
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# --- Helper Classes for Game Entities ---

class Particle:
    def __init__(self, pos, vel, lifespan, color, radius_start, radius_end):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.radius_start = radius_start
        self.radius_end = radius_end

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.vel *= 0.98  # Damping

    def draw(self, surface):
        if self.lifespan > 0:
            life_frac = self.lifespan / self.max_lifespan
            current_radius = int(self.radius_start * life_frac + self.radius_end * (1 - life_frac))
            if current_radius > 0:
                alpha = int(255 * life_frac)
                temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, self.color + (alpha,), (current_radius, current_radius), current_radius)
                surface.blit(temp_surf, self.pos - pygame.math.Vector2(current_radius, current_radius), special_flags=pygame.BLEND_RGBA_ADD)

class Projectile:
    def __init__(self, pos, vel, size=6):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.size = size
        self.trail = deque(maxlen=10)

    def update(self):
        self.trail.append(pygame.math.Vector2(self.pos))
        self.pos += self.vel

class Enemy:
    def __init__(self, pos, wave, size=12):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * 0.5
        self.size = size
        self.base_speed = 0.5 + wave * 0.05

    def update(self, player_pos, polarity, force_multiplier):
        # Polarity force
        to_player = player_pos - self.pos
        dist_sq = to_player.length_squared()
        if dist_sq > 1: # Avoid division by zero and extreme forces
            force_mag = force_multiplier / dist_sq
            force_vec = to_player.normalize() * force_mag * polarity
            self.vel += force_vec

        # Add base random-ish movement
        self.vel += pygame.math.Vector2(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))

        # Clamp speed and apply damping
        speed = self.vel.length()
        if speed > self.base_speed * 2:
            self.vel.scale_to_length(self.base_speed * 2)
        
        self.vel *= 0.99 # Damping
        self.pos += self.vel


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A top-down arcade shooter where you control a polarity-shifting ship to attract or repel hexagonal enemies and destroy them."
    )
    user_guide = (
        "Use arrow keys to aim your reticle. Press space to fire projectiles. Press shift to switch between attract (+) and repel (-) polarity."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_POS = pygame.math.Vector2(WIDTH // 2, HEIGHT // 2)
    PLAYER_SIZE = 16
    AIM_RETICLE_DIST = 40
    AIM_SPEED = 5
    PROJECTILE_SPEED = 10
    MAX_STEPS = 1000

    # --- Colors (Neon Aesthetic) ---
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (20, 40, 70)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 128, 128)
    COLOR_ENEMY_TYPES = [(255, 100, 0), (255, 0, 100), (200, 0, 255)]
    COLOR_ENEMY_GLOW = (128, 50, 0)
    COLOR_PROJECTILE = (200, 255, 255)
    COLOR_EXPLOSION = (255, 255, 200)
    COLOR_TEXT = (220, 220, 220)
    COLOR_ATTRACT = (100, 100, 255)
    COLOR_REPEL = (255, 100, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_polarity = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False
        
        self.player_polarity = 1  # 1 for attract, -1 for repel
        self.aim_pos = pygame.math.Vector2(self.PLAYER_POS.x + self.AIM_RETICLE_DIST, self.PLAYER_POS.y)
        
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False
        
        self.player_polarity = 1
        self.aim_pos = pygame.math.Vector2(self.PLAYER_POS.x + self.AIM_RETICLE_DIST, self.PLAYER_POS.y)
        
        self.enemies.clear()
        self.projectiles.clear()
        self.particles.clear()

        self.prev_space_held = True # Prevent firing on first frame
        self.prev_shift_held = True # Prevent switching on first frame

        self._start_new_wave()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            self.steps += 1
            
            # --- Handle Actions ---
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            self._handle_input(movement, space_held, shift_held)
            
            # --- Update Game Logic ---
            reward += self._update_projectiles()
            self._update_enemies()
            self._update_particles()
            
            # --- Check Wave Completion ---
            if not self.enemies and not self.game_over:
                # sfx: wave_clear_sound
                reward += 10
                self._start_new_wave()
                for _ in range(50):
                    self._create_explosion(self.PLAYER_POS, 1, (255, 255, 100), 20)

        # --- Check Termination Conditions ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if self.game_over:
            reward = -100
        
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _handle_input(self, movement, space_held, shift_held):
        # Aiming
        if movement == 1: self.aim_pos.y -= self.AIM_SPEED
        elif movement == 2: self.aim_pos.y += self.AIM_SPEED
        elif movement == 3: self.aim_pos.x -= self.AIM_SPEED
        elif movement == 4: self.aim_pos.x += self.AIM_SPEED

        # Clamp aimer position to be around the player
        direction = self.aim_pos - self.PLAYER_POS
        if direction.length() > 0:
            self.aim_pos = self.PLAYER_POS + direction.normalize() * self.AIM_RETICLE_DIST

        # Fire projectile (on press)
        if space_held and not self.prev_space_held:
            # sfx: shoot_laser
            direction = (self.aim_pos - self.PLAYER_POS).normalize()
            vel = direction * self.PROJECTILE_SPEED
            self.projectiles.append(Projectile(pygame.math.Vector2(self.PLAYER_POS), vel))
        self.prev_space_held = space_held

        # Switch polarity (on press)
        if shift_held and not self.prev_shift_held:
            # sfx: polarity_switch
            self.player_polarity *= -1
        self.prev_shift_held = shift_held

    def _start_new_wave(self):
        self.wave += 1
        num_enemies = 2 + self.wave
        
        for _ in range(num_enemies):
            # Spawn enemies at the edges of the screen
            side = random.randint(0, 3)
            if side == 0: # top
                pos = (random.uniform(0, self.WIDTH), -30)
            elif side == 1: # bottom
                pos = (random.uniform(0, self.WIDTH), self.HEIGHT + 30)
            elif side == 2: # left
                pos = (-30, random.uniform(0, self.HEIGHT))
            else: # right
                pos = (self.WIDTH + 30, random.uniform(0, self.HEIGHT))
            self.enemies.append(Enemy(pos, self.wave))

    def _update_projectiles(self):
        hit_reward = 0
        for p in self.projectiles[:]:
            p.update()
            # Check for out of bounds
            if not self.screen.get_rect().collidepoint(p.pos):
                self.projectiles.remove(p)
                continue
            
            # Check for collision with enemies
            for e in self.enemies[:]:
                if (p.pos - e.pos).length() < p.size + e.size:
                    # sfx: enemy_explosion
                    self._create_explosion(e.pos, 30, self.COLOR_EXPLOSION, 15)
                    self.enemies.remove(e)
                    if p in self.projectiles: self.projectiles.remove(p)
                    self.score += 10
                    hit_reward += 1.0 # Per-hit reward
                    break
        return hit_reward

    def _update_enemies(self):
        for e in self.enemies:
            e.update(self.PLAYER_POS, self.player_polarity, 2000) # Polarity force
            # Check for collision with player
            if (e.pos - self.PLAYER_POS).length() < e.size + self.PLAYER_SIZE:
                if not self.game_over:
                    # sfx: player_death_explosion
                    self._create_explosion(self.PLAYER_POS, 100, (255, 50, 50), 40)
                    self.game_over = True
            
            # Boundary collision (bounce)
            if e.pos.x < e.size or e.pos.x > self.WIDTH - e.size: e.vel.x *= -1
            if e.pos.y < e.size or e.pos.y > self.HEIGHT - e.size: e.vel.y *= -1

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _create_explosion(self, position, num_particles, color, max_speed):
        for _ in range(num_particles):
            vel = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
            if vel.length() > 0:
                vel.scale_to_length(random.uniform(0.5, max_speed))
            lifespan = random.randint(15, 30)
            radius = random.uniform(1, 4)
            self.particles.append(Particle(pygame.math.Vector2(position), vel, lifespan, color, radius, 0))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_projectiles()
        self._render_enemies()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    # --- Rendering Methods ---
    
    @staticmethod
    def _draw_glowing_hexagon(surface, color, center, size, width=0):
        # Base hexagon
        points = [
            (center[0] + size * math.cos(math.pi / 180 * (60 * i + 30)),
             center[1] + size * math.sin(math.pi / 180 * (60 * i + 30)))
            for i in range(6)
        ]
        
        # Glow effect
        for i in range(5, 0, -1):
            glow_size = size + i * 1.5
            glow_alpha = 30 - i * 5
            glow_points = [
                (center[0] + glow_size * math.cos(math.pi / 180 * (60 * j + 30)),
                 center[1] + glow_size * math.sin(math.pi / 180 * (60 * j + 30)))
                for j in range(6)
            ]
            try:
                pygame.gfxdraw.aapolygon(surface, glow_points, color + (glow_alpha,))
            except TypeError:
                pygame.gfxdraw.aapolygon(surface, glow_points, (*color[:3], glow_alpha))


        # Main shape
        try:
            pygame.gfxdraw.aapolygon(surface, points, color)
            if width == 0:
                pygame.gfxdraw.filled_polygon(surface, points, color)
        except TypeError: # Handle cases where color already has alpha
            pygame.gfxdraw.aapolygon(surface, points, (*color[:3], color[3] if len(color)>3 else 255))
            if width == 0:
                pygame.gfxdraw.filled_polygon(surface, points, (*color[:3], color[3] if len(color)>3 else 255))


    def _render_background(self):
        hex_size = 40
        hex_width = hex_size * 2
        hex_height = math.sqrt(3) * hex_size
        for row in range(-1, int(self.HEIGHT / hex_height) + 2):
            for col in range(-1, int(self.WIDTH / (hex_width * 0.75)) + 1):
                x_offset = col * hex_width * 0.75
                y_offset = row * hex_height + (hex_height / 2 if col % 2 else 0)
                self._draw_glowing_hexagon(self.screen, self.COLOR_GRID, (x_offset, y_offset), hex_size, width=1)
    
    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_player(self):
        if self.game_over:
            return
            
        # Polarity field
        polarity_color = self.COLOR_ATTRACT if self.player_polarity == 1 else self.COLOR_REPEL
        radius = 250 + math.sin(self.steps * 0.1) * 10
        temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, polarity_color + (20,), (radius, radius), radius)
        self.screen.blit(temp_surf, self.PLAYER_POS - pygame.math.Vector2(radius, radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Player hexagon
        self._draw_glowing_hexagon(self.screen, self.COLOR_PLAYER, self.PLAYER_POS, self.PLAYER_SIZE)
        
        # Aiming reticle
        pygame.draw.line(self.screen, self.COLOR_PLAYER + (100,), self.PLAYER_POS, self.aim_pos, 1)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, self.aim_pos, 4, 1)

        # Polarity indicator
        polarity_text = "+" if self.player_polarity == 1 else "-"
        text_surf = self.font_polarity.render(polarity_text, True, polarity_color)
        text_rect = text_surf.get_rect(center=self.PLAYER_POS + pygame.math.Vector2(0, -self.PLAYER_SIZE - 20))
        self.screen.blit(text_surf, text_rect)


    def _render_enemies(self):
        for e in self.enemies:
            color = self.COLOR_ENEMY_TYPES[self.wave % len(self.COLOR_ENEMY_TYPES)]
            self._draw_glowing_hexagon(self.screen, color, e.pos, e.size)

    def _render_projectiles(self):
        for p in self.projectiles:
            # Trail
            for i, pos in enumerate(p.trail):
                alpha = int(150 * (i / len(p.trail)))
                self._draw_glowing_hexagon(self.screen, self.COLOR_PROJECTILE + (alpha,), pos, p.size * (i/len(p.trail)), 0)
            # Main projectile
            self._draw_glowing_hexagon(self.screen, self.COLOR_PROJECTILE, p.pos, p.size)

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Wave
        wave_text = f"WAVE: {self.wave}"
        wave_surf = self.font_ui.render(wave_text, True, self.COLOR_TEXT)
        wave_rect = wave_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(wave_surf, wave_rect)

        if self.game_over:
            go_font = pygame.font.SysFont("Consolas", 60, bold=True)
            go_text = "GAME OVER"
            go_surf = go_font.render(go_text, True, (255, 50, 50))
            go_rect = go_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(go_surf, go_rect)

# --- To run and play the game manually ---
if __name__ == '__main__':
    # Re-enable video driver for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Hexa-Blast")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Manual Control Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Display the game ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'R' to restart
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting_for_reset = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        waiting_for_reset = False
                clock.tick(30)
                
        clock.tick(30) # Run at 30 FPS

    pygame.quit()