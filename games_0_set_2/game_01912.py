
# Generated: 2025-08-27T18:40:33.143793
# Source Brief: brief_01912.md
# Brief Index: 1912

        
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


# Helper classes for game objects
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w = 20
        self.h = 30
        self.speed = 5
        self.health = 3
        self.max_health = 3
        self.fire_cooldown = 0
        self.max_fire_cooldown = 8  # Fire rate
        self.shield_active = False
        self.shield_timer = 0
        self.shield_duration = 15  # 0.5 seconds at 30fps
        self.shield_cooldown = 0
        self.shield_max_cooldown = 90  # 3 seconds cooldown
        self.invulnerable_timer = 0

    def is_alive(self):
        return self.health > 0

class Projectile:
    def __init__(self, x, y, dy, color, size):
        self.x = x
        self.y = y
        self.dy = dy
        self.color = color
        self.size = size

class Enemy:
    def __init__(self, x, y, wave, np_random):
        self.x = x
        self.y = y
        self.w = 25
        self.h = 25
        self.np_random = np_random
        self.fire_cooldown = self.np_random.integers(30, 90)

        # Difficulty scaling based on wave
        if wave == 1: # Simple downward movement
            self.type = 'grunt'
            self.speed_x = 0
            self.speed_y = 1.5
            self.color = (255, 80, 80)
        elif wave == 2: # Sinusoidal movement
            self.type = 'swooper'
            self.speed_x = 2
            self.speed_y = 1
            self.amplitude = 40
            self.frequency = 0.05
            self.initial_x = x
            self.color = (255, 150, 50)
        elif wave == 3: # Diagonal bouncers
            self.type = 'bouncer'
            self.speed_x = 2.5 * (1 if self.np_random.random() > 0.5 else -1)
            self.speed_y = 2.5
            self.color = (200, 100, 255)
        else: # Faster, more aggressive
            self.type = 'hunter'
            self.speed_x = 0
            self.speed_y = 2
            self.color = (255, 50, 200)

    def move(self, screen_width):
        if self.type == 'swooper':
            self.y += self.speed_y
            self.x = self.initial_x + self.amplitude * math.sin(self.frequency * self.y)
        elif self.type == 'bouncer':
            self.x += self.speed_x
            self.y += self.speed_y
            if self.x <= 0 or self.x >= screen_width - self.w:
                self.speed_x *= -1
        else: # Grunt and Hunter
            self.x += self.speed_x
            self.y += self.speed_y

    def should_fire(self):
        self.fire_cooldown -= 1
        if self.fire_cooldown <= 0:
            self.fire_cooldown = self.np_random.integers(60, 120)
            return True
        return False

class Particle:
    def __init__(self, x, y, np_random):
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = np_random.integers(10, 20)
        self.color = random.choice([(255, 220, 50), (255, 150, 0), (255, 50, 50)])

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        return self.lifetime > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Hold Shift for a temporary shield. Press Space to fire."
    )
    game_description = (
        "Pilot a lone starship against waves of alien invaders. Survive until wave 5 to win."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500 # Increased for longer gameplay
    WIN_WAVE = 5

    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_PLAYER_PROJECTILE = (255, 255, 255)
    COLOR_ENEMY_PROJECTILE = (255, 80, 80)
    COLOR_SHIELD = (100, 180, 255, 128)
    COLOR_TEXT = (220, 220, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.player = None
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []

        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False
        self.prev_space_held = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player = Player(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 50)
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False
        self.prev_space_held = False
        
        self._spawn_stars(200)
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.1  # Survival reward
        self.steps += 1

        # --- Handle Input and Cooldowns ---
        self._handle_input(action)
        self.player.fire_cooldown = max(0, self.player.fire_cooldown - 1)
        self.player.shield_cooldown = max(0, self.player.shield_cooldown - 1)
        self.player.invulnerable_timer = max(0, self.player.invulnerable_timer - 1)
        if self.player.shield_active:
            self.player.shield_timer -= 1
            reward -= 0.1 # Shield usage penalty
            if self.player.shield_timer <= 0:
                self.player.shield_active = False
                self.player.shield_cooldown = self.player.shield_max_cooldown

        # --- Update Game Objects ---
        self._update_player()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        # --- Collision Detection ---
        reward += self._handle_collisions()

        # --- Game Progression ---
        if not self.enemies and self.player.is_alive():
            self._spawn_wave()
        
        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            if self.player.is_alive() and self.wave >= self.WIN_WAVE:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.player.y -= self.player.speed
        if movement == 2: self.player.y += self.player.speed
        if movement == 3: self.player.x -= self.player.speed
        if movement == 4: self.player.x += self.player.speed
        
        # Firing
        if space_held and self.player.fire_cooldown == 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append(Projectile(self.player.x, self.player.y - self.player.h / 2, -10, self.COLOR_PLAYER_PROJECTILE, 4))
            self.player.fire_cooldown = self.player.max_fire_cooldown
        
        # Shield
        if shift_held and not self.player.shield_active and self.player.shield_cooldown == 0:
            # sfx: shield_activate.wav
            self.player.shield_active = True
            self.player.shield_timer = self.player.shield_duration

        self.prev_space_held = space_held

    def _update_player(self):
        self.player.x = np.clip(self.player.x, self.player.w / 2, self.SCREEN_WIDTH - self.player.w / 2)
        self.player.y = np.clip(self.player.y, self.player.h / 2, self.SCREEN_HEIGHT - self.player.h / 2)

    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if 0 < p.y < self.SCREEN_HEIGHT]
        for p in self.player_projectiles: p.y += p.dy

        self.enemy_projectiles = [p for p in self.enemy_projectiles if 0 < p.y < self.SCREEN_HEIGHT]
        for p in self.enemy_projectiles: p.y += p.dy

    def _update_enemies(self):
        for e in self.enemies:
            e.move(self.SCREEN_WIDTH)
            if e.should_fire() and e.y > 0:
                # sfx: enemy_shoot.wav
                projectile_speed = 3 + (self.wave * 0.5)
                self.enemy_projectiles.append(Projectile(e.x, e.y + e.h / 2, projectile_speed, self.COLOR_ENEMY_PROJECTILE, 3))
        self.enemies = [e for e in self.enemies if e.y < self.SCREEN_HEIGHT + e.h]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Enemies
        for p in self.player_projectiles[:]:
            for e in self.enemies[:]:
                if abs(p.x - e.x) < (p.size + e.w) / 2 and abs(p.y - e.y) < (p.size + e.h) / 2:
                    # sfx: explosion.wav
                    self.enemies.remove(e)
                    self.player_projectiles.remove(p)
                    self.score += 10
                    reward += 1
                    for _ in range(20): self.particles.append(Particle(e.x, e.y, self.np_random))
                    break
        
        # Enemy projectiles vs Player
        if self.player.invulnerable_timer == 0:
            for p in self.enemy_projectiles[:]:
                if abs(p.x - self.player.x) < (p.size + self.player.w) / 2 and abs(p.y - self.player.y) < (p.size + self.player.h) / 2:
                    self.enemy_projectiles.remove(p)
                    if not self.player.shield_active:
                        # sfx: player_hit.wav
                        self.player.health -= 1
                        self.player.invulnerable_timer = 60 # 2 seconds invulnerability
                        for _ in range(30): self.particles.append(Particle(self.player.x, self.player.y, self.np_random))
                    else:
                        # sfx: shield_deflect.wav
                        for _ in range(5): self.particles.append(Particle(p.x, p.y, self.np_random))
                    break
        return reward

    def _spawn_wave(self):
        self.wave += 1
        if self.wave >= self.WIN_WAVE:
            return
            
        num_enemies = 5 + (self.wave - 1) * 5
        for i in range(num_enemies):
            x = (i + 1) * (self.SCREEN_WIDTH / (num_enemies + 1))
            y = self.np_random.integers(-150, -50)
            self.enemies.append(Enemy(x, y, self.wave, self.np_random))

    def _spawn_stars(self, count):
        self.stars = []
        for _ in range(count):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT)
            size = self.np_random.choice([1, 2, 3])
            brightness = self.np_random.integers(50, 150)
            self.stars.append(((x, y), size, (brightness, brightness, brightness)))

    def _check_termination(self):
        return not self.player.is_alive() or self.steps >= self.MAX_STEPS or self.wave >= self.WIN_WAVE

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for pos, size, color in self.stars:
            pygame.draw.circle(self.screen, color, pos, size / 2)
            
        # Particles
        for p in self.particles:
            alpha = int(255 * (p.lifetime / 20))
            p_color = (p.color[0], p.color[1], p.color[2], alpha)
            temp_surf = pygame.Surface((p.lifetime, p.lifetime), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, p_color, (p.lifetime/2, p.lifetime/2), p.lifetime/2)
            self.screen.blit(temp_surf, (int(p.x - p.lifetime/2), int(p.y - p.lifetime/2)), special_flags=pygame.BLEND_RGBA_ADD)

        # Player Projectiles
        for p in self.player_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), p.size, p.color)
            pygame.gfxdraw.aacircle(self.screen, int(p.x), int(p.y), p.size, p.color)

        # Enemy Projectiles
        for p in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), p.size, p.color)
            pygame.gfxdraw.aacircle(self.screen, int(p.x), int(p.y), p.size, p.color)

        # Enemies
        for e in self.enemies:
            rect = pygame.Rect(int(e.x - e.w/2), int(e.y - e.h/2), e.w, e.h)
            pygame.draw.rect(self.screen, e.color, rect, border_radius=4)
        
        # Player
        if self.player.is_alive():
            is_invulnerable_flicker = self.player.invulnerable_timer > 0 and (self.steps // 3) % 2 == 0
            if not is_invulnerable_flicker:
                # Glow
                glow_size = self.player.h * 1.5
                temp_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, self.COLOR_PLAYER_GLOW, (glow_size/2, glow_size/2), glow_size/2)
                self.screen.blit(temp_surf, (int(self.player.x - glow_size/2), int(self.player.y - glow_size/2)), special_flags=pygame.BLEND_RGBA_ADD)
                
                # Ship
                points = [
                    (self.player.x, self.player.y - self.player.h / 2),
                    (self.player.x - self.player.w / 2, self.player.y + self.player.h / 2),
                    (self.player.x + self.player.w / 2, self.player.y + self.player.h / 2)
                ]
                pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x,y in points], self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x,y in points], self.COLOR_PLAYER)

            # Shield
            if self.player.shield_active:
                shield_radius = int(max(self.player.w, self.player.h) * 0.8)
                alpha = int(128 * (self.player.shield_timer / self.player.shield_duration))
                shield_color = (self.COLOR_SHIELD[0], self.COLOR_SHIELD[1], self.COLOR_SHIELD[2], alpha)
                temp_surf = pygame.Surface((shield_radius*2, shield_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, shield_color, (shield_radius, shield_radius), shield_radius)
                self.screen.blit(temp_surf, (int(self.player.x - shield_radius), int(self.player.y - shield_radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text_str = f"WAVE: {self.wave}" if self.wave < self.WIN_WAVE else "VICTORY!"
        wave_text = self.font_large.render(wave_text_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Health
        health_text = self.font_small.render("HEALTH:", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH // 2 - 80, self.SCREEN_HEIGHT - 30))
        for i in range(self.player.max_health):
            color = self.COLOR_PLAYER if i < self.player.health else (80, 80, 80)
            rect = pygame.Rect(self.SCREEN_WIDTH // 2 + i * 25, self.SCREEN_HEIGHT - 32, 20, 20)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # Game Over Message
        if not self.player.is_alive():
            end_text = self.font_large.render("GAME OVER", True, (255, 50, 50))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave, "health": self.player.health}

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Arcade Shooter")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    # --- Game Loop ---
    running = True
    while running:
        # Action mapping for human control
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # Pygame uses a different coordinate system for surfaces, so we need to transpose
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(30) # Control the frame rate
        
    env.close()