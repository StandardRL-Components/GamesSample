import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:27:47.156860
# Source Brief: brief_01709.md
# Brief Index: 1709
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Functions for Visuals ---

def draw_glowing_circle(surface, color, center, radius, max_glow=15):
    """Draws a circle with a glow effect."""
    radius = int(radius)
    center = (int(center[0]), int(center[1]))
    
    for i in range(max_glow, 0, -1):
        alpha = int(150 * (1 - (i / max_glow)))
        glow_color = (*color, alpha)
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius + i, glow_color)
        
    pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
    pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)

def draw_health_bar(surface, pos, size, border_color, bg_color, health_color, progress):
    """Draws a health bar."""
    pygame.draw.rect(surface, bg_color, (*pos, *size))
    pygame.draw.rect(surface, health_color, (*pos, int(size[0] * progress), size[1]))
    pygame.draw.rect(surface, border_color, (*pos, *size), 2)

# --- Game Entity Classes ---

class Particle:
    def __init__(self, pos, vel, size, color, lifetime):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.size = size
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.vel *= 0.98 # Friction

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            current_size = self.size * (self.lifetime / self.max_lifetime)
            if current_size < 1: return
            
            # Use gfxdraw for anti-aliased, alpha-blended circles
            temp_surf = pygame.Surface((current_size*2, current_size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (current_size, current_size), current_size)
            surface.blit(temp_surf, self.pos - pygame.Vector2(current_size, current_size), special_flags=pygame.BLEND_RGBA_ADD)

class QuantumTile:
    def __init__(self, pos, size, lifetime):
        self.rect = pygame.Rect(pos, (size, size))
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.lifetime -= 1

    def draw(self, surface):
        alpha = 100 * (math.sin(self.lifetime * 0.2) * 0.5 + 0.5)
        color = (180, 220, 255, alpha)
        
        # Pulsing glow effect
        temp_surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        temp_surf.fill(color)
        surface.blit(temp_surf, self.rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.rect(surface, (200, 220, 240), self.rect, 1)

class Player:
    def __init__(self):
        self.pos = pygame.Vector2(320, 200)
        self.base_size = 15
        self.size = self.base_size
        self.target_size = self.base_size
        self.max_size = 40
        self.min_size = 8
        self.base_speed = 5
        self.health = 100
        self.max_health = 100
        self.grow_cooldown = 0
        self.shrink_cooldown = 0
        self.invincibility_timer = 0

    def update(self):
        # Smoothly interpolate size
        self.size += (self.target_size - self.size) * 0.1
        
        # Natural size regression to base
        self.target_size += (self.base_size - self.target_size) * 0.005
        self.target_size = max(self.min_size, min(self.max_size, self.target_size))

        # Update cooldowns
        if self.grow_cooldown > 0: self.grow_cooldown -= 1
        if self.shrink_cooldown > 0: self.shrink_cooldown -= 1
        if self.invincibility_timer > 0: self.invincibility_timer -= 1
        
    @property
    def speed(self):
        # Speed is inversely proportional to size
        return self.base_speed * (self.base_size / self.size)

    def draw(self, surface):
        color = (0, 150, 255)
        if self.invincibility_timer > 0 and self.invincibility_timer % 4 < 2:
            color = (255, 255, 255) # Flash white when invincible
        draw_glowing_circle(surface, color, self.pos, self.size)

class Enemy:
    def __init__(self, pos, health, size, speed, damage, is_boss=False):
        self.pos = pygame.Vector2(pos)
        self.base_size = size
        self.size = self.base_size
        self.target_size = self.base_size
        self.max_size = 60
        self.min_size = 5
        self.health = health
        self.max_health = health
        self.speed = speed
        self.damage = damage
        self.is_boss = is_boss
        self.hit_timer = 0

    def update(self, player_pos):
        # Move towards player
        direction = (player_pos - self.pos).normalize() if (player_pos - self.pos).length() > 0 else pygame.Vector2(0)
        self.pos += direction * self.speed

        # Smoothly interpolate size
        self.size += (self.target_size - self.size) * 0.1
        
        # Natural size regression
        self.target_size += (self.base_size - self.target_size) * 0.01

        if self.hit_timer > 0: self.hit_timer -= 1

    def draw(self, surface):
        color = (255, 50, 50) if not self.is_boss else (255, 0, 150)
        if self.hit_timer > 0:
            color = (255, 200, 200) # Flash white/pink on hit

        draw_glowing_circle(surface, color, self.pos, self.size)
        
        # Health bar
        bar_width = self.size * 2.5
        bar_pos = self.pos - pygame.Vector2(bar_width / 2, self.size + 10)
        draw_health_bar(surface, bar_pos, (bar_width, 5), (50,50,50), (100,0,0), (255,0,0), self.health / self.max_health)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Survive waves of enemies in a quantum arena by strategically growing and shrinking. "
        "Use a powerful area-of-effect blast or a swift dash to defeat your foes."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to unleash a growth blast and shift to perform a shrinking dash."
    )
    auto_advance = True

    # --- Colors and Constants ---
    COLOR_BG = (15, 18, 28)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    
    GROW_ATTACK_COOLDOWN = 30
    SHRINK_ATTACK_COOLDOWN = 45
    FINAL_WAVE = 7
    MAX_STEPS = 2500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.render_mode = render_mode
        self.screen_shake = 0

        # Game state variables initialized in reset
        self.player = None
        self.enemies = []
        self.particles = []
        self.tiles = []
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False
        self.upgrades = {"anchor": False, "invincibility": False}
        
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player = Player()
        self.enemies = []
        self.particles = []
        self.tiles = []
        
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False
        self.screen_shake = 0
        self.upgrades = {"anchor": False, "invincibility": False}
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input and Player Actions ---
        events = self._handle_input(movement, space_held, shift_held)
        reward += events.get("damage_dealt_reward", 0)

        # --- Update Game State ---
        update_events = self._update_game_state()
        reward += update_events.get("damage_taken_reward", 0)
        reward += update_events.get("enemy_defeat_reward", 0)
        
        if update_events.get("wave_cleared", False):
            # self.wave is incremented in _spawn_wave, so current wave is self.wave-1
            if self.wave - 1 in [3, 6]:
                reward += 10.0 # Boss wave clear bonus
                if self.wave - 1 == 3 and not self.upgrades["anchor"]:
                    self.upgrades["anchor"] = True
                elif self.wave - 1 == 6 and not self.upgrades["invincibility"]:
                    self.upgrades["invincibility"] = True
            
            if self.wave > self.FINAL_WAVE:
                terminated = True
                reward += 100.0 # Victory bonus
            else:
                self._spawn_wave()
                self.player.health = min(self.player.max_health, self.player.health + 25) # Heal between waves
        
        self.score += reward

        # --- Check Termination Conditions ---
        if self.player.health <= 0:
            terminated = True
            reward -= 10.0 # Penalty for dying
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        if terminated or truncated:
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        events = {"damage_dealt_reward": 0.0}
        
        # Player Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            self.player.pos += move_vec.normalize() * self.player.speed
        
        self.player.pos.x = np.clip(self.player.pos.x, self.player.size, self.width - self.player.size)
        self.player.pos.y = np.clip(self.player.pos.y, self.player.size, self.height - self.player.size)

        # Grow Attack (Space)
        if space_held and self.player.grow_cooldown == 0:
            self.player.grow_cooldown = self.GROW_ATTACK_COOLDOWN
            self.player.target_size = min(self.player.max_size, self.player.target_size + 5)
            # sound: "whoosh_grow.wav"
            
            attack_radius = self.player.size * 3
            self._create_particles(self.player.pos, 30, (255, 255, 100), 2, 6, 20) # Attack particles
            
            if self.upgrades["anchor"]:
                self.tiles.append(QuantumTile(self.player.pos - pygame.Vector2(40, 40), 80, 150))

            for enemy in self.enemies:
                if self.player.pos.distance_to(enemy.pos) < attack_radius + enemy.size:
                    damage = 5 + (self.player.size / self.player.base_size) * 10
                    enemy.health -= damage
                    enemy.target_size = max(enemy.min_size, enemy.target_size - 4)
                    enemy.hit_timer = 10
                    events["damage_dealt_reward"] += 0.1
                    self._create_particles(enemy.pos, 10, (255, 100, 100), 1, 4, 15) # Hit particles
        
        # Shrink Attack (Shift)
        if shift_held and self.player.shrink_cooldown == 0:
            self.player.shrink_cooldown = self.SHRINK_ATTACK_COOLDOWN
            self.player.target_size = max(self.player.min_size, self.player.target_size - 3)
            # sound: "zip_shrink.wav"

            if self.upgrades["invincibility"]:
                self.player.invincibility_timer = 20 # Short invincibility
            
            # This attack is a dash through enemies
            for _ in range(15): # Create a trail
                self._create_particles(self.player.pos, 1, (100, 200, 255), 1, 3, 25, trail=True)

            for enemy in self.enemies:
                # Check line segment intersection for dash
                if self.player.pos.distance_to(enemy.pos) < self.player.size + enemy.size:
                    damage = 5 + (self.player.speed / self.player.base_speed) * 5
                    enemy.health -= damage
                    enemy.target_size = min(enemy.max_size, enemy.target_size + 2)
                    enemy.hit_timer = 10
                    events["damage_dealt_reward"] += 0.1
                    self._create_particles(enemy.pos, 10, (255, 100, 100), 1, 4, 15)

        return events

    def _update_game_state(self):
        events = {"damage_taken_reward": 0.0, "enemy_defeat_reward": 0.0, "wave_cleared": False}
        
        self.player.update()
        
        # Check for player on quantum tile
        on_tile = False
        for tile in self.tiles:
            if tile.rect.collidepoint(self.player.pos):
                on_tile = True
                break
        if on_tile: # Stop size regression on tile
            self.player.target_size += (self.player.base_size - self.player.target_size) * -0.05

        for i in range(len(self.enemies) - 1, -1, -1):
            enemy = self.enemies[i]
            enemy.update(self.player.pos)
            
            # Enemy collision with player
            if enemy.pos.distance_to(self.player.pos) < enemy.size + self.player.size:
                if self.player.invincibility_timer == 0:
                    self.player.health -= enemy.damage
                    self.player.invincibility_timer = 30 # Post-hit invincibility
                    self.screen_shake = 10
                    events["damage_taken_reward"] -= 0.5
                    # sound: "player_hit.wav"
                
            if enemy.health <= 0:
                # sound: "enemy_explode.wav"
                self._create_particles(enemy.pos, 50, (255, 80, 80), 2, 8, 40)
                events["enemy_defeat_reward"] += 1.0 + (5.0 if enemy.is_boss else 0)
                self.enemies.pop(i)
        
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p.update()
            if p.lifetime <= 0:
                self.particles.pop(i)
        
        for i in range(len(self.tiles) - 1, -1, -1):
            t = self.tiles[i]
            t.update()
            if t.lifetime <= 0:
                self.tiles.pop(i)
        
        if not self.enemies:
            events["wave_cleared"] = True
            
        return events

    def _spawn_wave(self):
        self.wave += 1
        num_enemies = 2 + int(self.wave / 2)
        
        is_boss_wave = self.wave in [3, 6, self.FINAL_WAVE]
        
        for i in range(num_enemies):
            angle = (2 * math.pi / num_enemies) * i
            dist = 350
            pos = pygame.Vector2(self.width/2 + dist * math.cos(angle), self.height/2 + dist * math.sin(angle))
            pos.x = np.clip(pos.x, 20, self.width - 20)
            pos.y = np.clip(pos.y, 20, self.height - 20)
            
            difficulty_mod = 1 + (self.wave - 1) * 0.05
            is_boss = is_boss_wave and i == num_enemies - 1

            health = (50 if not is_boss else 200) * difficulty_mod
            size = 12 if not is_boss else 25
            speed = (1.5 if not is_boss else 1.0) / difficulty_mod
            damage = (5 if not is_boss else 15) * difficulty_mod

            self.enemies.append(Enemy(pos, health, size, speed, damage, is_boss))
    
    def _create_particles(self, pos, count, color, min_speed, max_speed, lifetime, trail=False):
        for _ in range(count):
            if trail:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(min_speed, max_speed) * 0.1
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(min_speed, max_speed)
            
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            size = random.uniform(1, 4)
            self.particles.append(Particle(pos, vel, size, color, random.randint(lifetime-5, lifetime+5)))

    def _get_observation(self):
        offset = pygame.Vector2(0, 0)
        if self.screen_shake > 0:
            self.screen_shake -= 1
            offset.x = random.randint(-self.screen_shake, self.screen_shake)
            offset.y = random.randint(-self.screen_shake, self.screen_shake)

        self.screen.fill(self.COLOR_BG)
        
        self._render_game(offset)
        self._render_ui(offset)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, offset):
        # Draw tiles first, as they are in the background
        for tile in self.tiles:
            tile_rect = tile.rect.copy()
            tile_rect.topleft += offset
            # Re-implementing draw logic here to apply offset
            alpha = 100 * (math.sin(tile.lifetime * 0.2) * 0.5 + 0.5)
            color = (180, 220, 255, alpha)
            temp_surf = pygame.Surface(tile.rect.size, pygame.SRCALPHA)
            temp_surf.fill(color)
            self.screen.blit(temp_surf, tile_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.rect(self.screen, (200, 220, 240), tile_rect, 1)

        # Draw particles on a separate surface for additive blending
        particle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for particle in self.particles:
            particle.draw(particle_surface)
        self.screen.blit(particle_surface, offset, special_flags=pygame.BLEND_RGBA_ADD)

        # Draw enemies and player on top
        for enemy in self.enemies:
            # Re-implementing draw logic here to apply offset
            color = (255, 50, 50) if not enemy.is_boss else (255, 0, 150)
            if enemy.hit_timer > 0:
                color = (255, 200, 200)
            draw_glowing_circle(self.screen, color, enemy.pos + offset, enemy.size)
            bar_width = enemy.size * 2.5
            bar_pos = enemy.pos + offset - pygame.Vector2(bar_width / 2, enemy.size + 10)
            draw_health_bar(self.screen, bar_pos, (bar_width, 5), (50,50,50), (100,0,0), (255,0,0), enemy.health / enemy.max_health)
        
        # Player is drawn last among game objects
        player_color = self.COLOR_PLAYER
        if self.player.invincibility_timer > 0 and self.player.invincibility_timer % 4 < 2:
            player_color = (255, 255, 255)
        draw_glowing_circle(self.screen, player_color, self.player.pos + offset, self.player.size)

    def _render_ui(self, offset):
        # Player Health
        draw_health_bar(self.screen, (10 + offset.x, 10 + offset.y), (200, 20), (50,50,50), (100,0,0), (0,200,100), self.player.health / self.player.max_health)
        health_text = self.font_small.render(f"{int(self.player.health)} / {self.player.max_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15 + offset.x, 12 + offset.y))

        # Score and Wave
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.width - score_text.get_width() - 10 + offset.x, 10 + offset.y))
        wave_text = self.font_small.render(f"WAVE: {self.wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.width - wave_text.get_width() - 10 + offset.x, 40 + offset.y))

        # Ability/Card UI
        # Grow Attack (Space)
        grow_color = (255, 255, 100) if self.player.grow_cooldown == 0 else (80, 80, 40)
        pygame.draw.rect(self.screen, grow_color, (self.width - 110 + offset.x, self.height - 40 + offset.y, 30, 30), border_radius=5)
        if self.player.grow_cooldown > 0:
            progress = self.player.grow_cooldown / self.GROW_ATTACK_COOLDOWN
            pygame.draw.rect(self.screen, (120,120,80), (self.width - 110 + offset.x, self.height - 40 + offset.y, 30, int(30 * progress)), border_radius=5)
        
        # Shrink Attack (Shift)
        shrink_color = (100, 200, 255) if self.player.shrink_cooldown == 0 else (40, 70, 80)
        pygame.draw.rect(self.screen, shrink_color, (self.width - 70 + offset.x, self.height - 40 + offset.y, 30, 30), border_radius=5)
        if self.player.shrink_cooldown > 0:
            progress = self.player.shrink_cooldown / self.SHRINK_ATTACK_COOLDOWN
            pygame.draw.rect(self.screen, (80,110,120), (self.width - 70 + offset.x, self.height - 40 + offset.y, 30, int(30 * progress)), border_radius=5)

        # Upgrade Indicators
        if self.upgrades["anchor"]:
            anchor_text = self.font_small.render("ANCHOR", True, (180, 220, 255))
            self.screen.blit(anchor_text, (self.width - 110 + offset.x, self.height - 60 + offset.y))
        if self.upgrades["invincibility"]:
            inv_text = self.font_small.render("PHASE", True, (180, 220, 255))
            self.screen.blit(inv_text, (self.width - 70 + offset.x, self.height - 60 + offset.y))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player.health,
            "enemies_left": len(self.enemies),
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is NOT part of the Gymnasium environment
    
    # Check if a display is available, otherwise skip manual testing
    try:
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((640, 400))
        is_playable = True
    except pygame.error:
        print("Pygame display not available. Skipping manual play test.")
        is_playable = False

    if is_playable:
        env = GameEnv()
        obs, info = env.reset()
        
        pygame.display.set_caption("Quantum Arena")
        
        terminated = False
        truncated = False
        total_reward = 0
        
        while not terminated and not truncated:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Blit the observation onto the display screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(30) # Limit to 30 FPS

        print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
        env.close()