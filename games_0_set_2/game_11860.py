import os
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes for Game Entities ---

class Particle:
    """A single particle for effects like explosions or trails."""
    def __init__(self, pos, vel, lifespan, color, size_range):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.size = random.uniform(size_range[0], size_range[1])

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.vel *= 0.98  # Damping

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            # Create a temporary surface for alpha blending
            particle_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, self.color + (alpha,), (int(self.size), int(self.size)), int(self.size))
            surface.blit(particle_surf, (int(self.pos.x - self.size), int(self.pos.y - self.size)))

class Projectile:
    """A projectile fired by a bot."""
    def __init__(self, pos, vel, color, damage, size=4):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.damage = damage
        self.size = size
        self.alive = True

    def update(self):
        self.pos += self.vel

    def draw(self, surface):
        # Draw a glowing line for the projectile
        end_pos = self.pos - self.vel.normalize() * 15
        pygame.draw.line(surface, self.color, self.pos, end_pos, self.size)
        # Glow effect
        pygame.draw.line(surface, self.color, self.pos, end_pos, self.size * 3)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.size, self.color)

    def is_offscreen(self, width, height):
        return not (0 < self.pos.x < width and 0 < self.pos.y < height)

class Bot:
    """Base class for Player and Enemy bots."""
    def __init__(self, pos, size, color, max_health, speed):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.size = size
        self.color = color
        self.max_health = max_health
        self.health = max_health
        self.speed = speed
        self.alive = True
        self.last_hit_timer = 0

    def update(self, screen_width, screen_height):
        self.pos += self.vel
        self.pos.x = np.clip(self.pos.x, self.size, screen_width - self.size)
        self.pos.y = np.clip(self.pos.y, self.size, screen_height - self.size)
        if self.last_hit_timer > 0:
            self.last_hit_timer -= 1

    def take_damage(self, amount):
        self.health -= amount
        self.last_hit_timer = 10 # Flash for 10 frames
        if self.health <= 0:
            self.health = 0
            self.alive = False
        return amount

    def get_rect(self):
        return pygame.Rect(self.pos.x - self.size, self.pos.y - self.size, self.size * 2, self.size * 2)

    def draw_health_bar(self, surface):
        if self.health < self.max_health:
            bar_width = self.size * 2
            bar_height = 5
            bar_x = self.pos.x - self.size
            bar_y = self.pos.y - self.size - 10

            health_ratio = self.health / self.max_health
            current_health_width = bar_width * health_ratio
            
            pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(surface, (255, 0, 0) if self.color[0] > 128 else (0, 255, 0), (bar_x, bar_y, current_health_width, bar_height))

    def draw_glow(self, surface, color, intensity):
        radius = int(self.size * intensity)
        for i in range(radius, 0, -2):
            alpha = int(80 * (1 - i / radius))
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), i, color + (alpha,))

    def draw(self, surface):
        # Flash white when hit
        draw_color = (255, 255, 255) if self.last_hit_timer > 0 else self.color
        
        # Main body
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.size, draw_color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.size, draw_color)
        
        # Glow
        self.draw_glow(surface, self.color, 1.8)
        
        self.draw_health_bar(surface)

# --- Main Gymnasium Environment Class ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Survive waves of enemy bots in this top-down arena shooter. Dodge attacks, "
        "manage cooldowns, and use your shield wisely to achieve victory."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to fire your laser and "
        "shift to activate your shield."
    )
    auto_advance = True

    # --- Colors and Constants ---
    COLOR_BG = (15, 20, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_SHIELD = (0, 192, 255)
    COLOR_LASER = (255, 255, 0)
    COLOR_UI = (220, 220, 240)
    
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500 # Increased for longer battles
    MAX_WAVES = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Game State Variables ---
        self.player = None
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.current_wave = 0
        self.game_over = False
        
        # Action state tracking
        self.prev_space_held = False
        self.prev_shift_held = False

        # --- Player Bot Stats ---
        self.player_stats = {
            "size": 15,
            "speed": 5,
            "max_health": 100,
            "laser_damage": 10,
            "laser_speed": 12,
            "primary_cooldown_max": 10, # 3 shots per second at 30fps
            "secondary_cooldown_max": 150, # 5 second cooldown
            "shield_duration_max": 60 # 2 second duration
        }
        self.player_primary_cooldown = 0
        self.player_secondary_cooldown = 0
        self.player_shield_timer = 0
        
        # Initialize state by calling reset
        # self.reset() # This is typically called by the user/runner, not in __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_wave = 0

        # Create player
        self.player = Bot(
            pos=(self.SCREEN_WIDTH // 4, self.SCREEN_HEIGHT // 2),
            size=self.player_stats["size"],
            color=self.COLOR_PLAYER,
            max_health=self.player_stats["max_health"],
            speed=self.player_stats["speed"]
        )
        
        # Reset cooldowns and effects
        self.player_primary_cooldown = 0
        self.player_secondary_cooldown = 0
        self.player_shield_timer = 0
        
        # Clear dynamic elements
        self.enemies.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        # Action state
        self.prev_space_held = False
        self.prev_shift_held = False

        # Start the first wave
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_player_input(movement, space_held, shift_held)
        
        # --- Update Game Logic ---
        self._update_cooldowns()
        self._update_player()
        self._update_enemies(reward)
        reward += self._update_projectiles()
        self._update_particles()
        reward += self._handle_collisions()
        
        # --- Check Game State ---
        terminated = False
        if not self.player.alive:
            self.game_over = True
            reward -= 100 # Large penalty for losing
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True
        
        if self.game_over:
            terminated = True
        
        # Check for wave completion
        if self.player.alive and not self.enemies:
            reward += 5.0 # Wave clear bonus
            self.score += 50
            if self.current_wave >= self.MAX_WAVES:
                self.game_over = True
                terminated = True
                reward += 100 # Large reward for winning
            else:
                self._spawn_wave()
                # Heal player slightly between waves
                self.player.health = min(self.player.max_health, self.player.health + 25)

        # Update previous action states
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held, shift_held):
        # Movement
        # Reset velocity
        self.player.vel.x = 0
        self.player.vel.y = 0

        if movement == 1: self.player.vel.y = -self.player.speed # Up
        elif movement == 2: self.player.vel.y = self.player.speed # Down
        elif movement == 3: self.player.vel.x = -self.player.speed # Left
        elif movement == 4: self.player.vel.x = self.player.speed # Right

        # Primary Weapon (Laser) - on press
        if space_held and not self.prev_space_held and self.player_primary_cooldown <= 0:
            self.player_primary_cooldown = self.player_stats["primary_cooldown_max"]
            direction = pygame.Vector2(1, 0) # Fire right by default
            self.projectiles.append(Projectile(
                self.player.pos, 
                direction * self.player_stats["laser_speed"],
                self.COLOR_LASER,
                self.player_stats["laser_damage"]
            ))

        # Secondary Ability (Shield) - on press
        if shift_held and not self.prev_shift_held and self.player_secondary_cooldown <= 0:
            self.player_secondary_cooldown = self.player_stats["secondary_cooldown_max"]
            self.player_shield_timer = self.player_stats["shield_duration_max"]

    def _update_cooldowns(self):
        if self.player_primary_cooldown > 0: self.player_primary_cooldown -= 1
        if self.player_secondary_cooldown > 0: self.player_secondary_cooldown -= 1
        if self.player_shield_timer > 0: self.player_shield_timer -= 1

    def _update_player(self):
        self.player.update(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

    def _update_enemies(self, reward):
        for enemy in self.enemies:
            # Simple AI: move towards player
            if (self.player.pos - enemy.pos).length() > 0:
                direction = (self.player.pos - enemy.pos).normalize()
                enemy.vel = direction * enemy.speed
            else:
                enemy.vel = pygame.Vector2(0)
            enemy.update(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p.update()
            hit = False
            # Check collision with enemies
            for enemy in self.enemies:
                if enemy.get_rect().collidepoint(p.pos):
                    damage_dealt = enemy.take_damage(p.damage)
                    reward += 0.1 # Reward for hitting
                    self.score += damage_dealt
                    self._create_explosion(p.pos, self.COLOR_ENEMY, 10)
                    p.alive = False
                    hit = True
                    break
            if hit:
                continue
            
            if p.is_offscreen(self.SCREEN_WIDTH, self.SCREEN_HEIGHT):
                p.alive = False
            
            if p.alive:
                projectiles_to_keep.append(p)
        self.projectiles = projectiles_to_keep
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _handle_collisions(self):
        reward = 0
        # Player vs Enemy collision
        player_rect = self.player.get_rect()
        for enemy in self.enemies:
            if player_rect.colliderect(enemy.get_rect()):
                if self.player_shield_timer <= 0:
                    self.player.take_damage(0.5) # Minor collision damage
                    reward -= 0.1 # Penalty for taking damage
                enemy.take_damage(0.5)
                # Simple knockback
                if (self.player.pos - enemy.pos).length() > 0:
                    knockback = (self.player.pos - enemy.pos).normalize() * 2
                    self.player.pos += knockback
                    enemy.pos -= knockback

        # Check for dead enemies
        alive_enemies = []
        for enemy in self.enemies:
            if enemy.alive:
                alive_enemies.append(enemy)
            else:
                reward += 1.0 # Reward for defeating an enemy
                self.score += 100
                self._create_explosion(enemy.pos, self.COLOR_ENEMY, 30)
        self.enemies = alive_enemies
        return reward

    def _spawn_wave(self):
        self.current_wave += 1
        num_enemies = 1 + self.current_wave 
        
        for i in range(num_enemies):
            scale_factor = 1 + (self.current_wave - 1) * 0.05
            enemy_health = 20 * scale_factor
            enemy_speed = random.uniform(1.0, 2.0) * scale_factor

            pos_x = self.SCREEN_WIDTH - 50
            pos_y = random.randint(50, self.SCREEN_HEIGHT - 50)
            
            self.enemies.append(Bot(
                pos=(pos_x, pos_y),
                size=12,
                color=self.COLOR_ENEMY,
                max_health=enemy_health,
                speed=min(enemy_speed, self.player.speed - 1) # Cap speed
            ))

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            vel = pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3))
            lifespan = random.randint(15, 30)
            self.particles.append(Particle(pos, vel, lifespan, color, (1, 4)))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "player_health": self.player.health,
            "enemies_left": len(self.enemies),
        }

    def _render_game(self):
        for p in self.particles:
            p.draw(self.screen)
        
        for enemy in self.enemies:
            enemy.draw(self.screen)
            
        if self.player.alive:
            self.player.draw(self.screen)
            if self.player_shield_timer > 0:
                ratio = self.player_shield_timer / self.player_stats["shield_duration_max"]
                alpha = int(100 + 100 * math.sin(ratio * math.pi))
                radius = int(self.player.size * 1.5)
                pygame.gfxdraw.filled_circle(self.screen, int(self.player.pos.x), int(self.player.pos.y), radius, self.COLOR_SHIELD + (alpha // 2,))
                pygame.gfxdraw.aacircle(self.screen, int(self.player.pos.x), int(self.player.pos.y), radius, self.COLOR_SHIELD + (alpha,))
        
        for p in self.projectiles:
            p.draw(self.screen)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        wave_text = self.font_main.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        bar_width = 100
        bar_height = 10
        primary_ratio = self.player_primary_cooldown / self.player_stats["primary_cooldown_max"]
        pygame.draw.rect(self.screen, (50, 50, 50), (10, self.SCREEN_HEIGHT - 30, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_LASER, (10, self.SCREEN_HEIGHT - 30, bar_width * (1 - primary_ratio), bar_height))
        
        secondary_ratio = self.player_secondary_cooldown / self.player_stats["secondary_cooldown_max"]
        pygame.draw.rect(self.screen, (50, 50, 50), (10, self.SCREEN_HEIGHT - 15, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_SHIELD, (10, self.SCREEN_HEIGHT - 15, bar_width * (1 - secondary_ratio), bar_height))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        win = self.player.alive and self.current_wave >= self.MAX_WAVES
        text = "VICTORY" if win else "DEFEATED"
        color = self.COLOR_PLAYER if win else self.COLOR_ENEMY
        
        game_over_text = self.font_big.render(text, True, color)
        text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        
        overlay.blit(game_over_text, text_rect)
        self.screen.blit(overlay, (0, 0))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For this to work, you might need to comment out the os.environ line at the top
    # or run with a virtual display buffer like xvfb.
    
    # To run with display, temporarily comment this line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # And uncomment this one:
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for manual play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bot Annihilation")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        # --- Frame Rate ---
        clock.tick(30) # Run at 30 FPS

    env.close()