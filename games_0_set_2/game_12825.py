import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

# Helper classes for game entities
class Particle:
    def __init__(self, pos, vel, color, radius, lifetime):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.radius = radius
        self.lifetime = lifetime
        self.life = lifetime

    def update(self, dt):
        self.pos += self.vel * dt
        self.life -= dt
        return self.life > 0

    def draw(self, surface):
        alpha = max(0, min(255, int(255 * (self.life / self.lifetime))))
        r, g, b = self.color
        # Draw a filled circle with alpha
        temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (r, g, b, alpha), (self.radius, self.radius), self.radius)
        surface.blit(temp_surf, self.pos - pygame.Vector2(self.radius, self.radius), special_flags=pygame.BLEND_RGBA_ADD)

class Projectile:
    def __init__(self, pos, target_pos, element_idx, element_props):
        self.pos = pygame.Vector2(pos)
        self.element_idx = element_idx
        self.props = element_props
        self.color = self.props['color']
        self.radius = 8
        self.speed = 700
        
        direction = (target_pos - self.pos)
        if direction.length() > 0:
            self.vel = direction.normalize() * self.speed
        else:
            self.vel = pygame.Vector2(0, -1) * self.speed # Failsafe
        
        self.trail = deque(maxlen=10)

    def update(self, dt):
        self.trail.append(self.pos.copy())
        self.pos += self.vel * dt

    def draw(self, surface):
        # Draw trail
        for i, pos in enumerate(self.trail):
            alpha = int(255 * (i / len(self.trail)))
            pygame.gfxdraw.filled_circle(surface, int(pos.x), int(pos.y), int(self.radius * (i / len(self.trail))), (*self.color, alpha))

        # Draw main projectile with glow
        GameEnv._draw_glowing_circle(surface, self.color, self.pos, self.radius, 10)

class Enemy:
    def __init__(self, health, speed, center_pos):
        self.pos = pygame.Vector2(center_pos)
        self.health = health
        self.max_health = health
        self.speed = speed
        self.radius = 12
        direction_angle = random.uniform(0, 2 * math.pi)
        self.vel = pygame.Vector2(math.cos(direction_angle), math.sin(direction_angle)) * self.speed

    def update(self, dt):
        self.pos += self.vel * dt

    def draw(self, surface):
        # Body
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, (255, 0, 128))
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, (255, 0, 128))
        
        # Health bar
        if self.health < self.max_health:
            bar_width = 30
            bar_height = 4
            bar_pos = self.pos - pygame.Vector2(bar_width / 2, self.radius + 8)
            health_ratio = self.health / self.max_health
            
            pygame.draw.rect(surface, (50, 0, 0), (*bar_pos, bar_width, bar_height))
            pygame.draw.rect(surface, (255, 0, 0), (*bar_pos, bar_width * health_ratio, bar_height))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend the arena by shooting elemental projectiles at incoming enemies. "
        "Switch elements to build combos and maximize your score before time runs out."
    )
    user_guide = (
        "Controls: ←→ to move along the arena. ↑↓ to switch elements. Press space to fire at the nearest enemy."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30.0
    ARENA_RADIUS = 160
    ARENA_CENTER = pygame.Vector2(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 20)
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_ARENA = (40, 30, 80)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    
    ELEMENTS = {
        0: {'name': 'Fire', 'color': (255, 100, 0)},
        1: {'name': 'Ice', 'color': (100, 200, 255)},
        2: {'name': 'Lightning', 'color': (255, 255, 0)},
        3: {'name': 'Earth', 'color': (140, 90, 40)},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = 0
        self.player_health = 0
        self.player_angle = 0
        self.player_pos = pygame.Vector2(0, 0)
        self.selected_weapon_idx = 0
        self.last_fired_weapon_idx = -1
        self.combo = 0
        self.combo_aura_size = 0
        self.fire_cooldown = 0
        self.weapon_switch_cooldown = 0
        self.enemy_spawn_cooldown = 0
        self.base_enemy_spawn_rate = 0
        self.base_enemy_health = 0
        
        self.enemies = []
        self.projectiles = []
        self.particles = []

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.timer = 60.0
        self.player_health = 100
        self.player_angle = math.pi / 2 # Top of the circle
        self.selected_weapon_idx = 0
        self.last_fired_weapon_idx = -1 # -1 means no weapon fired yet
        self.combo = 0
        self.combo_aura_size = 0

        self.fire_cooldown = 0
        self.weapon_switch_cooldown = 0
        
        self.base_enemy_spawn_rate = 2.0 # seconds
        self.enemy_spawn_cooldown = self.base_enemy_spawn_rate
        self.base_enemy_health = 10

        self.enemies.clear()
        self.projectiles.clear()
        self.particles.clear()

        self._update_player_pos()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        dt = 1.0 / self.FPS
        reward = 0
        
        # --- Update Timers & Difficulty ---
        self.timer = max(0, self.timer - dt)
        self.fire_cooldown = max(0, self.fire_cooldown - dt)
        self.weapon_switch_cooldown = max(0, self.weapon_switch_cooldown - dt)
        
        # Difficulty scaling
        time_elapsed = 60.0 - self.timer
        current_spawn_rate = max(0.25, 2.0 - time_elapsed * 0.025) # Spawn rate increases
        current_enemy_health = 10 + int(time_elapsed / 10) * 5
        
        # --- Handle Input ---
        movement, fire_action, _ = action
        
        # Player movement
        if movement == 3: # Left (CCW)
            self.player_angle += 2.5 * dt
        elif movement == 4: # Right (CW)
            self.player_angle -= 2.5 * dt
        self._update_player_pos()

        # Weapon selection
        if self.weapon_switch_cooldown == 0:
            if movement == 1: # Up
                self.selected_weapon_idx = (self.selected_weapon_idx + 1) % len(self.ELEMENTS)
                self.weapon_switch_cooldown = 0.2
            elif movement == 2: # Down
                self.selected_weapon_idx = (self.selected_weapon_idx - 1 + len(self.ELEMENTS)) % len(self.ELEMENTS)
                self.weapon_switch_cooldown = 0.2
        
        # Firing
        if fire_action == 1 and self.fire_cooldown == 0:
            reward += self._fire_weapon()

        # --- Update Game State ---
        
        # Update and collide projectiles
        projectiles_to_remove = []
        for proj in self.projectiles:
            proj.update(dt)
            if not self.screen.get_rect().collidepoint(proj.pos):
                projectiles_to_remove.append(proj)
                continue
            
            enemies_to_remove = []
            for enemy in self.enemies:
                if proj.pos.distance_to(enemy.pos) < proj.radius + enemy.radius:
                    # Hit!
                    enemy.health -= 20 # All projectiles do same base damage
                    reward += 0.1
                    self._create_explosion(proj.pos, proj.color, 15)
                    # SFX: enemy_hit.wav
                    
                    if enemy.health <= 0:
                        self.score += 10 + self.combo * 2
                        reward += 1.0
                        if self.combo > 0 and self.combo % 5 == 0:
                            reward += 5.0
                        self._create_explosion(enemy.pos, (255, 200, 200), 30)
                        # SFX: enemy_explode.wav
                        if enemy not in enemies_to_remove:
                            enemies_to_remove.append(enemy)
                    
                    if proj not in projectiles_to_remove:
                        projectiles_to_remove.append(proj)
                    break # Projectile hits only one enemy
            
            for enemy in enemies_to_remove:
                self.enemies.remove(enemy)

        for proj in projectiles_to_remove:
            if proj in self.projectiles:
                self.projectiles.remove(proj)

        # Update and collide enemies
        enemies_to_remove = []
        for enemy in self.enemies:
            enemy.update(dt)
            if enemy.pos.distance_to(self.player_pos) < enemy.radius + 15: # 15 is player radius
                self.player_health -= 20
                self.combo = 0 # Reset combo on hit
                # SFX: player_hurt.wav
                self._create_explosion(self.player_pos, (255, 50, 50), 25)
                if enemy not in enemies_to_remove:
                    enemies_to_remove.append(enemy)
        for enemy in enemies_to_remove:
            self.enemies.remove(enemy)

        # Spawn enemies
        self.enemy_spawn_cooldown -= dt
        if self.enemy_spawn_cooldown <= 0:
            self.enemies.append(Enemy(current_enemy_health, random.uniform(30, 60), self.ARENA_CENTER))
            self.enemy_spawn_cooldown = current_spawn_rate
        
        # Update particles
        self.particles = [p for p in self.particles if p.update(dt)]
        
        # Update combo aura
        target_aura = min(80, self.combo * 2.5)
        self.combo_aura_size += (target_aura - self.combo_aura_size) * 0.1

        # --- Check Termination ---
        terminated = self.player_health <= 0 or self.timer <= 0
        truncated = False
        if terminated:
            self.game_over = True
            if self.player_health <= 0:
                reward = -100
            elif self.timer <= 0:
                reward = 100
                self.score += 1000 # Survival bonus
        
        self.steps += 1
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_player_pos(self):
        self.player_pos.x = self.ARENA_CENTER.x + self.ARENA_RADIUS * math.cos(self.player_angle)
        self.player_pos.y = self.ARENA_CENTER.y - self.ARENA_RADIUS * math.sin(self.player_angle)

    def _fire_weapon(self):
        self.fire_cooldown = 0.3
        target = self._find_nearest_enemy()
        if not target:
            return 0 # No reward if no target

        # SFX: fire_element.wav
        
        # Combo logic
        reward = 0
        if self.last_fired_weapon_idx != -1 and self.selected_weapon_idx != self.last_fired_weapon_idx:
            self.combo += 1
            reward += 0.5
        else:
            if self.last_fired_weapon_idx != -1: # Don't reset on first shot
                self.combo = 0

        self.last_fired_weapon_idx = self.selected_weapon_idx
        
        proj = Projectile(self.player_pos, target.pos, self.selected_weapon_idx, self.ELEMENTS[self.selected_weapon_idx])
        self.projectiles.append(proj)
        self._create_muzzle_flash()
        return reward

    def _find_nearest_enemy(self):
        if not self.enemies:
            return None
        
        nearest_enemy = min(self.enemies, key=lambda e: self.player_pos.distance_squared_to(e.pos))
        return nearest_enemy
    
    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 200)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = random.uniform(2, 6)
            lifetime = random.uniform(0.3, 0.8)
            self.particles.append(Particle(pos, vel, color, radius, lifetime))

    def _create_muzzle_flash(self):
        element_color = self.ELEMENTS[self.selected_weapon_idx]['color']
        direction = (self.ARENA_CENTER - self.player_pos).normalize()
        for i in range(10):
            vel = direction * random.uniform(100, 300) + pygame.Vector2(random.uniform(-50, 50), random.uniform(-50, 50))
            pos = self.player_pos + direction * 15
            radius = random.uniform(2, 5)
            lifetime = random.uniform(0.1, 0.3)
            self.particles.append(Particle(pos, vel, element_color, radius, lifetime))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "combo": self.combo, "health": self.player_health}

    def _render_game(self):
        # Arena outline
        pygame.gfxdraw.aacircle(self.screen, int(self.ARENA_CENTER.x), int(self.ARENA_CENTER.y), self.ARENA_RADIUS, self.COLOR_ARENA)
        
        # Combo aura
        if self.combo_aura_size > 1:
            aura_color = self.ELEMENTS[self.last_fired_weapon_idx % len(self.ELEMENTS)]['color'] if self.last_fired_weapon_idx != -1 else (255,255,255)
            alpha = min(100, int(self.combo_aura_size * 0.5))
            temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, int(self.player_pos.x), int(self.player_pos.y), int(15 + self.combo_aura_size), (*aura_color, alpha))
            self.screen.blit(temp_surf, (0,0), special_flags=pygame.BLEND_RGBA_ADD)

        # Particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)
            
        # Projectiles
        for proj in self.projectiles:
            proj.draw(self.screen)
            
        # Player
        p_size = 15
        p_angle_rad = self.player_angle
        p_points = [
            self.player_pos + pygame.Vector2(p_size, 0).rotate_rad(p_angle_rad + math.pi/2),
            self.player_pos + pygame.Vector2(-p_size, 0).rotate_rad(p_angle_rad + math.pi/2),
            self.player_pos + (self.ARENA_CENTER - self.player_pos).normalize() * p_size * 1.5
        ]
        pygame.gfxdraw.aapolygon(self.screen, p_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, p_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / 100)
        health_bar_rect = pygame.Rect(10, 10, 200, 20)
        health_fill_rect = pygame.Rect(10, 10, 200 * health_ratio, 20)
        pygame.draw.rect(self.screen, (80, 0, 0), health_bar_rect)
        pygame.draw.rect(self.screen, (255, 0, 0), health_fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, health_bar_rect, 1)

        # Timer
        timer_text = self.font_large.render(f"{self.timer:.1f}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 5))
        self.screen.blit(timer_text, timer_rect)

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 35))

        # Combo
        if self.combo > 1:
            combo_text = self.font_large.render(f"{self.combo}x COMBO", True, self.COLOR_UI_TEXT)
            combo_rect = combo_text.get_rect(midtop=(self.SCREEN_WIDTH / 2, 5))
            self.screen.blit(combo_text, combo_rect)

        # Weapon Selector
        for i, props in self.ELEMENTS.items():
            x_pos = self.SCREEN_WIDTH / 2 - 60 + i * 40
            y_pos = self.SCREEN_HEIGHT - 30
            color = props['color']
            
            if i == self.selected_weapon_idx:
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, (x_pos - 2, y_pos - 2, 24, 24), 2, border_radius=4)
                self._draw_glowing_circle(self.screen, color, (x_pos + 10, y_pos + 10), 10, 10)
            else:
                pygame.draw.rect(self.screen, color, (x_pos, y_pos, 20, 20), border_radius=4)
                
    @staticmethod
    def _draw_glowing_circle(surface, color, center, radius, glow_width):
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        for i in range(glow_width, 0, -1):
            alpha = int(100 * (1 - i / glow_width))
            pygame.gfxdraw.aacircle(surface, center[0], center[1], radius + i, (*color, alpha))
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run main block in headless mode. Exiting.")
    else:
        env = GameEnv()
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Elemental Combo Shooter")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            movement = 0 # None
            fire = 0 # Released
            shift = 0 # Released
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                fire = 1
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1
                
            action = [movement, fire, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                total_reward = 0
                obs, info = env.reset()
                # Add a small delay before restarting
                pygame.time.wait(2000)

            clock.tick(env.FPS)
            
        pygame.quit()