import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:56:04.270872
# Source Brief: brief_00784.md
# Brief Index: 784
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a steampunk platformer. The player controls a rusty robot
    that can switch between a repair mode (to fix broken platforms) and a combat mode
    (to defeat enemies). The goal is to collect scrap, defeat enemies, and ultimately
    take down the factory's boss.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Control a steampunk robot through a hazardous factory. Switch between combat mode to defeat enemies and "
        "repair mode to fix broken platforms on your way to the final boss."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press 'space' to fire your tool and 'shift' to switch between repair and combat modes."
    )
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRAVITY = 0.5
    PLAYER_ACCEL = 0.6
    PLAYER_FRICTION = -0.15
    PLAYER_JUMP_STRENGTH = -11
    PLAYER_MAX_SPEED_X = 6
    PLAYER_MAX_SPEED_Y = 12
    PLAYER_HEALTH_MAX = 100
    MAX_STEPS = 5000

    # Colors (Steampunk/Industrial Palette)
    COLOR_BG = (20, 25, 30)
    COLOR_BG_ACCENT = (35, 40, 50)
    COLOR_PLAYER = (0, 200, 255) # Bright Cyan
    COLOR_PLAYER_GLOW = (0, 100, 128)
    COLOR_ENEMY = (255, 60, 30) # Red-Orange
    COLOR_ENEMY_GLOW = (128, 30, 15)
    COLOR_BOSS = (255, 120, 0)
    COLOR_BOSS_GLOW = (180, 60, 0)
    COLOR_PLATFORM = (100, 90, 80) # Rusty Brown-Gray
    COLOR_PLATFORM_BROKEN = (120, 50, 40)
    COLOR_SCRAP = (255, 215, 0) # Gold
    COLOR_REPAIR_BEAM = (0, 220, 255)
    COLOR_COMBAT_BEAM = (255, 100, 80)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR_BG = (80, 20, 20)
    COLOR_HEALTH_BAR_FG = (200, 40, 40)

    # --- Helper Classes ---

    class Particle:
        def __init__(self, pos, vel, color, size, lifespan):
            self.pos = list(pos)
            self.vel = list(vel)
            self.color = color
            self.size = size
            self.lifespan = lifespan

        def update(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            self.lifespan -= 1
            self.size = max(0, self.size - 0.1)

        def draw(self, surface):
            if self.size > 0:
                pygame.draw.circle(surface, self.color, (int(self.pos[0]), int(self.pos[1])), int(self.size))

    class Projectile:
        def __init__(self, pos, direction, p_type):
            self.pos = list(pos)
            self.type = p_type # 'repair' or 'combat'
            self.speed = 15
            self.vel = [self.speed * direction, 0]
            self.lifespan = 40 # 640 / 15
            self.size = 3 if p_type == 'repair' else 4
            self.color = GameEnv.COLOR_REPAIR_BEAM if p_type == 'repair' else GameEnv.COLOR_COMBAT_BEAM

        def update(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            self.lifespan -= 1

        def draw(self, surface):
            end_pos = (self.pos[0] - self.vel[0] * 0.5, self.pos[1] - self.vel[1] * 0.5)
            pygame.draw.line(surface, self.color, (int(self.pos[0]), int(self.pos[1])), (int(end_pos[0]), int(end_pos[1])), self.size * 2)
            pygame.draw.circle(surface, (255, 255, 255), (int(self.pos[0]), int(self.pos[1])), self.size)
            
        @property
        def rect(self):
            return pygame.Rect(self.pos[0] - self.size, self.pos[1] - self.size, self.size * 2, self.size * 2)

    class Player:
        def __init__(self):
            self.size = pygame.Vector2(24, 32)
            self.reset()

        def reset(self):
            self.pos = pygame.Vector2(100, 200)
            self.vel = pygame.Vector2(0, 0)
            self.health = GameEnv.PLAYER_HEALTH_MAX
            self.on_ground = False
            self.is_combat_mode = False
            self.facing_direction = 1 # 1 for right, -1 for left
            self.fire_cooldown = 0
            self.invulnerability_timer = 0

        def update(self, platforms):
            # Apply friction
            self.vel.x += self.vel.x * GameEnv.PLAYER_FRICTION
            if abs(self.vel.x) < 0.1: self.vel.x = 0

            # Apply gravity
            self.vel.y += GameEnv.GRAVITY
            
            # Clamp velocity
            self.vel.x = max(-GameEnv.PLAYER_MAX_SPEED_X, min(GameEnv.PLAYER_MAX_SPEED_X, self.vel.x))
            self.vel.y = max(-GameEnv.PLAYER_MAX_SPEED_Y, min(GameEnv.PLAYER_MAX_SPEED_Y, self.vel.y))

            # Move and collide
            self.pos.x += self.vel.x
            self.handle_collisions('x', platforms)
            self.pos.y += self.vel.y
            self.on_ground = False
            self.handle_collisions('y', platforms)
            
            # Keep in bounds
            self.pos.x = max(self.size.x / 2, min(GameEnv.WIDTH - self.size.x / 2, self.pos.x))
            if self.pos.y > GameEnv.HEIGHT - self.size.y / 2:
                self.pos.y = GameEnv.HEIGHT - self.size.y / 2
                self.vel.y = 0
                self.on_ground = True

            if self.fire_cooldown > 0: self.fire_cooldown -= 1
            if self.invulnerability_timer > 0: self.invulnerability_timer -= 1
        
        @property
        def rect(self):
            return pygame.Rect(self.pos.x - self.size.x / 2, self.pos.y - self.size.y / 2, self.size.x, self.size.y)
        
        def handle_collisions(self, axis, platforms):
            player_rect = self.rect
            for plat in platforms:
                if not plat.is_broken and player_rect.colliderect(plat.rect):
                    if axis == 'x':
                        if self.vel.x > 0: player_rect.right = plat.rect.left
                        if self.vel.x < 0: player_rect.left = plat.rect.right
                        self.pos.x = player_rect.centerx
                        self.vel.x = 0
                    if axis == 'y':
                        if self.vel.y > 0:
                            player_rect.bottom = plat.rect.top
                            self.on_ground = True
                            self.vel.y = 0
                        if self.vel.y < 0:
                            player_rect.top = plat.rect.bottom
                            self.vel.y = 0
                        self.pos.y = player_rect.centery

        def take_damage(self, amount):
            if self.invulnerability_timer == 0:
                self.health = max(0, self.health - amount)
                self.invulnerability_timer = 60 # 2 seconds of invulnerability
                return True
            return False

        def draw(self, surface):
            # Glow effect
            if self.invulnerability_timer > 0 and (self.invulnerability_timer // 3) % 2 == 0:
                return # Flicker when invincible
            
            glow_size = int(self.size.y * 0.8)
            glow_alpha = 80
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), glow_size, (*GameEnv.COLOR_PLAYER_GLOW, glow_alpha))
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(glow_size * 0.7), (*GameEnv.COLOR_PLAYER_GLOW, glow_alpha + 20))

            # Main body
            player_rect = self.rect
            pygame.draw.rect(surface, GameEnv.COLOR_PLAYER, player_rect, border_radius=4)
            # "Eye"
            eye_pos_x = self.pos.x + self.facing_direction * 5
            eye_pos_y = self.pos.y - 5
            pygame.draw.circle(surface, (255, 255, 255), (int(eye_pos_x), int(eye_pos_y)), 3)

    class Platform:
        def __init__(self, x, y, w, h, broken=False):
            self.rect = pygame.Rect(x, y, w, h)
            self.is_broken = broken
        
        def repair(self):
            if self.is_broken:
                self.is_broken = False
                return True
            return False

        def draw(self, surface):
            color = GameEnv.COLOR_PLATFORM_BROKEN if self.is_broken else GameEnv.COLOR_PLATFORM
            pygame.draw.rect(surface, color, self.rect)
            if self.is_broken:
                pygame.draw.rect(surface, GameEnv.COLOR_ENEMY, self.rect, 2)

    class Enemy:
        def __init__(self, x, y, patrol_dist, is_boss=False):
            self.is_boss = is_boss
            self.size = pygame.Vector2(40, 40) if is_boss else pygame.Vector2(28, 28)
            self.pos = pygame.Vector2(x, y - self.size.y / 2)
            self.vel = pygame.Vector2(random.choice([-1, 1]), 0)
            self.patrol_start_x = x - patrol_dist
            self.patrol_end_x = x + patrol_dist
            self.health = 200 if is_boss else 30
            self.color = GameEnv.COLOR_BOSS if is_boss else GameEnv.COLOR_ENEMY
            self.glow_color = GameEnv.COLOR_BOSS_GLOW if is_boss else GameEnv.COLOR_ENEMY_GLOW
            self.speed = 1.0

        def update(self):
            self.pos.x += self.vel.x * self.speed
            if self.pos.x <= self.patrol_start_x or self.pos.x >= self.patrol_end_x:
                self.vel.x *= -1
        
        @property
        def rect(self):
            return pygame.Rect(self.pos.x - self.size.x / 2, self.pos.y - self.size.y / 2, self.size.x, self.size.y)

        def take_damage(self, amount):
            self.health = max(0, self.health - amount)

        def draw(self, surface):
            # Glow
            glow_size = int(self.size.y * 0.9)
            glow_alpha = 100
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), glow_size, (*self.glow_color, glow_alpha))
            
            # Body
            body_rect = self.rect
            pygame.draw.rect(surface, self.color, body_rect, border_radius=5)
            # Eye
            eye_color = (255, 255, 255)
            eye_pos = (int(self.pos.x), int(self.pos.y - 5))
            pygame.draw.circle(surface, eye_color, eye_pos, 4)
            pygame.draw.circle(surface, (0,0,0), eye_pos, 2)

    class Scrap:
        def __init__(self, pos):
            self.pos = pygame.Vector2(pos)
            self.vel = pygame.Vector2(random.uniform(-2, 2), random.uniform(-4, -1))
            self.size = 6

        def update(self):
            self.vel.y += GameEnv.GRAVITY * 0.5
            self.pos += self.vel
            if self.pos.y > GameEnv.HEIGHT - self.size:
                self.pos.y = GameEnv.HEIGHT - self.size
                self.vel.y = 0
                self.vel.x *= 0.8

        @property
        def rect(self):
            return pygame.Rect(self.pos.x - self.size, self.pos.y - self.size, self.size * 2, self.size * 2)

        def draw(self, surface):
            points = []
            for i in range(5):
                angle = math.radians(i * 72 + 45)
                points.append((self.pos.x + self.size * math.cos(angle), self.pos.y + self.size * math.sin(angle)))
            pygame.gfxdraw.aapolygon(surface, points, GameEnv.COLOR_SCRAP)
            pygame.gfxdraw.filled_polygon(surface, points, GameEnv.COLOR_SCRAP)


    # --- Gym Env Implementation ---
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Game state variables
        self.player = self.Player()
        self.platforms = []
        self.enemies = []
        self.boss = None
        self.projectiles = []
        self.particles = []
        self.scraps = []
        
        self.steps = 0
        self.score = 0
        self.scrap_count = 0
        self.bosses_defeated = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.np_random = None

    def _create_particles(self, pos, count, color, speed_range, lifespan_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(*speed_range)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(*lifespan_range)
            size = self.np_random.uniform(2, 5)
            self.particles.append(self.Particle(pos, vel, color, size, lifespan))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.scrap_count = 0
        self.bosses_defeated = 0
        
        self.player.reset()
        
        # Level Generation
        self.platforms = [
            # Floor
            self.Platform(0, 380, self.WIDTH, 20),
            # Starting area
            self.Platform(0, 300, 200, 20),
            # Gap that needs repair
            self.Platform(280, 250, 100, 20, broken=True),
            # Upper path
            self.Platform(450, 180, 150, 20),
            # Boss arena
            self.Platform(250, 100, 390, 20),
        ]
        
        self.enemies = [self.Enemy(500, 380, 100)]
        self.boss = self.Enemy(450, 100, 150, is_boss=True)
        self.enemies.append(self.boss)
        
        self.projectiles.clear()
        self.particles.clear()
        self.scraps.clear()
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input and Player Actions ---
        if movement == 1 and self.player.on_ground: # Jump
            self.player.vel.y = self.PLAYER_JUMP_STRENGTH
            self._create_particles(self.player.rect.midbottom, 5, (150, 150, 150), (0.5, 2), (10, 20))
        # Down action (no drop-through platforms in this design)
        if movement == 3: # Left
            self.player.vel.x -= self.PLAYER_ACCEL
            self.player.facing_direction = -1
        if movement == 4: # Right
            self.player.vel.x += self.PLAYER_ACCEL
            self.player.facing_direction = 1
            
        # Handle single-press for space and shift
        fire_action = space_held and not self.prev_space_held
        switch_mode_action = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        if switch_mode_action:
            self.player.is_combat_mode = not self.player.is_combat_mode
            color = self.COLOR_COMBAT_BEAM if self.player.is_combat_mode else self.COLOR_REPAIR_BEAM
            self._create_particles(self.player.pos, 15, color, (1, 3), (15, 25))

        if fire_action and self.player.fire_cooldown == 0:
            p_type = 'combat' if self.player.is_combat_mode else 'repair'
            proj_start_pos = (self.player.pos.x + self.player.facing_direction * 20, self.player.pos.y)
            self.projectiles.append(self.Projectile(proj_start_pos, self.player.facing_direction, p_type))
            self.player.fire_cooldown = 15 # Cooldown in steps

        # --- Update Game State ---
        self.steps += 1
        self.player.update(self.platforms)
        
        for enemy in self.enemies:
            enemy.speed = 1.0 + 0.05 * self.bosses_defeated
            enemy.update()
            if self.player.rect.colliderect(enemy.rect):
                if self.player.take_damage(25 if enemy.is_boss else 10):
                    reward -= 0.1
                    self._create_particles(self.player.pos, 10, self.COLOR_ENEMY, (1, 4), (20, 40))
        
        for scrap in self.scraps[:]:
            scrap.update()
            if self.player.rect.colliderect(scrap.rect):
                self.scrap_count += 1
                reward += 0.1
                self.scraps.remove(scrap)
                self._create_particles(scrap.pos, 5, self.COLOR_SCRAP, (0.5, 1.5), (10, 15))
        
        dead_enemies = []
        for proj in self.projectiles[:]:
            proj.update()
            if proj.lifespan <= 0:
                self.projectiles.remove(proj)
                continue
            
            removed_proj = False
            if proj.type == 'repair':
                for plat in self.platforms:
                    if plat.is_broken and proj.rect.colliderect(plat.rect):
                        if plat.repair():
                            reward += 1.0
                            self._create_particles(plat.rect.center, 30, (255, 255, 100), (1, 3), (20, 30))
                        self.projectiles.remove(proj)
                        removed_proj = True
                        break
            
            elif proj.type == 'combat':
                for enemy in self.enemies:
                    if proj.rect.colliderect(enemy.rect):
                        enemy.take_damage(15)
                        self._create_particles(enemy.pos, 5, self.COLOR_COMBAT_BEAM, (0.5, 2), (5, 10))
                        if enemy.health <= 0:
                            dead_enemies.append(enemy)
                        self.projectiles.remove(proj)
                        removed_proj = True
                        break
            if removed_proj:
                continue
        
        for enemy in dead_enemies:
            if enemy in self.enemies: self.enemies.remove(enemy)
            self._create_particles(enemy.pos, 50, enemy.color, (1, 5), (30, 60))
            for _ in range(self.np_random.integers(2, 6)):
                self.scraps.append(self.Scrap(enemy.pos))
            
            if enemy.is_boss:
                reward += 20.0
                self.bosses_defeated += 1
                if self.bosses_defeated == 1: # Final boss
                    reward += 100.0
                    terminated = True
            else:
                reward += 5.0
        
        for particle in self.particles[:]:
            particle.update()
            if particle.lifespan <= 0:
                self.particles.remove(particle)

        # --- Spawn new enemies ---
        if self.steps > 0 and self.steps % 500 == 0:
            if len(self.enemies) < 5:
                spawn_x = self.np_random.choice([50, self.WIDTH - 50])
                spawn_y = 380
                self.enemies.append(self.Enemy(spawn_x, spawn_y, 50))
        
        # --- Check Termination Conditions ---
        if self.player.health <= 0:
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_background(self):
        # Static background elements for industrial feel
        # Vertical pipes
        pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (50, 0, 15, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (200, 100, 10, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (550, -50, 20, self.HEIGHT))
        # Horizontal girders
        pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (0, 50, self.WIDTH, 8))
        pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (0, 200, self.WIDTH, 12))

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player.health / self.PLAYER_HEALTH_MAX
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_small.render(f"HP", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 11))

        # Mode Icon
        mode_rect = pygame.Rect(10, 35, 30, 30)
        pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, mode_rect, border_radius=3)
        if self.player.is_combat_mode: # Crosshair
            pygame.draw.circle(self.screen, self.COLOR_COMBAT_BEAM, mode_rect.center, 10, 2)
            pygame.draw.line(self.screen, self.COLOR_COMBAT_BEAM, (mode_rect.centerx, mode_rect.top + 5), (mode_rect.centerx, mode_rect.bottom - 5), 2)
            pygame.draw.line(self.screen, self.COLOR_COMBAT_BEAM, (mode_rect.left + 5, mode_rect.centery), (mode_rect.right - 5, mode_rect.centery), 2)
        else: # Wrench
            pygame.draw.circle(self.screen, self.COLOR_REPAIR_BEAM, (mode_rect.centerx - 5, mode_rect.centery - 5), 8, 3)
            pygame.draw.rect(self.screen, self.COLOR_REPAIR_BEAM, (mode_rect.centerx, mode_rect.centery, 10, 5))

        # Scrap Count
        scrap_text = self.font_large.render(f"{self.scrap_count}", True, self.COLOR_SCRAP)
        text_rect = scrap_text.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(scrap_text, text_rect)
        # Scrap Icon
        scrap_icon_pos = (text_rect.left - 20, text_rect.centery)
        scrap_icon = self.Scrap(scrap_icon_pos)
        scrap_icon.size = 10
        scrap_icon.draw(self.screen)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        
        for plat in self.platforms: plat.draw(self.screen)
        for scrap in self.scraps: scrap.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        self.player.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        for particle in self.particles: particle.draw(self.screen)
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player.health,
            "scrap": self.scrap_count,
            "mode": "combat" if self.player.is_combat_mode else "repair",
            "boss_health": self.boss.health if self.boss else 0,
        }
        
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is NOT part of the required environment implementation
    
    # Un-comment the following line to run with display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Rusty Robot Adventure")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset")
    print("Q: Quit")
    
    while True:
        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        quit_game = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quit_game = True
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=42)
                    total_reward = 0
                    print("--- Environment Reset ---")
        if quit_game:
            break

        if terminated or truncated:
            print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset(seed=42)
            total_reward = 0

        clock.tick(GameEnv.metadata["render_fps"])

    print(f"\nQuitting. Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()