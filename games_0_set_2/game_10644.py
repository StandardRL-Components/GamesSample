import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:42:36.125667
# Source Brief: brief_00644.md
# Brief Index: 644
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque, defaultdict

# --- Helper Classes for Game Objects ---

class Particle:
    """A single particle for effects like explosions, trails, etc."""
    def __init__(self, x, y, vx, vy, color, size, lifetime):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.life = float(lifetime)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.size = max(0, self.size * (self.life / self.lifetime))

    def draw(self, surface, scroll_x):
        if self.life > 0:
            pos = (int(self.x - scroll_x), int(self.y))
            pygame.draw.circle(surface, self.color, pos, int(self.size))

class Projectile:
    """A projectile fired by the player or an enemy."""
    def __init__(self, x, y, vx, owner, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = 0
        self.owner = owner
        self.color = color
        self.radius = 5 if owner == 'player' else 7
        self.trail = deque(maxlen=10)

    def update(self):
        self.trail.append((self.x, self.y))
        self.x += self.vx
        self.y += self.vy

    def draw(self, surface, scroll_x):
        # Draw trail
        for i, pos in enumerate(self.trail):
            alpha = int(255 * (i / len(self.trail)))
            trail_color = self.color + (alpha,)
            temp_surf = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, trail_color, (self.radius, self.radius), self.radius * (i / len(self.trail)))
            surface.blit(temp_surf, (int(pos[0] - scroll_x - self.radius), int(pos[1] - self.radius)))

        # Draw main projectile
        pos = (int(self.x - scroll_x), int(self.y))
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], self.radius, self.color)

class PowerUpComponent:
    """A collectible component for crafting."""
    def __init__(self, x, y, component_type):
        self.x = x
        self.y = y
        self.type = component_type
        self.color_map = {'Y': (255, 255, 0), 'C': (0, 255, 255), 'M': (255, 0, 255)}
        self.color = self.color_map[self.type]
        self.radius = 8
        self.bob_angle = random.uniform(0, 2 * math.pi)
        self.bob_speed = 0.05
        self.initial_y = y

    def update(self):
        self.bob_angle += self.bob_speed
        self.y = self.initial_y + math.sin(self.bob_angle) * 5

    def draw(self, surface, scroll_x):
        pos = (int(self.x - scroll_x), int(self.y))
        # Glow effect
        glow_radius = int(self.radius * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.color + (60,), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))
        # Core
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], self.radius, self.color)

class Enemy:
    """An enemy entity."""
    def __init__(self, x, y, speed_multiplier, attack_freq_multiplier):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 40
        self.max_health = 3
        self.health = self.max_health
        self.vx = (random.choice([-1, 1]) * 1.5) * speed_multiplier
        self.vy = 0
        self.attack_cooldown = 120 / attack_freq_multiplier
        self.attack_timer = random.randint(0, int(self.attack_cooldown))
        self.damage_flash_timer = 0
        self.color = (255, 50, 50) # Red

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def take_damage(self, amount):
        self.health -= amount
        self.damage_flash_timer = 10
        return self.health <= 0

    def update(self, player_x, player_y):
        if self.damage_flash_timer > 0:
            self.damage_flash_timer -= 1

        # Simple AI: move back and forth, shoot at player
        self.x += self.vx
        if self.x < 0 or self.x > 2000: # World bounds
            self.vx *= -1

        self.attack_timer -= 1
        should_shoot = self.attack_timer <= 0
        if should_shoot:
            self.attack_timer = self.attack_cooldown

        return should_shoot

    def draw(self, surface, scroll_x):
        rect = pygame.Rect(int(self.x - scroll_x), int(self.y), self.width, self.height)
        color = (255, 255, 255) if self.damage_flash_timer > 0 else self.color
        
        # Glitch effect
        if random.random() < 0.1:
            glitch_rect = rect.copy()
            glitch_rect.x += random.randint(-5, 5)
            pygame.draw.rect(surface, (0, 255, 0, 50), glitch_rect)
        
        pygame.draw.rect(surface, color, rect)
        
        # Health bar
        if self.health < self.max_health:
            health_pct = self.health / self.max_health
            pygame.draw.rect(surface, (50, 50, 50), (rect.left, rect.top - 8, self.width, 5))
            pygame.draw.rect(surface, (0, 255, 0), (rect.left, rect.top - 8, int(self.width * health_pct), 5))


class Player:
    """The player character."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 25
        self.height = 50
        self.vx = 0
        self.vy = 0
        self.max_health = 100
        self.health = self.max_health
        self.on_ground = False
        self.last_move_direction = 1 # 1 for right, -1 for left
        self.color = (0, 150, 255) # Bright Blue

        self.shoot_cooldown = 15 # frames
        self.shoot_timer = 0
        
        # Power-ups
        self.active_powerups = {} # e.g., {"shield": 300, "rapid_fire": 600}

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, movement, jump_action, floor_y):
        # Horizontal movement
        accel = 0.8
        friction = 0.9
        if movement == 3: # Left
            self.vx -= accel
            self.last_move_direction = -1
        elif movement == 4: # Right
            self.vx += accel
            self.last_move_direction = 1
        self.vx *= friction
        if abs(self.vx) < 0.1: self.vx = 0
        self.vx = np.clip(self.vx, -6, 6)
        self.x += self.vx

        # Vertical movement (gravity and jump)
        if jump_action and self.on_ground:
            self.vy = -12 # Jump strength
            # sfx: player_jump.wav
            self.on_ground = False

        self.vy += 0.6 # Gravity
        self.y += self.vy

        if self.y + self.height >= floor_y:
            self.y = floor_y - self.height
            self.vy = 0
            self.on_ground = True
        
        # Update timers
        if self.shoot_timer > 0:
            self.shoot_timer -= 1
        
        for powerup, duration in list(self.active_powerups.items()):
            if duration > 0:
                self.active_powerups[powerup] -= 1
            else:
                del self.active_powerups[powerup]

    def shoot(self):
        cooldown = self.shoot_cooldown
        if "rapid_fire" in self.active_powerups:
            cooldown = 5
        
        if self.shoot_timer == 0:
            self.shoot_timer = cooldown
            # sfx: player_shoot.wav
            return True
        return False

    def take_damage(self, amount):
        if "shield" in self.active_powerups:
            # sfx: shield_hit.wav
            return False # No damage taken
        
        self.health = max(0, self.health - amount)
        # sfx: player_damage.wav
        return self.health <= 0

    def draw(self, surface, scroll_x):
        pos = (int(self.x - scroll_x), int(self.y))
        rect = pygame.Rect(pos[0], pos[1], self.width, self.height)
        
        # Shield effect
        if "shield" in self.active_powerups:
            shield_radius = max(self.width, self.height) // 2 + 10
            shield_center = (rect.centerx, rect.centery)
            alpha = 100 + int(math.sin(pygame.time.get_ticks() * 0.01) * 50)
            temp_surf = pygame.Surface((shield_radius*2, shield_radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, shield_radius, shield_radius, shield_radius, (0, 200, 255, alpha))
            pygame.gfxdraw.aacircle(temp_surf, shield_radius, shield_radius, shield_radius, (200, 255, 255, alpha))
            surface.blit(temp_surf, (shield_center[0] - shield_radius, shield_center[1] - shield_radius))
        
        # Player Glow
        glow_radius = int(self.height * 0.7)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.color + (80,), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (rect.centerx - glow_radius, rect.centery - glow_radius))

        # Player Body
        pygame.draw.rect(surface, self.color, rect, border_radius=3)
        
        # "Eye" to show direction
        eye_x = rect.centerx + self.last_move_direction * 5
        eye_y = rect.y + 15
        pygame.draw.circle(surface, (255, 255, 255), (eye_x, eye_y), 3)

# --- Main Gymnasium Environment Class ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Battle waves of glitchy enemies in a cyber-dystopian world. "
        "Collect components to craft powerful upgrades and survive as long as possible."
    )
    user_guide = (
        "Controls: Use ←→ to move and ↑ to jump. Press space to shoot and hold shift to craft available power-ups."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = 2000
        self.FLOOR_Y = 360

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_PLAYER_PROJ = (0, 255, 255)
        self.COLOR_ENEMY_PROJ = (255, 100, 0)
        
        # Game state variables are initialized in reset()
        self.player = None
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.powerup_components = []
        self.component_inventory = defaultdict(int)
        self.unlocked_powerups = set()
        self.craftable_powerups = []
        self.world_scroll_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_steps = 5000
        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 3.0 # seconds
        self.enemy_speed_multiplier = 1.0
        self.enemy_attack_freq_multiplier = 1.0
        self.reward_flags = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = Player(self.SCREEN_WIDTH / 4, self.FLOOR_Y - 50)
        self.world_scroll_x = 0
        
        self.enemies.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.powerup_components.clear()
        
        self.component_inventory = defaultdict(int)
        self.unlocked_powerups = {"shield"} # Start with shield unlocked
        self.craftable_powerups = []
        
        # Difficulty scaling reset
        self.enemy_spawn_rate = 3.0
        self.enemy_speed_multiplier = 1.0
        self.enemy_attack_freq_multiplier = 1.0
        self.enemy_spawn_timer = self.enemy_spawn_rate * 30
        
        # Reward flags reset
        self.reward_flags = {"survived_60s": False, "survived_120s": False}
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1
        shift_held = action[2] == 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward += 0.1 # Survival reward

        # --- UPDATE GAME LOGIC ---

        # 1. Player actions
        self.player.update(movement, movement == 1, self.FLOOR_Y)
        if space_held and self.player.shoot():
            proj_x = self.player.x + self.player.width / 2
            proj_y = self.player.y + self.player.height / 2
            proj_vx = 12 * self.player.last_move_direction
            self.projectiles.append(Projectile(proj_x, proj_y, proj_vx, 'player', self.COLOR_PLAYER_PROJ))
            # Muzzle flash particles
            for _ in range(5):
                p_vx = proj_vx * 0.2 + random.uniform(-2, 2)
                p_vy = random.uniform(-2, 2)
                self.particles.append(Particle(proj_x, proj_y, p_vx, p_vy, self.COLOR_PLAYER_PROJ, random.randint(2, 4), 10))
        
        if shift_held:
            self._craft_powerup()

        # 2. Update enemies and handle their attacks
        new_projectiles = []
        for enemy in self.enemies:
            if enemy.update(self.player.x, self.player.y):
                # sfx: enemy_shoot.wav
                angle_to_player = math.atan2(self.player.y - enemy.y, (self.player.x - self.world_scroll_x) - (enemy.x - self.world_scroll_x))
                proj_vx = math.cos(angle_to_player) * 6
                proj_vy = math.sin(angle_to_player) * 6 # Not used in this brief, but good practice
                new_projectiles.append(Projectile(enemy.x + enemy.width/2, enemy.y + enemy.height/2, proj_vx, 'enemy', self.COLOR_ENEMY_PROJ))
        self.projectiles.extend(new_projectiles)

        # 3. Update projectiles, particles, components
        for p in self.projectiles: p.update()
        for p in self.particles: p.update()
        for c in self.powerup_components: c.update()

        # 4. Handle collisions
        # Player projectiles vs Enemies
        for proj in self.projectiles[:]:
            if proj.owner == 'player':
                for enemy in self.enemies[:]:
                    if enemy.get_rect().colliderect(proj.x - proj.radius, proj.y - proj.radius, proj.radius*2, proj.radius*2):
                        reward += 1.0 # Hit reward
                        if enemy.take_damage(1):
                            # sfx: enemy_explode.wav
                            reward += 5.0 # Defeat reward
                            self.score += 10
                            self._create_explosion(enemy.x + enemy.width/2, enemy.y + enemy.height/2, enemy.color)
                            # Chance to drop a component
                            if random.random() < 0.7:
                                comp_type = random.choice(['Y', 'C', 'M'])
                                self.powerup_components.append(PowerUpComponent(enemy.x, enemy.y + enemy.height, comp_type))
                            self.enemies.remove(enemy)
                        if proj in self.projectiles: self.projectiles.remove(proj)
                        break
        
        # Enemy projectiles vs Player
        player_rect_world = self.player.get_rect()
        for proj in self.projectiles[:]:
            if proj.owner == 'enemy':
                if player_rect_world.colliderect(proj.x - proj.radius, proj.y - proj.radius, proj.radius*2, proj.radius*2):
                    if self.player.take_damage(10):
                        self.game_over = True
                        reward = -10.0 # Game over penalty
                    self._create_explosion(proj.x, proj.y, self.COLOR_ENEMY_PROJ)
                    if proj in self.projectiles: self.projectiles.remove(proj)

        # Player vs Power-up components
        for comp in self.powerup_components[:]:
            if player_rect_world.colliderect(comp.x - comp.radius, comp.y - comp.radius, comp.radius*2, comp.radius*2):
                # sfx: collect_item.wav
                reward += 2.0
                self.component_inventory[comp.type] += 1
                self.powerup_components.remove(comp)
                self._update_craftable_powerups()


        # 5. Clean up off-screen/dead objects
        self.projectiles = [p for p in self.projectiles if 0 < p.x < self.WORLD_WIDTH and p.y < self.SCREEN_HEIGHT]
        self.particles = [p for p in self.particles if p.life > 0]
        
        # 6. Update world scroll to keep player centered
        self.world_scroll_x = self.player.x - self.SCREEN_WIDTH / 4
        self.world_scroll_x = np.clip(self.world_scroll_x, 0, self.WORLD_WIDTH - self.SCREEN_WIDTH)

        # 7. Difficulty scaling and enemy spawning
        self._update_difficulty()
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self._spawn_enemy()
            self.enemy_spawn_timer = self.enemy_spawn_rate * 30 # 30 FPS

        # 8. Check for time-based rewards
        time_survived_s = self.steps / 30.0
        if time_survived_s >= 60 and not self.reward_flags["survived_60s"]:
            reward += 50
            self.reward_flags["survived_60s"] = True
        if time_survived_s >= 120 and not self.reward_flags["survived_120s"]:
            reward += 100
            self.reward_flags["survived_120s"] = True

        # 9. Check for termination
        terminated = self.game_over or self.steps >= self.max_steps
        if self.game_over:
            self._create_explosion(self.player.x + self.player.width/2, self.player.y + self.player.height/2, self.player.color)
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_difficulty(self):
        time_s = self.steps / 30.0
        self.enemy_spawn_rate = max(0.5, 3.0 - time_s * 0.01) # Faster spawning
        
        if time_s > 60:
            self.enemy_speed_multiplier = 1.2
            self.enemy_attack_freq_multiplier = 1.2
            self.unlocked_powerups.add("rapid_fire")
            self._update_craftable_powerups()
        elif time_s > 120:
            self.enemy_speed_multiplier = 1.5
            self.enemy_attack_freq_multiplier = 1.5

    def _spawn_enemy(self):
        # Spawn off-screen
        side = random.choice([-1, 1])
        if side == -1:
            x = self.world_scroll_x - 50
        else:
            x = self.world_scroll_x + self.SCREEN_WIDTH + 50
        x = np.clip(x, 0, self.WORLD_WIDTH)
        y = self.FLOOR_Y - 40 # Enemy height
        self.enemies.append(Enemy(x, y, self.enemy_speed_multiplier, self.enemy_attack_freq_multiplier))

    def _create_explosion(self, x, y, color):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 6)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            size = random.randint(3, 7)
            lifetime = random.randint(20, 40)
            self.particles.append(Particle(x, y, vx, vy, color, size, lifetime))

    def _update_craftable_powerups(self):
        self.craftable_powerups = []
        # Define recipes
        recipes = {
            "shield": {'Y': 1, 'C': 1, 'M': 1},
            "rapid_fire": {'Y': 2, 'C': 1}
        }
        for name, recipe in recipes.items():
            if name in self.unlocked_powerups:
                can_craft = all(self.component_inventory[comp] >= count for comp, count in recipe.items())
                if can_craft:
                    self.craftable_powerups.append((name, recipe))

    def _craft_powerup(self):
        if not self.craftable_powerups:
            return
        
        # Craft the first available power-up
        name, recipe = self.craftable_powerups[0]
        
        # Consume components
        for comp, count in recipe.items():
            self.component_inventory[comp] -= count
        
        # sfx: craft_powerup.wav
        # Activate power-up
        if name == "shield":
            self.player.active_powerups["shield"] = 300 # 10 seconds
        elif name == "rapid_fire":
            self.player.active_powerups["rapid_fire"] = 600 # 20 seconds
            
        self._update_craftable_powerups()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Scrolling grid
        grid_size = 50
        start_x = -int(self.world_scroll_x % grid_size)
        for x in range(start_x, self.SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        
        # Floor
        pygame.draw.line(self.screen, (150, 150, 255), (0, self.FLOOR_Y), (self.SCREEN_WIDTH, self.FLOOR_Y), 2)
        
        # Glitch/distortion lines
        if random.random() < 0.2:
            y = random.randint(0, self.SCREEN_HEIGHT)
            h = random.randint(1, 3)
            pygame.draw.rect(self.screen, (255, 255, 255, 30), (0, y, self.SCREEN_WIDTH, h))

    def _render_game(self):
        # Draw in order: components, enemies, player, projectiles, particles
        for c in self.powerup_components: c.draw(self.screen, self.world_scroll_x)
        for e in self.enemies: e.draw(self.screen, self.world_scroll_x)
        self.player.draw(self.screen, self.world_scroll_x)
        for p in self.projectiles: p.draw(self.screen, self.world_scroll_x)
        for p in self.particles: p.draw(self.screen, self.world_scroll_x)

    def _render_ui(self):
        # Health Bar
        health_pct = self.player.health / self.player.max_health
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, (50, 0, 0), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 200, 50), (10, 10, int(bar_width * health_pct), bar_height))
        health_text = self.font_ui.render(f"HP: {self.player.health}", True, (255, 255, 255))
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Power-up components inventory
        y_text = self.font_ui.render(f"Y: {self.component_inventory['Y']}", True, (255, 255, 0))
        c_text = self.font_ui.render(f"C: {self.component_inventory['C']}", True, (0, 255, 255))
        m_text = self.font_ui.render(f"M: {self.component_inventory['M']}", True, (255, 0, 255))
        
        inv_width = y_text.get_width() + c_text.get_width() + m_text.get_width() + 40
        start_x = (self.SCREEN_WIDTH - inv_width) / 2
        
        self.screen.blit(y_text, (start_x, self.SCREEN_HEIGHT - 30))
        self.screen.blit(c_text, (start_x + y_text.get_width() + 20, self.SCREEN_HEIGHT - 30))
        self.screen.blit(m_text, (start_x + y_text.get_width() + c_text.get_width() + 40, self.SCREEN_HEIGHT - 30))
        
        # Crafting prompt
        if self.craftable_powerups:
            craft_text = self.font_ui.render(f"SHIFT to craft: {self.craftable_powerups[0][0].upper()}", True, (255, 255, 255))
            self.screen.blit(craft_text, ((self.SCREEN_WIDTH - craft_text.get_width()) / 2, self.SCREEN_HEIGHT - 60))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_big.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player.health,
            "inventory": dict(self.component_inventory),
            "active_powerups": list(self.player.active_powerups.keys()),
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Cyber Glitch Fighter")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1 # up
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2 # down
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3 # left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4 # right
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already the rendered screen, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        clock.tick(30) # Lock to 30 FPS
        
    print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()