import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:16:48.037640
# Source Brief: brief_00376.md
# Brief Index: 376
# """import gymnasium as gym

# --- Constants ---
# Game
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
FPS = 30
MAX_STEPS = 5000
LEVEL_WIDTH_FACTOR = 15
LEVEL_WIDTH = SCREEN_WIDTH * LEVEL_WIDTH_FACTOR

# Colors (Cyberpunk Neon)
COLOR_BG = (10, 5, 25)
COLOR_GRID_MAJOR = (30, 20, 50)
COLOR_GRID_MINOR = (20, 15, 40)
COLOR_PLAYER = (0, 200, 255)
COLOR_PLAYER_GLOW = (0, 100, 155)
COLOR_CLONE = (100, 220, 255)
COLOR_CLONE_GLOW = (50, 110, 155)
COLOR_ENEMY = (255, 50, 50)
COLOR_ENEMY_GLOW = (155, 25, 25)
COLOR_PLATFORM = (0, 255, 100)
COLOR_PLATFORM_GLOW = (0, 100, 40)
COLOR_RESOURCE = (255, 255, 0)
COLOR_RESOURCE_GLOW = (150, 150, 0)
COLOR_PLAYER_PROJ = (200, 220, 255)
COLOR_ENEMY_PROJ = (255, 150, 150)
COLOR_WHITE = (240, 240, 240)
COLOR_PURPLE = (200, 0, 255)
COLOR_GREEN = (50, 255, 50)

# Physics
GRAVITY = 0.8
PLAYER_SPEED = 6
JUMP_STRENGTH = -15
FRICTION = 0.85

# --- Helper Classes ---

class Particle:
    def __init__(self, pos, vel, color, size, lifetime):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.size = size
        self.lifetime = lifetime

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface, camera_x):
        if self.lifetime > 0 and self.size > 0:
            glow_size = int(self.size * 1.5)
            pos_on_screen = (int(self.pos[0] - camera_x), int(self.pos[1]))
            try:
                # Use a temporary surface for alpha blending to avoid errors with some pygame backends
                temp_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, glow_size, glow_size, glow_size, (*self.color, 50))
                surface.blit(temp_surf, (pos_on_screen[0] - glow_size, pos_on_screen[1] - glow_size))
                pygame.gfxdraw.filled_circle(surface, pos_on_screen[0], pos_on_screen[1], int(self.size), self.color)
            except (ValueError, TypeError): # Handle cases where size becomes invalid
                pass


class Projectile:
    def __init__(self, pos, vel, owner, damage):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.owner = owner  # 'player' or 'enemy'
        self.damage = damage
        self.lifetime = 120 # 4 seconds at 30 FPS

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1

    def get_rect(self):
        return pygame.Rect(self.pos[0] - 3, self.pos[1] - 3, 6, 6)

    def draw(self, surface, camera_x):
        color = COLOR_PLAYER_PROJ if self.owner == 'player' else COLOR_ENEMY_PROJ
        pos_on_screen = (int(self.pos[0] - camera_x), int(self.pos[1]))
        start_pos = (int(self.pos[0] - self.vel[0] * 2 - camera_x), int(self.pos[1] - self.vel[1] * 2))
        
        # Glow
        pygame.draw.line(surface, (*color, 100), start_pos, pos_on_screen, 6)
        # Core
        pygame.draw.line(surface, color, start_pos, pos_on_screen, 2)


class Platform:
    def __init__(self, rect, is_moving=False, move_range=(0,0), move_speed=0):
        self.rect = rect
        self.is_moving = is_moving
        self.move_range = move_range
        self.move_speed = move_speed
        self.initial_y = rect.y
        self.direction = 1

    def update(self):
        if self.is_moving:
            self.rect.y += self.move_speed * self.direction
            if self.rect.y <= self.initial_y - self.move_range[1]:
                self.direction = 1
            elif self.rect.y >= self.initial_y + self.move_range[1]:
                self.direction = -1

    def draw(self, surface, camera_x):
        rect_on_screen = self.rect.copy()
        rect_on_screen.x -= int(camera_x)
        
        # Glow effect
        glow_rect = rect_on_screen.inflate(8, 8)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*COLOR_PLATFORM_GLOW, 100), s.get_rect(), border_radius=4)
        surface.blit(s, glow_rect.topleft)

        # Main platform
        pygame.draw.rect(surface, COLOR_PLATFORM, rect_on_screen, border_radius=4)
        pygame.draw.rect(surface, COLOR_WHITE, rect_on_screen, 1, border_radius=4)


class Resource:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.rect = pygame.Rect(pos[0] - 8, pos[1] - 8, 16, 16)
        self.bob_angle = random.uniform(0, 2 * math.pi)

    def update(self):
        self.bob_angle += 0.1
        self.rect.y = self.pos[1] + math.sin(self.bob_angle) * 3

    def draw(self, surface, camera_x):
        rect_on_screen = self.rect.copy()
        rect_on_screen.x -= int(camera_x)
        
        # Glow
        glow_rect = rect_on_screen.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*COLOR_RESOURCE_GLOW, 150), s.get_rect(), border_radius=10)
        surface.blit(s, glow_rect.topleft)
        
        # Core
        pygame.draw.rect(surface, COLOR_RESOURCE, rect_on_screen, border_radius=8)


class Actor:
    def __init__(self, pos, size, health):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([0, 0], dtype=float)
        self.size = size
        self.health = health
        self.max_health = health
        self.on_ground = False
        self.facing_right = True

    def get_rect(self):
        # pos[1] is the bottom of the actor
        return pygame.Rect(self.pos[0] - self.size[0] / 2, self.pos[1] - self.size[1], self.size[0], self.size[1])

    def apply_physics(self, platforms):
        self.vel[1] += GRAVITY
        self.pos += self.vel
        self.vel[0] *= FRICTION
        self.on_ground = False

        rect = self.get_rect()
        for plat in platforms:
            if rect.colliderect(plat.rect):
                # Check vertical collision (landing on top)
                if self.vel[1] > 0 and rect.bottom - self.vel[1] <= plat.rect.top + 1: # +1 for float precision
                    self.pos[1] = plat.rect.top
                    self.vel[1] = 0
                    self.on_ground = True
                # Check horizontal collision
                elif rect.right > plat.rect.left and rect.left < plat.rect.right:
                    # from left
                    if self.vel[0] > 0 and rect.right - self.vel[0] <= plat.rect.left:
                        self.pos[0] = plat.rect.left - self.size[0] / 2
                        self.vel[0] = 0
                    # from right
                    elif self.vel[0] < 0 and rect.left - self.vel[0] >= plat.rect.right:
                        self.pos[0] = plat.rect.right + self.size[0] / 2
                        self.vel[0] = 0

        # World bounds
        self.pos[0] = np.clip(self.pos[0], self.size[0]/2, LEVEL_WIDTH - self.size[0]/2)
        if self.pos[1] > SCREEN_HEIGHT + 100: # Fell off world
            self.health = 0

class Player(Actor):
    def __init__(self, pos, size, health):
        super().__init__(pos, size, health)
        self.shoot_cooldown = 0
        self.clone_cooldown = 0
        self.jumps_left = 1
        self.augmentations = {
            "max_jumps": 1,
            "clone_duration": 150, # 5 seconds
            "rapid_fire": 1.0,
            "projectile_damage": 10,
        }
    
    def update_cooldowns(self):
        if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
        if self.clone_cooldown > 0: self.clone_cooldown -= 1

    def jump(self):
        if self.jumps_left > 0:
            self.vel[1] = JUMP_STRENGTH
            self.on_ground = False
            self.jumps_left -= 1
            return True
        return False

    def reset_jumps(self):
        if self.on_ground:
            self.jumps_left = self.augmentations["max_jumps"]

    def draw(self, surface, camera_x):
        rect = self.get_rect()
        rect.x -= int(camera_x)

        # Glow
        glow_rect = rect.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*COLOR_PLAYER_GLOW, 128), s.get_rect(), border_radius=5)
        surface.blit(s, glow_rect.topleft)

        # Body
        pygame.draw.rect(surface, COLOR_PLAYER, rect, border_radius=3)
        
        # "Eye" indicating direction
        eye_x = rect.centerx + (5 if self.facing_right else -5)
        pygame.draw.circle(surface, COLOR_WHITE, (eye_x, rect.centery - 5), 3)


class Clone(Actor):
    def __init__(self, pos, size, duration, facing_right):
        super().__init__(pos, size, 9999) # Clones share player health pool
        self.lifetime = duration
        self.facing_right = facing_right
        self.shoot_cooldown = 0
    
    def update(self):
        self.lifetime -= 1
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

    def draw(self, surface, camera_x):
        rect = self.get_rect()
        rect.x -= int(camera_x)

        alpha = int(max(0, min(255, self.lifetime * 3)))
        if alpha <= 0: return

        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        
        # Body
        pygame.draw.rect(s, (*COLOR_CLONE, alpha), s.get_rect(), border_radius=3)
        
        # Eye
        eye_x = s.get_rect().centerx + (5 if self.facing_right else -5)
        pygame.draw.circle(s, (*COLOR_WHITE, alpha), (eye_x, s.get_rect().centery - 5), 3)

        surface.blit(s, rect.topleft)

class Enemy(Actor):
    def __init__(self, pos, size, health, patrol_dist):
        super().__init__(pos, size, health)
        self.initial_pos_x = pos[0]
        self.patrol_dist = patrol_dist
        self.vel[0] = random.choice([-2, 2])
        self.facing_right = self.vel[0] > 0
        self.shoot_cooldown = random.randint(60, 120)

    def update(self, player_pos, projectile_speed):
        # Patrol logic
        if self.on_ground:
            if self.pos[0] > self.initial_pos_x + self.patrol_dist:
                self.vel[0] = -2
                self.facing_right = False
            elif self.pos[0] < self.initial_pos_x - self.patrol_dist:
                self.vel[0] = 2
                self.facing_right = True

        # Cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        
        # Shooting logic
        should_shoot = False
        dist_to_player = np.linalg.norm(self.pos - player_pos)
        if self.shoot_cooldown <= 0 and dist_to_player < 400:
            self.shoot_cooldown = random.randint(90, 150)
            should_shoot = True
        
        return should_shoot, projectile_speed

    def draw(self, surface, camera_x):
        rect = self.get_rect()
        rect.x -= int(camera_x)

        # Glow
        glow_rect = rect.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*COLOR_ENEMY_GLOW, 128), s.get_rect(), border_radius=5)
        surface.blit(s, glow_rect.topleft)

        # Body
        pygame.draw.rect(surface, COLOR_ENEMY, rect, border_radius=3)

        # Health bar
        if self.health < self.max_health:
            health_pct = self.health / self.max_health
            bar_width = int(rect.width * health_pct)
            pygame.draw.rect(surface, COLOR_GREEN, (rect.left, rect.top - 8, bar_width, 4))
            pygame.draw.rect(surface, COLOR_WHITE, (rect.left, rect.top - 8, rect.width, 4), 1)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": FPS}
    
    game_description = (
        "Cyberpunk side-scrolling platformer. Shoot enemies, create clones, and navigate treacherous platforms to reach the goal."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to shoot and shift to create a clone."
    )
    auto_advance = True

    # Persistent state across resets
    unlocked_augmentations = ["double_jump"]
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.render_mode = render_mode
        self.camera_x = 0

        self.prev_space_held = False
        self.prev_shift_held = False
    
    def _generate_level(self):
        self.platforms = []
        self.resources = []
        self.enemies = []
        
        # Start platform
        start_plat = Platform(pygame.Rect(50, SCREEN_HEIGHT - 50, 300, 50))
        self.platforms.append(start_plat)
        
        current_x = start_plat.rect.right
        current_y = start_plat.rect.top
        
        while current_x < LEVEL_WIDTH - SCREEN_WIDTH:
            gap = random.uniform(40, 120)
            y_change = random.uniform(-100, 100)
            plat_width = random.uniform(80, 250)
            
            new_x = current_x + gap
            new_y = np.clip(current_y + y_change, 100, SCREEN_HEIGHT - 50)
            
            is_moving = random.random() < 0.2
            move_range = (0, random.uniform(20, 60)) if is_moving else (0,0)
            move_speed = random.uniform(0.5, 1.5) if is_moving else 0
            
            new_plat = Platform(pygame.Rect(new_x, new_y, plat_width, 20), is_moving, move_range, move_speed)
            self.platforms.append(new_plat)

            # Spawn resources
            if random.random() < 0.5:
                res_pos = (new_plat.rect.centerx, new_plat.rect.top - 20)
                self.resources.append(Resource(res_pos))

            # Spawn enemies
            if random.random() < self.enemy_spawn_rate:
                enemy_pos = (new_plat.rect.centerx, new_plat.rect.top)
                self.enemies.append(Enemy(enemy_pos, (24, 30), 30, new_plat.rect.width / 2 - 20))
            
            current_x = new_plat.rect.right
            current_y = new_plat.rect.top
        
        # End platform
        self.goal_rect = pygame.Rect(LEVEL_WIDTH - 200, SCREEN_HEIGHT - 150, 100, 100)
        end_plat = Platform(self.goal_rect)
        self.platforms.append(end_plat)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.step_reward = 0

        # Difficulty scaling
        self.enemy_spawn_rate = 0.1
        self.platform_move_speed_mod = 1.0
        self.enemy_projectile_speed = 4.0

        # Place player on the starting platform
        start_plat_top = SCREEN_HEIGHT - 50
        self.player = Player((150, start_plat_top), (20, 40), 100)
        self.clones = []
        self.projectiles = []
        self.particles = []

        self._generate_level()
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.resources_for_aug = 0

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.step_reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- Action Handling ---
        movement = action[0]
        space_pressed = (action[1] == 1) and not self.prev_space_held
        shift_pressed = (action[2] == 1) and not self.prev_shift_held
        
        if movement == 1: # Jump
            if self.player.jump():
                self._create_particles(self.player.get_rect().midbottom, 5, COLOR_PLAYER, vel_y_range=(-3, -1))
        if movement == 3: # Left
            self.player.vel[0] -= 1.5
            self.player.facing_right = False
        if movement == 4: # Right
            self.player.vel[0] += 1.5
            self.player.facing_right = True

        if space_pressed and self.player.shoot_cooldown <= 0:
            self._shoot(self.player)

        if shift_pressed and self.player.clone_cooldown <= 0 and len(self.clones) < 3:
            self.player.clone_cooldown = 90 # 3 second cooldown
            clone_pos = (self.player.pos[0], self.player.pos[1])
            new_clone = Clone(clone_pos, self.player.size, int(self.player.augmentations["clone_duration"]), self.player.facing_right)
            self.clones.append(new_clone)
            self._create_particles(new_clone.pos, 20, COLOR_CLONE)

        self.prev_space_held = (action[1] == 1)
        self.prev_shift_held = (action[2] == 1)

        # --- Game Logic Update ---
        self.player.update_cooldowns()
        self.player.apply_physics(self.platforms)
        self.player.reset_jumps()

        for clone in self.clones:
            clone.update()
            if clone.shoot_cooldown <= 0:
                self._shoot(clone)
        self.clones = [c for c in self.clones if c.lifetime > 0]

        for enemy in self.enemies:
            enemy.apply_physics(self.platforms)
            should_shoot, proj_speed = enemy.update(self.player.pos, self.enemy_projectile_speed)
            if should_shoot:
                self._shoot(enemy, proj_speed)

        for proj in self.projectiles: proj.update()
        for particle in self.particles: particle.update()
        for res in self.resources: res.update()
        for plat in self.platforms: plat.update()

        # --- Collision Detection ---
        player_rect = self.player.get_rect()

        # Projectiles vs Actors
        new_projectiles = []
        for proj in self.projectiles:
            proj_hit = False
            if proj.owner == 'player':
                for enemy in self.enemies:
                    if enemy.get_rect().colliderect(proj.get_rect()):
                        enemy.health -= proj.damage
                        self._create_particles(proj.pos, 10, COLOR_ENEMY)
                        proj_hit = True
                        break
            elif proj.owner == 'enemy':
                if player_rect.colliderect(proj.get_rect()):
                    self.player.health -= proj.damage
                    self.step_reward -= 0.1
                    self._create_particles(proj.pos, 10, COLOR_PLAYER)
                    proj_hit = True
            
            if not proj_hit and proj.lifetime > 0:
                new_projectiles.append(proj)
        self.projectiles = new_projectiles
        
        # Player vs Resources
        collected_resources = []
        for res in self.resources:
            if player_rect.colliderect(res.rect):
                collected_resources.append(res)
                self.score += 1
                self.step_reward += 0.1
                self.resources_for_aug += 1
                self._create_particles(res.rect.center, 15, COLOR_RESOURCE)
        self.resources = [r for r in self.resources if r not in collected_resources]
        
        # Cleanup dead enemies
        dead_enemies = [e for e in self.enemies if e.health <= 0]
        if dead_enemies:
            self.score += 10 * len(dead_enemies)
            self.step_reward += 1 * len(dead_enemies)
            for e in dead_enemies: self._create_particles(e.pos, 30, COLOR_ENEMY)
        self.enemies = [e for e in self.enemies if e.health > 0]
        
        # Cleanup particles
        self.particles = [p for p in self.particles if p.lifetime > 0]

        # --- Augmentations ---
        if self.resources_for_aug >= 5:
            self.resources_for_aug -= 5
            self._apply_random_augmentation()

        # --- Difficulty Scaling ---
        if self.steps > 0 and self.steps % 500 == 0:
            self.enemy_spawn_rate = min(0.5, self.enemy_spawn_rate + 0.05)
            self.platform_move_speed_mod = min(2.0, self.platform_move_speed_mod + 0.02)
        if self.steps > 0 and self.steps % 250 == 0:
            self.enemy_projectile_speed = min(8.0, self.enemy_projectile_speed + 0.01)

        # --- Termination Check ---
        terminated = False
        truncated = False
        reward = self.step_reward

        if self.player.health <= 0:
            terminated = True
            reward = -100
            self.game_over = True
        
        if player_rect.colliderect(self.goal_rect):
            terminated = True
            reward = 100
            self.step_reward += 5 # For unlocking recipe
            self.game_over = True
            self.score += 1000
            # Unlock a new augmentation for future runs
            if "rapid_fire_2" not in self.unlocked_augmentations:
                self.unlocked_augmentations.append("rapid_fire_2")
            elif "clone_shoot" not in self.unlocked_augmentations:
                self.unlocked_augmentations.append("clone_shoot")

        if self.steps >= MAX_STEPS:
            truncated = True
            self.game_over = True
        
        if self.render_mode == "human":
            self.render()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _shoot(self, actor):
        if isinstance(actor, Player):
            actor.shoot_cooldown = 30 / self.player.augmentations["rapid_fire"]
            damage = self.player.augmentations["projectile_damage"]
            owner = 'player'
        elif isinstance(actor, Clone):
            actor.shoot_cooldown = 45 # Clones shoot slower
            damage = self.player.augmentations["projectile_damage"]
            owner = 'player'
        else: # Enemy
            damage = 10
            owner = 'enemy'
        
        vel_x = 10 if actor.facing_right else -10
        proj_pos = (actor.pos[0], actor.pos[1] - actor.size[1] / 2)
        
        new_proj = Projectile(proj_pos, (vel_x, 0), owner, damage)
        self.projectiles.append(new_proj)
        self._create_particles(proj_pos, 5, COLOR_PLAYER_PROJ if owner == 'player' else COLOR_ENEMY_PROJ, vel_x_mult=0.2 if actor.facing_right else -0.2)
    
    def _apply_random_augmentation(self):
        self._create_particles(self.player.pos, 50, COLOR_PURPLE)
        if not self.unlocked_augmentations: return
        choice = random.choice(self.unlocked_augmentations)
        if choice == "double_jump":
            self.player.augmentations["max_jumps"] = min(3, self.player.augmentations["max_jumps"] + 1)
        elif choice == "rapid_fire_2":
            self.player.augmentations["rapid_fire"] *= 1.25
        elif choice == "clone_shoot": # This is a conceptual one, let's make it increase duration
            self.player.augmentations["clone_duration"] *= 1.3

    def _create_particles(self, pos, count, color, vel_y_range=(-5, 5), vel_x_mult=1):
        for _ in range(count):
            vel = [random.uniform(-5, 5) * vel_x_mult, random.uniform(vel_y_range[0], vel_y_range[1])]
            size = random.uniform(2, 5)
            lifetime = random.randint(10, 25)
            self.particles.append(Particle(pos, vel, color, size, lifetime))

    def _get_observation(self):
        self._render_to_surface()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        elif self.render_mode == "human":
            # Human rendering is handled in the main loop after _render_to_surface
            # and pygame.display.flip()
            if not getattr(self, 'display', None):
                pygame.display.set_caption("Cyberpunk Platformer")
                self.display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            
            self._render_to_surface()
            self.display.blit(self.screen, (0,0))
            pygame.display.flip()
            self.clock.tick(FPS)
            return

    def _render_to_surface(self):
        self.screen.fill(COLOR_BG)
        
        # Update camera
        self.camera_x = self.player.pos[0] - SCREEN_WIDTH / 2
        self.camera_x = np.clip(self.camera_x, 0, LEVEL_WIDTH - SCREEN_WIDTH)

        # Background grid (parallax)
        for i in range(0, LEVEL_WIDTH, 50):
            x = i - int(self.camera_x * 0.8)
            if -50 < x < SCREEN_WIDTH + 50:
                pygame.draw.line(self.screen, COLOR_GRID_MINOR, (x, 0), (x, SCREEN_HEIGHT))
        for i in range(0, SCREEN_HEIGHT, 50):
            pygame.draw.line(self.screen, COLOR_GRID_MINOR, (0, i), (SCREEN_WIDTH, i))
        for i in range(0, LEVEL_WIDTH, 200):
            x = i - int(self.camera_x * 0.8)
            if -50 < x < SCREEN_WIDTH + 50:
                pygame.draw.line(self.screen, COLOR_GRID_MAJOR, (x, 0), (x, SCREEN_HEIGHT))

        # Game elements
        for plat in self.platforms: plat.draw(self.screen, self.camera_x)
        
        # Goal portal
        goal_on_screen = self.goal_rect.copy()
        goal_on_screen.x -= self.camera_x
        angle = (self.steps % 180) * 2 * (math.pi / 180)
        for i in range(8):
            a = angle + i * (math.pi/4)
            x1 = goal_on_screen.centerx + math.cos(a) * 40
            y1 = goal_on_screen.centery + math.sin(a) * 40
            x2 = goal_on_screen.centerx + math.cos(a) * 60
            y2 = goal_on_screen.centery + math.sin(a) * 60
            pygame.draw.line(self.screen, COLOR_PURPLE, (x1, y1), (x2, y2), 4)

        for res in self.resources: res.draw(self.screen, self.camera_x)
        for enemy in self.enemies: enemy.draw(self.screen, self.camera_x)
        for clone in self.clones: clone.draw(self.screen, self.camera_x)
        self.player.draw(self.screen, self.camera_x)
        for proj in self.projectiles: proj.draw(self.screen, self.camera_x)
        for particle in self.particles: particle.draw(self.screen, self.camera_x)
        
        self._render_ui()

    def _render_ui(self):
        # Health Bar
        health_pct = self.player.health / self.player.max_health
        bar_width = int(200 * health_pct)
        pygame.draw.rect(self.screen, COLOR_ENEMY, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, COLOR_PLAYER, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, COLOR_WHITE, (10, 10, 200, 20), 2)
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, COLOR_WHITE)
        self.screen.blit(score_text, (SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{MAX_STEPS}", True, COLOR_WHITE)
        self.screen.blit(steps_text, (SCREEN_WIDTH - steps_text.get_width() - 10, 30))

        # Augmentation progress
        pygame.draw.rect(self.screen, COLOR_WHITE, (10, 40, 104, 12), 1)
        aug_prog_width = int(100 * (self.resources_for_aug / 5))
        pygame.draw.rect(self.screen, COLOR_PURPLE, (12, 42, aug_prog_width, 8))

        if self.game_over:
            msg = "LEVEL COMPLETE" if self.player.health > 0 and not self.steps >= MAX_STEPS else "GAME OVER"
            color = COLOR_GREEN if "COMPLETE" in msg else COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player.health,
            "level_progress": self.player.pos[0] / LEVEL_WIDTH
        }
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    
    terminated = False
    truncated = False
    obs, info = env.reset()
    
    while not terminated and not truncated:
        movement = 0
        space = 0
        shift = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                terminated = False
                truncated = False

        if terminated: continue

        # Key polling for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_UP]: movement = 1
        # if keys[pygame.K_DOWN]: movement = 2 # Down is not used
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

    print("Game Over!")
    env.close()