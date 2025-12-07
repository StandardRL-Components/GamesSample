
# Generated: 2025-08-27T19:39:13.742499
# Source Brief: brief_02212.md
# Brief Index: 2212

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Press space to attack. Defeat the boss at the end of the dungeon."
    )

    game_description = (
        "An isometric action RPG. Battle through a dungeon, collect gold, and defeat the final boss."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_WALL = (50, 60, 70)
    COLOR_FLOOR = (70, 80, 90)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_SHADOW = (30, 90, 150)
    COLOR_ENEMY_GOBLIN = (220, 50, 50)
    COLOR_ENEMY_SHADOW = (130, 30, 30)
    COLOR_BOSS = (150, 50, 250)
    COLOR_BOSS_SHADOW = (90, 30, 150)
    COLOR_GOLD = (255, 215, 0)
    COLOR_WHITE = (240, 240, 240)
    COLOR_UI_BG = (40, 50, 60, 200)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_HEALTH_BAR_BG = (120, 50, 50)

    # Game World
    TILE_SIZE = 40
    WORLD_WIDTH = 50
    WORLD_HEIGHT = 15

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
        
        self.font_ui = pygame.font.Font(None, 24)
        self.font_damage = pygame.font.Font(None, 20)
        self.font_gold = pygame.font.Font(None, 22)

        self.render_mode = render_mode
        
        self.reset()
        
        # This is a dummy call to validate_implementation during initialization.
        # In a real scenario, you might not ship this, but it's good for development.
        try:
            self.validate_implementation()
        except AssertionError as e:
            print(f"Implementation validation failed: {e}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        
        self._generate_dungeon()
        
        self.player = Player(pos=pygame.Vector2(self.TILE_SIZE * 3, self.TILE_SIZE * self.WORLD_HEIGHT / 2))
        self.boss = Boss(pos=pygame.Vector2(self.TILE_SIZE * (self.WORLD_WIDTH - 6), self.TILE_SIZE * self.WORLD_HEIGHT / 2))
        
        self.enemies = []
        self.projectiles = []
        self.golds = []
        self.particles = []
        self.floating_texts = []
        
        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 90 # frames
        self.gold_spawn_timer = 150 # frames
        
        self.prev_space_held = False
        self.camera_offset = pygame.Vector2(0, 0)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self.reward_this_step = 0

        self._handle_input(movement, space_held)
        
        self._update_player()
        self._update_enemies()
        self._update_boss()
        self._update_projectiles()
        self._update_particles()
        self._update_floating_texts()
        
        self._handle_collisions()
        self._handle_spawns()
        
        self._cleanup()

        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.player.health <= 0:
                self.reward_this_step -= 100
            elif self.boss.health <= 0:
                self.reward_this_step += 100

        self.score += self.reward_this_step
        
        if self.auto_advance:
            self.clock.tick(30)
            
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._update_camera()
        
        self._render_dungeon()
        self._render_entities()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": int(self.score), "steps": self.steps, "health": self.player.health, "gold": self.player.gold}

    def _generate_dungeon(self):
        self.world = np.ones((self.WORLD_WIDTH, self.WORLD_HEIGHT), dtype=int) # 1 = wall
        # Carve out a long corridor
        self.world[2:self.WORLD_WIDTH-2, 3:self.WORLD_HEIGHT-3] = 0 # 0 = floor
        # Boss room
        self.world[self.WORLD_WIDTH-12:self.WORLD_WIDTH-2, 2:self.WORLD_HEIGHT-2] = 0

    def _handle_input(self, movement, space_held):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right
        
        if move_vec.length_squared() > 0:
            self.player.move(move_vec, self.world, self.TILE_SIZE)
        
        if space_held and not self.prev_space_held:
            self.player.attack(self)
        self.prev_space_held = space_held

    def _update_player(self):
        self.player.update()

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy.update(self.player, self.world, self.TILE_SIZE, self)

    def _update_boss(self):
        self.boss.update(self.player, self)

    def _update_projectiles(self):
        for p in self.projectiles:
            p.update()
    
    def _update_particles(self):
        for p in self.particles:
            p.update()

    def _update_floating_texts(self):
        for ft in self.floating_texts:
            ft.update()

    def _handle_collisions(self):
        # Player attack vs enemies/boss
        if self.player.is_attacking():
            attack_rect = self.player.get_attack_rect()
            for enemy in self.enemies:
                if not enemy.hit_this_frame and attack_rect.colliderect(enemy.get_rect()):
                    damage = self.player.attack_damage + self.np_random.integers(-2, 3)
                    enemy.take_damage(damage, self)
                    self.reward_this_step += 0.1
                    # sfx: sword_hit
            # vs Boss
            if not self.boss.hit_this_frame and attack_rect.colliderect(self.boss.get_rect()):
                damage = self.player.attack_damage + self.np_random.integers(-2, 3)
                self.boss.take_damage(damage, self)
                self.reward_this_step += 0.1
                # sfx: sword_hit_boss
        
        # Projectiles vs Player
        for p in self.projectiles:
            if not p.hit and self.player.get_rect().colliderect(p.get_rect()):
                p.hit = True
                damage = p.damage + self.np_random.integers(-1, 2)
                self.player.take_damage(damage, self)
                self.reward_this_step -= 0.1 * damage
                # sfx: player_hit
        
        # Enemies vs Player
        for enemy in self.enemies:
            if enemy.is_attacking() and enemy.get_attack_rect().colliderect(self.player.get_rect()):
                damage = enemy.attack_damage + self.np_random.integers(-1, 2)
                self.player.take_damage(damage, self)
                self.reward_this_step -= 0.1 * damage
                # sfx: player_hit

        # Player vs Gold
        player_rect = self.player.get_rect()
        for gold in self.golds:
            if player_rect.colliderect(gold.get_rect()):
                gold.collected = True
                self.player.gold += gold.value
                self.reward_this_step += 10
                self.floating_texts.append(FloatingText(gold.pos, f"+{gold.value}G", self.COLOR_GOLD, self.font_gold))
                # sfx: gold_pickup

    def _handle_spawns(self):
        # Enemy spawn
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            spawn_x = self.np_random.integers(5, self.WORLD_WIDTH - 15) * self.TILE_SIZE
            spawn_y = self.np_random.integers(4, self.WORLD_HEIGHT - 4) * self.TILE_SIZE
            pos = pygame.Vector2(spawn_x, spawn_y)
            
            # Ensure spawn is on floor and not too close to player
            if self.world[int(pos.x / self.TILE_SIZE)][int(pos.y / self.TILE_SIZE)] == 0 and pos.distance_to(self.player.pos) > 200:
                enemy_type = self.np_random.choice(['goblin', 'mage'])
                health_multiplier = (1 + 0.05 * math.floor(self.steps / 50))
                self.enemies.append(Enemy(pos, enemy_type, health_multiplier))
                
            spawn_rate_reduction = 0.01 * math.floor(self.steps / 100)
            self.enemy_spawn_timer = max(30, self.enemy_spawn_rate - spawn_rate_reduction * 30)

        # Gold spawn
        self.gold_spawn_timer -= 1
        if self.gold_spawn_timer <= 0:
            spawn_x = self.np_random.integers(5, self.WORLD_WIDTH - 15) * self.TILE_SIZE
            spawn_y = self.np_random.integers(4, self.WORLD_HEIGHT - 4) * self.TILE_SIZE
            pos = pygame.Vector2(spawn_x, spawn_y)
            if self.world[int(pos.x / self.TILE_SIZE)][int(pos.y / self.TILE_SIZE)] == 0:
                self.golds.append(Gold(pos))
            self.gold_spawn_timer = self.np_random.integers(150, 300)

    def _cleanup(self):
        self.enemies = [e for e in self.enemies if e.health > 0]
        self.projectiles = [p for p in self.projectiles if not p.hit and p.lifespan > 0]
        self.golds = [g for g in self.golds if not g.collected]
        self.particles = [p for p in self.particles if p.lifespan > 0]
        self.floating_texts = [ft for ft in self.floating_texts if ft.lifespan > 0]
        
        for enemy in self.enemies: enemy.hit_this_frame = False
        self.boss.hit_this_frame = False
        
    def _check_termination(self):
        return self.player.health <= 0 or self.boss.health <= 0 or self.steps >= self.MAX_STEPS

    def _update_camera(self):
        target_cam_x = self.player.pos.x - self.SCREEN_WIDTH / 2
        target_cam_y = self.player.pos.y - self.SCREEN_HEIGHT / 2
        self.camera_offset.x += (target_cam_x - self.camera_offset.x) * 0.1
        self.camera_offset.y += (target_cam_y - self.camera_offset.y) * 0.1

    def _iso_to_screen(self, x, y):
        screen_x = (x - y) + self.SCREEN_WIDTH / 2 - self.camera_offset.x + self.camera_offset.y
        screen_y = (x + y) / 2 + self.SCREEN_HEIGHT / 4 - (self.camera_offset.x + self.camera_offset.y) / 2
        return int(screen_x), int(screen_y)

    def _render_dungeon(self):
        tile_w_iso, tile_h_iso = self._iso_to_screen(self.TILE_SIZE, 0)[0] - self._iso_to_screen(0,0)[0], self._iso_to_screen(0, self.TILE_SIZE)[1] - self._iso_to_screen(0,0)[1]
        
        start_x = max(0, int(self.camera_offset.x / self.TILE_SIZE) - 5)
        end_x = min(self.WORLD_WIDTH, int((self.camera_offset.x + self.SCREEN_WIDTH*1.5) / self.TILE_SIZE))
        start_y = max(0, int(self.camera_offset.y / self.TILE_SIZE) - 5)
        end_y = min(self.WORLD_HEIGHT, int((self.camera_offset.y + self.SCREEN_HEIGHT*2.5) / self.TILE_SIZE))

        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                screen_pos = self._iso_to_screen(x * self.TILE_SIZE, y * self.TILE_SIZE)
                tile_type = self.world[x][y]
                
                points = [
                    screen_pos,
                    (screen_pos[0] + tile_w_iso, screen_pos[1] + tile_h_iso),
                    (screen_pos[0], screen_pos[1] + tile_h_iso * 2),
                    (screen_pos[0] - tile_w_iso, screen_pos[1] + tile_h_iso),
                ]
                
                if tile_type == 0: # Floor
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_FLOOR)
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_FLOOR)
                elif tile_type == 1: # Wall
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_WALL)
                    # Simple 3D effect
                    top_points = [
                        (points[0][0], points[0][1] - tile_h_iso * 2),
                        (points[1][0], points[1][1] - tile_h_iso * 2),
                        points[1],
                        points[0]
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, top_points, self.COLOR_WALL)
                    pygame.gfxdraw.filled_polygon(self.screen, [points[3], (points[3][0], points[3][1]-tile_h_iso*2), (top_points[0][0], top_points[0][1]), points[0]], (40,50,60))


    def _render_entities(self):
        entities = sorted(self.enemies + self.golds + [self.player, self.boss] + self.projectiles, key=lambda e: e.pos.y)
        for entity in entities:
            entity.render(self.screen, self._iso_to_screen)

    def _render_effects(self):
        for p in self.particles:
            p.render(self.screen, self._iso_to_screen)
        for ft in self.floating_texts:
            ft.render(self.screen, self._iso_to_screen)

    def _render_ui(self):
        # Player Health
        health_ratio = self.player.health / self.player.max_health
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, bar_width + 10, 30), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (15, 15, bar_width, 20), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (15, 15, max(0, bar_width * health_ratio), 20), border_radius=3)
        health_text = self.font_ui.render(f"{int(self.player.health)}/{int(self.player.max_health)}", True, self.COLOR_WHITE)
        self.screen.blit(health_text, (20, 17))
        
        # Boss Health
        if self.player.pos.distance_to(self.boss.pos) < 500:
            boss_health_ratio = self.boss.health / self.boss.max_health
            boss_bar_width = self.SCREEN_WIDTH - 40
            pygame.draw.rect(self.screen, self.COLOR_UI_BG, (20, self.SCREEN_HEIGHT - 40, boss_bar_width, 25), border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (25, self.SCREEN_HEIGHT - 35, boss_bar_width - 10, 15), border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_BOSS, (25, self.SCREEN_HEIGHT - 35, max(0, (boss_bar_width - 10) * boss_health_ratio), 15), border_radius=3)
            boss_text = self.font_ui.render("FINAL BOSS", True, self.COLOR_WHITE)
            self.screen.blit(boss_text, (self.SCREEN_WIDTH/2 - boss_text.get_width()/2, self.SCREEN_HEIGHT - 38))

        # Gold
        gold_text = self.font_ui.render(f"Gold: {self.player.gold}", True, self.COLOR_GOLD)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.SCREEN_WIDTH - gold_text.get_width() - 20, 10, gold_text.get_width() + 10, 30), border_radius=5)
        self.screen.blit(gold_text, (self.SCREEN_WIDTH - gold_text.get_width() - 15, 17))

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool) and trunc is False and isinstance(info, dict)
        print("✓ Implementation validated successfully")

# --- Helper Classes ---

class Player:
    def __init__(self, pos):
        self.pos = pos
        self.target_pos = pos.copy()
        self.vel = pygame.Vector2(0, 0)
        self.speed = 4.5
        self.size = 12
        self.health = 100
        self.max_health = 100
        self.gold = 0
        self.attack_cooldown = 0
        self.attack_duration = 0
        self.attack_damage = 10
        self.last_move_dir = pygame.Vector2(1, 0)
        self.invulnerable_timer = 0
    
    def move(self, direction, world, tile_size):
        if self.attack_duration > 0: return
        self.last_move_dir = direction.copy()
        new_pos = self.pos + direction * self.speed
        
        world_x, world_y = int(new_pos.x / tile_size), int(new_pos.y / tile_size)
        if 0 <= world_x < GameEnv.WORLD_WIDTH and 0 <= world_y < GameEnv.WORLD_HEIGHT and world[world_x][world_y] == 0:
            self.pos = new_pos

    def attack(self, env):
        if self.attack_cooldown <= 0:
            self.attack_cooldown = 20 # frames
            self.attack_duration = 8 # frames
            # sfx: sword_swing
    
    def take_damage(self, amount, env):
        if self.invulnerable_timer <= 0:
            assert amount >= 0
            self.health = max(0, self.health - amount)
            self.invulnerable_timer = 30 # frames of invulnerability
            env.floating_texts.append(FloatingText(self.pos, f"-{int(amount)}", GameEnv.COLOR_ENEMY_GOBLIN, env.font_damage))
            for _ in range(10):
                env.particles.append(Particle(self.pos, lifespan=15, color=GameEnv.COLOR_PLAYER, size_range=(1,3)))

    def update(self):
        if self.attack_cooldown > 0: self.attack_cooldown -= 1
        if self.attack_duration > 0: self.attack_duration -= 1
        if self.invulnerable_timer > 0: self.invulnerable_timer -= 1
        
    def is_attacking(self):
        return self.attack_duration > 0
    
    def get_attack_rect(self):
        attack_center = self.pos + self.last_move_dir * (self.size + 15)
        return pygame.Rect(attack_center.x - 15, attack_center.y - 15, 30, 30)
        
    def get_rect(self):
        return pygame.Rect(self.pos.x - self.size, self.pos.y - self.size, self.size * 2, self.size * 2)
        
    def render(self, screen, iso_to_screen):
        screen_pos = iso_to_screen(self.pos.x, self.pos.y)
        shadow_pos = (screen_pos[0], screen_pos[1] + self.size)
        
        # Shadow
        pygame.gfxdraw.filled_ellipse(screen, shadow_pos[0], shadow_pos[1], self.size, self.size // 2, GameEnv.COLOR_PLAYER_SHADOW)
        
        # Body
        body_color = GameEnv.COLOR_PLAYER
        if self.invulnerable_timer > 0 and (self.invulnerable_timer // 3) % 2 == 0:
            body_color = GameEnv.COLOR_WHITE
        pygame.gfxdraw.filled_circle(screen, screen_pos[0], screen_pos[1], self.size, body_color)
        pygame.gfxdraw.aacircle(screen, screen_pos[0], screen_pos[1], self.size, body_color)
        
        # Attack animation
        if self.is_attacking():
            progress = 1 - (self.attack_duration / 8)
            angle_offset = math.pi / 2
            start_angle = self.last_move_dir.angle_to(pygame.Vector2(1,0)) * (math.pi/180) - angle_offset
            
            for i in range(4):
                angle = start_angle + (i - 1.5) * 0.5
                arc_pos = self.pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * (self.size + 10 + progress * 15)
                arc_screen_pos = iso_to_screen(arc_pos.x, arc_pos.y)
                size = int(5 * (1-progress))
                if size > 0:
                    pygame.gfxdraw.filled_circle(screen, arc_screen_pos[0], arc_screen_pos[1], size, GameEnv.COLOR_WHITE)

class Enemy:
    def __init__(self, pos, type, health_multiplier):
        self.pos = pos
        self.type = type
        self.size = 10
        self.speed = 2.0 if type == 'goblin' else 1.0
        self.health = (50 if type == 'goblin' else 30) * health_multiplier
        self.max_health = self.health
        self.attack_damage = 10 if type == 'goblin' else 15
        self.attack_cooldown = 0
        self.attack_range = 30 if type == 'goblin' else 250
        self.aggro_range = 300
        self.hit_this_frame = False
        self.attack_charge = 0
    
    def update(self, player, world, tile_size, env):
        if self.attack_cooldown > 0: self.attack_cooldown -= 1
        
        dist_to_player = self.pos.distance_to(player.pos)
        
        if dist_to_player < self.aggro_range:
            direction = (player.pos - self.pos).normalize() if dist_to_player > 0 else pygame.Vector2(0,0)
            
            if self.type == 'goblin':
                if dist_to_player > self.attack_range:
                    new_pos = self.pos + direction * self.speed
                    world_x, world_y = int(new_pos.x / tile_size), int(new_pos.y / tile_size)
                    if 0 <= world_x < GameEnv.WORLD_WIDTH and 0 <= world_y < GameEnv.WORLD_HEIGHT and world[world_x][world_y] == 0:
                        self.pos = new_pos
                elif self.attack_cooldown <= 0:
                    self.attack_cooldown = 60
            
            elif self.type == 'mage':
                if dist_to_player < self.attack_range - 50: # back away
                    new_pos = self.pos - direction * self.speed
                    world_x, world_y = int(new_pos.x / tile_size), int(new_pos.y / tile_size)
                    if 0 <= world_x < GameEnv.WORLD_WIDTH and 0 <= world_y < GameEnv.WORLD_HEIGHT and world[world_x][world_y] == 0:
                        self.pos = new_pos
                elif self.attack_cooldown <= 0:
                    self.attack_cooldown = 120
                    self.attack_charge = 30
                    # sfx: mage_charge
                
        if self.attack_charge > 0:
            self.attack_charge -= 1
            if self.attack_charge == 0:
                # fire projectile
                proj_dir = (player.pos - self.pos).normalize()
                env.projectiles.append(Projectile(self.pos.copy(), proj_dir, GameEnv.COLOR_ENEMY_GOBLIN, self.attack_damage, owner='enemy'))
                # sfx: mage_fire

    def take_damage(self, amount, env):
        assert amount >= 0
        self.health = max(0, self.health - amount)
        self.hit_this_frame = True
        env.floating_texts.append(FloatingText(self.pos, f"-{int(amount)}", GameEnv.COLOR_WHITE, env.font_damage))
        if self.health <= 0:
            env.reward_this_step += 1
            for _ in range(20):
                env.particles.append(Particle(self.pos, lifespan=20, color=GameEnv.COLOR_ENEMY_GOBLIN, size_range=(2,5), speed_range=(1,4)))
            if env.np_random.random() < 0.3: # 30% chance to drop gold
                env.golds.append(Gold(self.pos.copy()))
            # sfx: enemy_die
        else:
            for _ in range(5):
                env.particles.append(Particle(self.pos, lifespan=10, color=GameEnv.COLOR_WHITE, size_range=(1,2)))
            # sfx: enemy_hit

    def is_attacking(self):
        return self.type == 'goblin' and self.attack_cooldown > 45 # attack frame
    
    def get_attack_rect(self):
        return self.get_rect()

    def get_rect(self):
        return pygame.Rect(self.pos.x - self.size, self.pos.y - self.size, self.size * 2, self.size * 2)
    
    def render(self, screen, iso_to_screen):
        screen_pos = iso_to_screen(self.pos.x, self.pos.y)
        shadow_pos = (screen_pos[0], screen_pos[1] + self.size)
        pygame.gfxdraw.filled_ellipse(screen, shadow_pos[0], shadow_pos[1], self.size, self.size // 2, GameEnv.COLOR_ENEMY_SHADOW)
        
        color = GameEnv.COLOR_ENEMY_GOBLIN if self.type == 'goblin' else (200, 50, 200)
        pygame.gfxdraw.filled_circle(screen, screen_pos[0], screen_pos[1], self.size, color)
        
        if self.attack_charge > 0:
            charge_size = int(self.size * 1.5 * (1 - self.attack_charge / 30))
            pygame.gfxdraw.aacircle(screen, screen_pos[0], screen_pos[1], charge_size, color)

class Boss(Enemy):
    def __init__(self, pos):
        super().__init__(pos, 'boss', 1.0)
        self.size = 25
        self.health = 1000
        self.max_health = 1000
        self.attack_damage = 20
        self.attack_pattern_timer = 200
        self.current_pattern = 0

    def update(self, player, env):
        self.attack_pattern_timer -= 1
        if self.attack_pattern_timer <= 0:
            self.current_pattern = (self.current_pattern + 1) % 2
            if self.current_pattern == 0: # Projectile Volley
                self.attack_pattern_timer = 150
                for i in range(8):
                    angle = i * (math.pi / 4)
                    proj_dir = pygame.Vector2(math.cos(angle), math.sin(angle))
                    env.projectiles.append(Projectile(self.pos.copy(), proj_dir, GameEnv.COLOR_BOSS, self.attack_damage, owner='boss'))
                # sfx: boss_volley
            elif self.current_pattern == 1: # Summon Minions
                self.attack_pattern_timer = 300
                for i in range(2):
                    offset = pygame.Vector2(env.np_random.uniform(-80, 80), env.np_random.uniform(-80, 80))
                    env.enemies.append(Enemy(self.pos + offset, 'goblin', 0.5))
                # sfx: boss_summon

    def take_damage(self, amount, env):
        assert amount >= 0
        self.health = max(0, self.health - amount)
        self.hit_this_frame = True
        env.floating_texts.append(FloatingText(self.pos, f"-{int(amount)}", GameEnv.COLOR_WHITE, env.font_damage))
        if self.health <= 0:
            env.reward_this_step += 100 # This is a placeholder; final reward is in step()
            for _ in range(100):
                env.particles.append(Particle(self.pos, lifespan=60, color=GameEnv.COLOR_BOSS, size_range=(3,8), speed_range=(2,6)))
            # sfx: boss_die
        else:
             # sfx: boss_hit
             for _ in range(8):
                env.particles.append(Particle(self.pos, lifespan=15, color=GameEnv.COLOR_WHITE, size_range=(2,4)))
    
    def render(self, screen, iso_to_screen):
        screen_pos = iso_to_screen(self.pos.x, self.pos.y)
        shadow_pos = (screen_pos[0], screen_pos[1] + self.size)
        pygame.gfxdraw.filled_ellipse(screen, shadow_pos[0], shadow_pos[1], self.size, self.size // 2, GameEnv.COLOR_BOSS_SHADOW)
        
        # Pulsing effect
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.002))
        current_size = int(self.size + pulse * 4)
        pygame.gfxdraw.filled_circle(screen, screen_pos[0], screen_pos[1], current_size, GameEnv.COLOR_BOSS)
        pygame.gfxdraw.aacircle(screen, screen_pos[0], screen_pos[1], current_size, GameEnv.COLOR_BOSS)

class Projectile:
    def __init__(self, pos, direction, color, damage, owner):
        self.pos = pos
        self.vel = direction * (5 if owner == 'enemy' else 8)
        self.color = color
        self.damage = damage
        self.size = 5
        self.lifespan = 120
        self.hit = False
        self.owner = owner
    
    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
    
    def get_rect(self):
        return pygame.Rect(self.pos.x - self.size, self.pos.y - self.size, self.size*2, self.size*2)

    def render(self, screen, iso_to_screen):
        screen_pos = iso_to_screen(self.pos.x, self.pos.y)
        pygame.gfxdraw.filled_circle(screen, screen_pos[0], screen_pos[1], self.size, self.color)

class Gold:
    def __init__(self, pos):
        self.pos = pos
        self.value = 10
        self.size = 8
        self.collected = False
    
    def get_rect(self):
        return pygame.Rect(self.pos.x - self.size, self.pos.y - self.size, self.size*2, self.size*2)
        
    def render(self, screen, iso_to_screen):
        screen_pos = iso_to_screen(self.pos.x, self.pos.y)
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.005 + self.pos.x))
        current_size = int(self.size * 0.8 + pulse * self.size * 0.4)
        pygame.gfxdraw.filled_circle(screen, screen_pos[0], screen_pos[1], current_size, GameEnv.COLOR_GOLD)

class Particle:
    def __init__(self, pos, lifespan, color, size_range, speed_range=(1,3)):
        self.pos = pos.copy()
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*speed_range)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.size = random.randint(*size_range)
        
    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.vel *= 0.95
        
    def render(self, screen, iso_to_screen):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color = self.color + (alpha,)
            size = int(self.size * (self.lifespan / self.max_lifespan))
            if size > 0:
                screen_pos = iso_to_screen(self.pos.x, self.pos.y)
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                screen.blit(temp_surf, (screen_pos[0] - size, screen_pos[1] - size))

class FloatingText:
    def __init__(self, pos, text, color, font):
        self.pos = pos.copy()
        self.text = text
        self.color = color
        self.font = font
        self.lifespan = 45
        self.max_lifespan = 45
        
    def update(self):
        self.pos.y -= 0.5
        self.lifespan -= 1
        
    def render(self, screen, iso_to_screen):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            text_surf = self.font.render(self.text, True, self.color)
            text_surf.set_alpha(alpha)
            screen_pos = iso_to_screen(self.pos.x, self.pos.y)
            screen.blit(text_surf, (screen_pos[0] - text_surf.get_width()//2, screen_pos[1] - 15))