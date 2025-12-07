import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class Player:
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)
        self.max_health = 100
        self.health = self.max_health
        self.xp = 0
        self.facing = np.array([1.0, 1.0])
        self.attack_cooldown = 0
        self.iframes = 0  # Invincibility frames

class Enemy:
    def __init__(self, x, y, is_boss=False):
        self.x, self.y = float(x), float(y)
        self.is_boss = is_boss
        self.max_health = 100 if is_boss else 20
        self.health = self.max_health
        self.damage = 25 if is_boss else 10
        self.xp_reward = 50 if is_boss else 10
        self.size = 1.2 if is_boss else 0.8
        self.ai_state = 0
        self.ai_timer = 0
        self.target_pos = None

class Potion:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.heal_amount = 25

class Attack:
    def __init__(self, x, y, facing):
        self.x, self.y = x, y
        self.facing = facing
        self.lifetime = 6  # frames
        self.hit_enemies = set()

class Particle:
    def __init__(self, x, y, color, lifetime, velocity):
        self.x, self.y = x, y
        self.color = color
        self.lifetime = lifetime
        self.velocity = velocity

class FloatingText:
    def __init__(self, x, y, text, color, lifetime=30):
        self.x, self.y = x, y
        self.text = text
        self.color = color
        self.lifetime = lifetime


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Arrow keys to move. Hold Space to attack."
    game_description = "Explore a procedurally generated dungeon, battling enemies to reach and defeat the final boss on level 5."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAP_SIZE = 32
    TILE_WIDTH, TILE_HEIGHT = 48, 24
    
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_FLOOR = (40, 50, 60)
    COLOR_FLOOR_ACCENT = (50, 60, 70)
    COLOR_WALL_TOP = (80, 90, 100)
    COLOR_WALL_SIDE = (60, 70, 80)
    
    COLOR_PLAYER = (50, 150, 255)
    COLOR_PLAYER_GLOW = (50, 150, 255, 50)
    
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_GLOW = (255, 80, 80, 50)
    COLOR_BOSS = (200, 50, 200)
    COLOR_BOSS_GLOW = (200, 50, 200, 50)
    
    COLOR_POTION = (80, 255, 80)
    COLOR_POTION_GLOW = (80, 255, 80, 50)
    
    COLOR_EXIT = (255, 220, 100)
    COLOR_ATTACK = (255, 255, 255)
    
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_HEALTH = (200, 40, 40)
    COLOR_UI_HEALTH_BG = (80, 20, 20)
    COLOR_UI_XP = (40, 180, 200)
    COLOR_UI_XP_BG = (20, 60, 80)
    
    # Game parameters
    PLAYER_SPEED = 0.12
    ENEMY_SPEED = 0.05
    MAX_STEPS = 3000 # Increased from 1000 to allow for exploration
    FINAL_LEVEL = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_huge = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.player = None
        self.grid = None
        self.start_pos = None
        self.exit_pos = None
        self.enemies = []
        self.potions = []
        self.attacks = []
        self.particles = []
        self.floating_texts = []
        
        self.current_level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.game_over_timer = 0
        
        # We call reset once to set up the initial state. The seed will be None,
        # so the RNG will be initialized lazily.
        self.reset()
        # self.validate_implementation() # This can be removed for the final submission

    def _iso_to_screen(self, x, y):
        screen_x = self.SCREEN_WIDTH / 2 + (x - y) * self.TILE_WIDTH / 2
        screen_y = self.SCREEN_HEIGHT / 4 + (x + y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _get_camera_offset(self):
        cam_x, cam_y = self.player.x, self.player.y
        offset_x = self.SCREEN_WIDTH / 2 - (cam_x - cam_y) * self.TILE_WIDTH / 2
        offset_y = self.SCREEN_HEIGHT / 2 - (cam_x + cam_y) * self.TILE_HEIGHT / 2
        return offset_x, offset_y

    def _generate_level(self):
        self.grid = np.ones((self.MAP_SIZE, self.MAP_SIZE), dtype=np.int8)
        self.enemies, self.potions, self.attacks = [], [], []

        start_x, start_y = self.np_random.integers(5, self.MAP_SIZE - 5, size=2)
        self.start_pos = (start_x, start_y)
        
        # Random walk generation
        x, y = start_x, start_y
        path_len = int(self.MAP_SIZE * self.MAP_SIZE * 0.3)
        for _ in range(path_len):
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= x+i < self.MAP_SIZE and 0 <= y+j < self.MAP_SIZE:
                        self.grid[x+i, y+j] = 0
            
            dir = self.np_random.integers(0, 4)
            if dir == 0: x = min(self.MAP_SIZE - 2, x + 1)
            elif dir == 1: x = max(1, x - 1)
            elif dir == 2: y = min(self.MAP_SIZE - 2, y + 1)
            else: y = max(1, y - 1)
        self.exit_pos = (x, y)

        if self.player is None:
            self.player = Player(start_x, start_y)
        else:
            self.player.x, self.player.y = float(start_x), float(start_y)

        # Populate level
        floor_tiles = np.argwhere(self.grid == 0)
        self.np_random.shuffle(floor_tiles)
        
        num_enemies = self.current_level
        if self.current_level == self.FINAL_LEVEL:
            # Boss level
            boss_pos = floor_tiles[0]
            self.enemies.append(Enemy(boss_pos[0], boss_pos[1], is_boss=True))
            floor_tiles = floor_tiles[1:]
        else:
            # Normal level
            for i in range(min(num_enemies, len(floor_tiles))):
                pos = floor_tiles[i]
                if np.linalg.norm(np.array(pos) - np.array(self.start_pos)) > 5:
                    self.enemies.append(Enemy(pos[0], pos[1]))
            floor_tiles = floor_tiles[num_enemies:]

        num_potions = self.np_random.integers(1, 4)
        for i in range(min(num_potions, len(floor_tiles))):
            pos = floor_tiles[i]
            if np.linalg.norm(np.array(pos) - np.array(self.start_pos)) > 3:
                self.potions.append(Potion(pos[0], pos[1]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_level = 1
        
        # Player needs to be created before level generation if it doesn't exist
        if self.player is None:
            # A temporary position, will be overwritten by _generate_level
            self.player = Player(0, 0)
            
        self._generate_level()
        
        self.player.health = self.player.max_health
        self.player.xp = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.game_over_timer = 0
        
        self.particles.clear()
        self.floating_texts.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            self.game_over_timer += 1
            reward = 0
            terminated = self.game_over_timer > 90 # Wait 3 seconds
            if terminated:
                # On the final termination frame, apply the large negative reward
                reward = -100 if not self.game_won else 100
            return self._get_observation(), reward, terminated, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = -0.001 # Small penalty for each step to encourage efficiency

        # --- Update game logic ---
        self._update_player(movement, space_held)
        reward += self._update_enemies()
        reward += self._update_attacks()
        reward += self._update_pickups()
        self._update_effects()
        
        level_up_reward = self._check_level_transition()
        if level_up_reward > 0:
            reward += level_up_reward
        else:
            self.steps += 1
        
        self.score += reward
        terminated = self.player.health <= 0 or self.game_won
        truncated = self.steps >= self.MAX_STEPS
        
        if self.player.health <= 0 and not self.game_over:
            self.game_over = True
            self._create_particles(self.player.x, self.player.y, self.COLOR_PLAYER, 50)
            # The big penalty is applied on the final frame after the death animation
        
        if self.game_won and not self.game_over:
            self.game_over = True
            # The big reward is applied on the final frame
            
        is_final_step = terminated or truncated
        return self._get_observation(), reward, is_final_step and self.game_over_timer > 90, is_final_step and not (self.game_over_timer > 90), self._get_info()

    def _update_player(self, movement, space_held):
        if self.player.iframes > 0: self.player.iframes -= 1
        if self.player.attack_cooldown > 0: self.player.attack_cooldown -= 1
        
        # Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec = np.array([0, -1])  # Up
        elif movement == 2: move_vec = np.array([0, 1])   # Down
        elif movement == 3: move_vec = np.array([-1, 0])  # Left
        elif movement == 4: move_vec = np.array([1, 0])   # Right
        
        if np.linalg.norm(move_vec) > 0:
            self.player.facing = move_vec
            
            new_x = self.player.x + move_vec[0] * self.PLAYER_SPEED
            new_y = self.player.y + move_vec[1] * self.PLAYER_SPEED

            if self.grid[int(new_x), int(self.player.y)] == 0:
                self.player.x = new_x
            if self.grid[int(self.player.x), int(new_y)] == 0:
                self.player.y = new_y

        # Attack
        if space_held and self.player.attack_cooldown == 0:
            # sfx: player_swing.wav
            self.player.attack_cooldown = 15
            attack_pos_x = self.player.x + self.player.facing[0] * 0.7
            attack_pos_y = self.player.y + self.player.facing[1] * 0.7
            self.attacks.append(Attack(attack_pos_x, attack_pos_y, self.player.facing))

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            dist_to_player = np.linalg.norm(np.array([self.player.x, self.player.y]) - np.array([enemy.x, enemy.y]))

            if dist_to_player < 1.0 and self.player.iframes == 0:
                # sfx: player_hit.wav
                self.player.health = max(0, self.player.health - enemy.damage)
                self.player.iframes = 30 # 1 second of invincibility
                self.floating_texts.append(FloatingText(self.player.x, self.player.y, f"-{enemy.damage}", self.COLOR_ENEMY))
                self._create_particles(self.player.x, self.player.y, self.COLOR_ENEMY, 20)
                reward -= 0.5 # Penalty for getting hit

            # Simple AI: move towards player if close, otherwise stand still
            if dist_to_player < 8.0:
                move_dir = np.array([self.player.x - enemy.x, self.player.y - enemy.y])
                norm = np.linalg.norm(move_dir)
                if norm > 0:
                    move_dir /= norm
                    
                    new_x = enemy.x + move_dir[0] * self.ENEMY_SPEED
                    new_y = enemy.y + move_dir[1] * self.ENEMY_SPEED
                    
                    if self.grid[int(new_x), int(enemy.y)] == 0: enemy.x = new_x
                    if self.grid[int(enemy.x), int(new_y)] == 0: enemy.y = new_y
        return reward
    
    def _update_attacks(self):
        reward = 0
        for attack in self.attacks[:]:
            attack.lifetime -= 1
            if attack.lifetime <= 0:
                self.attacks.remove(attack)
                continue
            
            for enemy in self.enemies[:]:
                if enemy in attack.hit_enemies: continue
                dist = np.linalg.norm(np.array([attack.x, attack.y]) - np.array([enemy.x, enemy.y]))
                if dist < 0.5 + enemy.size * 0.5:
                    # sfx: enemy_hit.wav
                    damage = 15
                    enemy.health -= damage
                    attack.hit_enemies.add(enemy)
                    self.floating_texts.append(FloatingText(enemy.x, enemy.y, f"{damage}", self.COLOR_UI_TEXT))
                    self._create_particles(enemy.x, enemy.y, self.COLOR_ENEMY, 15)
                    
                    if enemy.health <= 0:
                        # sfx: enemy_die.wav
                        reward += 0.1 # Per brief
                        self.player.xp += enemy.xp_reward
                        self.floating_texts.append(FloatingText(enemy.x, enemy.y, f"+{enemy.xp_reward} XP", self.COLOR_UI_XP))
                        self._create_particles(enemy.x, enemy.y, self.COLOR_BOSS if enemy.is_boss else self.COLOR_ENEMY, 40)
                        
                        if enemy.is_boss:
                            reward += 100 # Per brief
                            self.game_won = True
                        
                        self.enemies.remove(enemy)
        return reward

    def _update_pickups(self):
        reward = 0
        for potion in self.potions[:]:
            dist = np.linalg.norm(np.array([self.player.x, self.player.y]) - np.array([potion.x, potion.y]))
            if dist < 1.0:
                # sfx: potion_pickup.wav
                heal = min(potion.heal_amount, self.player.max_health - self.player.health)
                if heal > 0:
                    self.player.health += heal
                    reward += 1 # Per brief
                    self.floating_texts.append(FloatingText(self.player.x, self.player.y, f"+{heal}", self.COLOR_POTION))
                    self._create_particles(self.player.x, self.player.y, self.COLOR_POTION, 20)
                self.potions.remove(potion)
        return reward
    
    def _check_level_transition(self):
        dist_to_exit = np.linalg.norm(np.array([self.player.x, self.player.y]) - np.array(self.exit_pos))
        if dist_to_exit < 1.2 and len(self.enemies) == 0:
            # sfx: level_up.wav
            self.current_level += 1
            if self.current_level > self.FINAL_LEVEL:
                self.game_won = True # Should be handled by boss death, but as a fallback
                return 0
            self._generate_level()
            self.steps = 0 # Reset step count for the new level
            return 5 # Per brief
        return 0

    def _update_effects(self):
        for p in self.particles[:]:
            p.lifetime -= 1
            if p.lifetime <= 0:
                self.particles.remove(p)
            else:
                p.x += p.velocity[0]
                p.y += p.velocity[1]
        
        for ft in self.floating_texts[:]:
            ft.lifetime -= 1
            if ft.lifetime <= 0:
                self.floating_texts.remove(ft)
            else:
                ft.y -= 0.02

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.02, 0.1)
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(Particle(x, y, color, lifetime, velocity))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_off_x, cam_off_y = self._get_camera_offset()
        
        # Get visible tile range
        min_x = max(0, int(self.player.x) - 20)
        max_x = min(self.MAP_SIZE, int(self.player.x) + 20)
        min_y = max(0, int(self.player.y) - 20)
        max_y = min(self.MAP_SIZE, int(self.player.y) + 20)
        
        # 1. Draw floor and exit portal
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if self.grid[x, y] == 0:
                    sx, sy = self._iso_to_screen(x, y)
                    sx += cam_off_x
                    sy += cam_off_y
                    
                    points = [
                        (sx, sy - self.TILE_HEIGHT / 2),
                        (sx + self.TILE_WIDTH / 2, sy),
                        (sx, sy + self.TILE_HEIGHT / 2),
                        (sx - self.TILE_WIDTH / 2, sy)
                    ]
                    color = self.COLOR_FLOOR_ACCENT if (x+y)%2 == 0 else self.COLOR_FLOOR
                    pygame.draw.polygon(self.screen, color, points)

        # Draw exit portal
        if self.current_level < self.FINAL_LEVEL or len(self.enemies) > 0:
            ex, ey = self.exit_pos
            sx, sy = self._iso_to_screen(ex, ey)
            sx += cam_off_x
            sy += cam_off_y
            pulse = (math.sin(self.steps * 0.1) + 1) / 2
            radius = int(self.TILE_WIDTH/2 * (0.8 + pulse * 0.2))
            color = self.COLOR_EXIT if len(self.enemies) == 0 else (100,100,100)
            pygame.gfxdraw.filled_circle(self.screen, int(sx), int(sy), radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(sx), int(sy), radius, color)

        # 2. Collect and sort all dynamic entities for Z-ordering
        render_list = self.potions + self.enemies + [self.player]
        render_list.sort(key=lambda e: e.x + e.y)

        # 3. Draw sorted entities (shadows first, then sprites)
        for entity in render_list:
            sx, sy = self._iso_to_screen(entity.x, entity.y)
            sx += cam_off_x
            sy += cam_off_y
            shadow_radius = self.TILE_WIDTH / 4
            pygame.gfxdraw.filled_ellipse(self.screen, int(sx), int(sy + 5), int(shadow_radius), int(shadow_radius/2), (0,0,0,100))

        for entity in render_list:
            sx, sy = self._iso_to_screen(entity.x, entity.y)
            sx += cam_off_x
            sy += cam_off_y
            
            if isinstance(entity, Player):
                if self.player.iframes > 0 and self.player.iframes % 4 < 2: continue
                radius = 8
                color, glow_color = self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW
                pygame.gfxdraw.filled_circle(self.screen, int(sx), int(sy - 10), radius+3, glow_color)
                pygame.gfxdraw.filled_circle(self.screen, int(sx), int(sy - 10), radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(sx), int(sy - 10), radius, color)
            elif isinstance(entity, Enemy):
                radius = int(6 * entity.size)
                color = self.COLOR_BOSS if entity.is_boss else self.COLOR_ENEMY
                glow_color = self.COLOR_BOSS_GLOW if entity.is_boss else self.COLOR_ENEMY_GLOW
                pygame.gfxdraw.filled_circle(self.screen, int(sx), int(sy - 10), radius+3, glow_color)
                pygame.gfxdraw.filled_circle(self.screen, int(sx), int(sy - 10), radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(sx), int(sy - 10), radius, color)
                # Health bar
                bar_w = 20 * entity.size
                bar_h = 4
                h_ratio = entity.health / entity.max_health
                pygame.draw.rect(self.screen, (50,0,0), (sx - bar_w/2, sy - 25, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, (sx - bar_w/2, sy - 25, bar_w * h_ratio, bar_h))
            elif isinstance(entity, Potion):
                radius = 5
                color, glow_color = self.COLOR_POTION, self.COLOR_POTION_GLOW
                pygame.gfxdraw.filled_circle(self.screen, int(sx), int(sy - 8), radius+3, glow_color)
                pygame.gfxdraw.filled_circle(self.screen, int(sx), int(sy - 8), radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(sx), int(sy - 8), radius, color)

        # 4. Draw walls
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if self.grid[x, y] == 1:
                    is_boundary = not (0 < x < self.MAP_SIZE-1 and 0 < y < self.MAP_SIZE-1 and self.grid[x, y-1] == 0)
                    if not is_boundary: continue
                    
                    sx, sy = self._iso_to_screen(x, y)
                    sx += cam_off_x
                    sy += cam_off_y
                    
                    wall_height = self.TILE_HEIGHT * 1.5
                    top_points = [
                        (sx, sy - self.TILE_HEIGHT / 2 - wall_height),
                        (sx + self.TILE_WIDTH / 2, sy - wall_height),
                        (sx, sy + self.TILE_HEIGHT / 2 - wall_height),
                        (sx - self.TILE_WIDTH / 2, sy - wall_height)
                    ]
                    side_points = [
                        (sx - self.TILE_WIDTH / 2, sy - wall_height),
                        (sx + self.TILE_WIDTH / 2, sy - wall_height),
                        (sx + self.TILE_WIDTH / 2, sy),
                        (sx - self.TILE_WIDTH / 2, sy)
                    ]
                    pygame.draw.polygon(self.screen, self.COLOR_WALL_SIDE, side_points)
                    pygame.draw.polygon(self.screen, self.COLOR_WALL_TOP, top_points)

        # 5. Draw attacks, particles, and floating text
        for attack in self.attacks:
            sx, sy = self._iso_to_screen(attack.x, attack.y)
            sx += cam_off_x
            sy += cam_off_y
            alpha = int(255 * (attack.lifetime / 6.0))
            pygame.draw.circle(self.screen, (*self.COLOR_ATTACK, alpha), (int(sx), int(sy-10)), 12, width=3)
            
        for p in self.particles:
            sx, sy = self._iso_to_screen(p.x, p.y)
            sx += cam_off_x
            sy += cam_off_y
            alpha = int(255 * (p.lifetime / 30.0))
            color = (*p.color, alpha)
            pygame.draw.circle(self.screen, color, (int(sx), int(sy-10)), 2)
            
        for ft in self.floating_texts:
            sx, sy = self._iso_to_screen(ft.x, ft.y)
            sx += cam_off_x
            sy += cam_off_y
            alpha = int(255 * (ft.lifetime / 30.0))
            color = (*ft.color[:3], alpha)
            text_surf = self.font_small.render(ft.text, True, color)
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (sx - text_surf.get_width()/2, sy - 20 - (30 - ft.lifetime)*0.5))

    def _render_ui(self):
        # Health Bar
        hp_ratio = self.player.health / self.player.max_health
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, 10, 200 * hp_ratio, 20))
        hp_text = self.font_small.render(f"HP: {self.player.health}/{self.player.max_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(hp_text, (15, 12))

        # XP Bar
        pygame.draw.rect(self.screen, self.COLOR_UI_XP_BG, (10, 35, 200, 15))
        # No max XP, so just show value
        xp_text = self.font_small.render(f"XP: {self.player.xp}", True, self.COLOR_UI_TEXT)
        self.screen.blit(xp_text, (15, 36))

        # Level display
        level_text = self.font_large.render(f"Dungeon Level: {self.current_level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = min(200, self.game_over_timer * 4)
            overlay.fill((0, 0, 0, alpha))
            self.screen.blit(overlay, (0, 0))
            
            if self.game_over_timer > 30:
                text_str = "YOU WON" if self.game_won else "GAME OVER"
                color = self.COLOR_POTION if self.game_won else self.COLOR_ENEMY
                end_text = self.font_huge.render(text_str, True, color)
                self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player.health,
            "xp": self.player.xp,
            "level": self.current_level,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # This is a helper function to check if the implementation is correct.
        # It is not part of the standard gym.Env API.
        print("✓ Running implementation validation...")
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game with keyboard controls.
    # It is not used by the evaluation system, but is helpful for testing.
    os.environ["SDL_VIDEODRIVER"] = "pygame"
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Human Playable Demo ---
    # Store the state of the keys
    keys = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False
    }
    
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Dungeon Crawler")
    
    while not done:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in keys:
                    keys[event.key] = True
            if event.type == pygame.KEYUP:
                if event.key in keys:
                    keys[event.key] = False
        
        # Map keys to action
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] else 0
        
        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()