
# Generated: 2025-08-27T23:59:23.368628
# Source Brief: brief_03650.md
# Brief Index: 3650

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# --- Helper Classes for Game Entities ---

class Particle:
    """A simple class for particle effects."""
    def __init__(self, x, y, vx, vy, size, lifespan, color, gravity=0.1):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.lifespan = lifespan
        self.color = color
        self.gravity = gravity

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

    def is_alive(self):
        return self.lifespan > 0 and self.size > 0

class BaseEntity:
    """Base class for all game entities (Player, Monster, Boss)."""
    def __init__(self, x, y, radius, color, max_health):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.max_health = max_health
        self.health = max_health
        self.hit_flash_timer = 0
        self.is_alive = True

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)
        self.hit_flash_timer = 10  # Flash for 10 frames
        if self.health == 0:
            self.is_alive = False
        return amount

    def update(self):
        if self.hit_flash_timer > 0:
            self.hit_flash_timer -= 1

    def get_render_color(self):
        if self.hit_flash_timer > 0 and self.hit_flash_timer % 2 == 0:
            return (255, 255, 255)  # White flash
        return self.color

class Player(BaseEntity):
    """The player character."""
    def __init__(self, x, y):
        super().__init__(x, y, radius=1.0, color=(255, 80, 80), max_health=50)
        self.attack_cooldown = 0
        self.attack_timer = 0
        self.attack_arc = None
        self.dodge_cooldown = 0
        self.dodge_timer = 0
        self.dodge_path = None
        self.last_move_dir = (0, 1) # Default down

    def is_invulnerable(self):
        return self.dodge_timer > 0

    def start_attack(self):
        if self.attack_cooldown == 0:
            self.attack_cooldown = 20 # 2/3 second cooldown
            self.attack_timer = 8 # Lasts 8 frames
            # SFX: Player sword swing
            return True
        return False

    def start_dodge(self, move_dir):
        if self.dodge_cooldown == 0:
            self.dodge_cooldown = 45 # 1.5 second cooldown
            self.dodge_timer = 10 # Dodge lasts 10 frames, invulnerable
            self.dodge_path = (self.x, self.y, self.x + move_dir[0] * 3, self.y + move_dir[1] * 3)
            # SFX: Player whoosh
            return True
        return False

    def update(self):
        super().update()
        if self.attack_cooldown > 0: self.attack_cooldown -= 1
        if self.attack_timer > 0: self.attack_timer -= 1
        if self.dodge_cooldown > 0: self.dodge_cooldown -= 1
        if self.dodge_timer > 0:
            self.dodge_timer -= 1
            # Interpolate position during dodge
            progress = 1.0 - (self.dodge_timer / 10.0)
            self.x = self.dodge_path[0] + (self.dodge_path[2] - self.dodge_path[0]) * progress
            self.y = self.dodge_path[1] + (self.dodge_path[3] - self.dodge_path[1]) * progress


class Monster(BaseEntity):
    """A basic enemy monster."""
    def __init__(self, x, y):
        super().__init__(x, y, radius=0.8, color=(80, 220, 80), max_health=10)
        self.attack_cooldown = 0
        self.speed = 0.06

    def update(self, player_pos):
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        
        dx = player_pos[0] - self.x
        dy = player_pos[1] - self.y
        dist = math.hypot(dx, dy)

        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed

    def can_attack(self, player_pos):
        if self.attack_cooldown == 0:
            dist = math.hypot(player_pos[0] - self.x, player_pos[1] - self.y)
            if dist < self.radius + 1.0:
                self.attack_cooldown = 60 # Attack every 2 seconds
                # SFX: Monster bite/hit
                return True
        return False

class Boss(BaseEntity):
    """The final boss."""
    def __init__(self, x, y):
        super().__init__(x, y, radius=2.0, color=(80, 80, 255), max_health=50)
        self.attack_cooldown = 0
        self.speed = 0.04
        self.phase_timer = 0
        self.state = "CHASE" # CHASE, TELEGRAPH_AOE, AOE_ATTACK
        self.aoe_center = None
        self.aoe_radius = 5

    def update(self, player_pos):
        super().update()
        self.phase_timer += 1
        
        if self.state == "CHASE":
            if self.phase_timer > 150: # Chase for 5 seconds
                self.phase_timer = 0
                self.state = "TELEGRAPH_AOE"
                self.aoe_center = (player_pos[0], player_pos[1])
                # SFX: Boss power up
            else: # Move towards player
                dx = player_pos[0] - self.x
                dy = player_pos[1] - self.y
                dist = math.hypot(dx, dy)
                if dist > self.radius:
                    self.x += (dx / dist) * self.speed
                    self.y += (dy / dist) * self.speed

        elif self.state == "TELEGRAPH_AOE":
            if self.phase_timer > 45: # Telegraph for 1.5 seconds
                self.phase_timer = 0
                self.state = "AOE_ATTACK"
                # SFX: Boss area attack explosion
            
        elif self.state == "AOE_ATTACK":
            if self.phase_timer > 15: # Attack lasts for 0.5 seconds
                self.phase_timer = 0
                self.state = "CHASE"
                self.aoe_center = None


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold shift to dodge and press space to attack."
    )

    game_description = (
        "Defeat waves of monsters and their boss in an isometric-2D action arena."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width, self.screen_height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Game Constants ---
        self.COLOR_BG = (40, 42, 54)
        self.COLOR_GRID = (60, 62, 74)
        self.COLOR_UI_TEXT = (248, 248, 242)
        self.COLOR_HEALTH_BG = (80, 80, 80)
        self.COLOR_HEALTH_FG = (255, 80, 80)
        self.FONT = pygame.font.Font(None, 32)
        self.FONT_LARGE = pygame.font.Font(None, 64)
        
        # --- World Properties ---
        self.world_size = 16
        self.iso_origin = (self.screen_width // 2, 80)
        self.tile_w, self.tile_h = 32, 16
        
        # --- Game State ---
        self.player = None
        self.monsters = []
        self.boss = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.max_steps = 30 * 60 # 60 seconds at 30fps

        self.reset()
        self.validate_implementation()

    def _to_iso(self, x, y):
        """Converts grid coordinates to isometric screen coordinates."""
        screen_x = self.iso_origin[0] + (x - y) * self.tile_w / 2
        screen_y = self.iso_origin[1] + (x + y) * self.tile_h / 2
        return int(screen_x), int(screen_y)

    def _create_particles(self, x, y, color, count=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 2 # Move up initially
            size = random.uniform(2, 5)
            lifespan = random.randint(20, 40)
            self.particles.append(Particle(x, y, vx, vy, size, lifespan, color))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player = Player(self.world_size / 2, self.world_size / 2)
        
        self.monsters = []
        for _ in range(10):
            x = self.np_random.uniform(1, self.world_size - 1)
            y = self.np_random.uniform(1, self.world_size - 1)
            self.monsters.append(Monster(x, y))

        self.boss = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # --- Handle Player Input ---
        movement_action = action[0]
        attack_action = action[1] == 1
        dodge_action = action[2] == 1

        move_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement_action, (0, 0))
        if move_dir != (0,0):
            self.player.last_move_dir = move_dir
        
        # Player is controlled by dodge or regular movement, not both
        if self.player.dodge_timer == 0:
            if dodge_action:
                self.player.start_dodge(self.player.last_move_dir)
            else:
                next_x = self.player.x + move_dir[0] * 0.15
                next_y = self.player.y + move_dir[1] * 0.15
                self.player.x = np.clip(next_x, 0, self.world_size)
                self.player.y = np.clip(next_y, 0, self.world_size)

        if attack_action:
            self.player.start_attack()

        # --- Update Game State ---
        self.player.update()
        for monster in self.monsters:
            monster.update((self.player.x, self.player.y))
        if self.boss:
            self.boss.update((self.player.x, self.player.y))

        # --- Handle Attacks and Collisions ---
        # Player attack
        if self.player.attack_timer > 0:
            attack_angle_base = math.atan2(self.player.last_move_dir[1], self.player.last_move_dir[0])
            attack_radius = 2.0
            for enemy in self.monsters + ([self.boss] if self.boss else []):
                dx = enemy.x - self.player.x
                dy = enemy.y - self.player.y
                dist = math.hypot(dx, dy)
                if 0 < dist < attack_radius + enemy.radius:
                    angle_to_enemy = math.atan2(dy, dx)
                    angle_diff = (attack_angle_base - angle_to_enemy + math.pi) % (2*math.pi) - math.pi
                    if abs(angle_diff) < math.pi / 3: # 120 degree arc
                        dmg = enemy.take_damage(1)
                        reward += 0.1 * dmg
                        # SFX: Enemy hit
                        if not enemy.is_alive:
                            reward += 10.0 if isinstance(enemy, Boss) else 1.0
                            self.score += 100 if isinstance(enemy, Boss) else 10
                            px, py = self._to_iso(enemy.x, enemy.y)
                            self._create_particles(px, py, enemy.color, 40)
                            if isinstance(enemy, Boss):
                                self.game_won = True

        # Enemy attacks
        if not self.player.is_invulnerable():
            for monster in self.monsters:
                if monster.can_attack((self.player.x, self.player.y)):
                    dmg = self.player.take_damage(5)
                    reward -= 0.1 * dmg
            
            if self.boss and self.boss.state == "AOE_ATTACK":
                dist_to_aoe = math.hypot(self.player.x - self.boss.aoe_center[0], self.player.y - self.boss.aoe_center[1])
                if dist_to_aoe < self.boss.aoe_radius:
                    dmg = self.player.take_damage(10)
                    reward -= 0.1 * dmg
        
        # --- Update Lists ---
        self.monsters = [m for m in self.monsters if m.is_alive]
        if self.boss and not self.boss.is_alive:
            self.boss = None
        
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles: p.update()

        # --- Game Flow ---
        if not self.monsters and not self.boss and not self.game_won:
            self.boss = Boss(self.world_size/2, 2)
            # SFX: Boss appear
        
        # --- Termination ---
        self.steps += 1
        terminated = not self.player.is_alive or self.game_won or self.steps >= self.max_steps
        if terminated and self.game_won:
            reward += 100 # Large bonus for winning

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(self.world_size + 1):
            p1 = self._to_iso(i, 0)
            p2 = self._to_iso(i, self.world_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
            p1 = self._to_iso(0, i)
            p2 = self._to_iso(self.world_size, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        # Collect all entities to sort by y-position for correct rendering
        entities = self.monsters + ([self.player] if self.player else []) + ([self.boss] if self.boss else [])
        entities.sort(key=lambda e: e.y)

        # Render boss AOE telegraph
        if self.boss and self.boss.state == "TELEGRAPH_AOE":
            alpha = 100 + 100 * math.sin(self.steps * 0.3)
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            iso_pos = self._to_iso(self.boss.aoe_center[0], self.boss.aoe_center[1])
            pygame.gfxdraw.filled_circle(s, iso_pos[0], iso_pos[1], int(self.boss.aoe_radius * self.tile_w/2), (*self.boss.color, alpha))
            self.screen.blit(s, (0,0))

        # Render boss AOE attack
        if self.boss and self.boss.state == "AOE_ATTACK":
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            iso_pos = self._to_iso(self.boss.aoe_center[0], self.boss.aoe_center[1])
            radius = int(self.boss.aoe_radius * self.tile_w/2 * (self.boss.phase_timer/15))
            alpha = 255 * (1 - self.boss.phase_timer/15)
            pygame.gfxdraw.filled_circle(s, iso_pos[0], iso_pos[1], radius, (*self.boss.color, alpha))
            pygame.gfxdraw.aacircle(s, iso_pos[0], iso_pos[1], radius, (*self.boss.color, alpha))
            self.screen.blit(s, (0,0))

        # Render entities
        for entity in entities:
            iso_pos = self._to_iso(entity.x, entity.y)
            shadow_pos = (iso_pos[0], iso_pos[1] + int(entity.radius * 5))
            pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1], int(entity.radius * self.tile_w/2.5), int(entity.radius * self.tile_h/2.5), (0,0,0,100))

            render_color = entity.get_render_color()
            if isinstance(entity, Player) and entity.is_invulnerable():
                s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(s, iso_pos[0], iso_pos[1], int(entity.radius * self.tile_w/2), (*render_color, 128))
                pygame.gfxdraw.aacircle(s, iso_pos[0], iso_pos[1], int(entity.radius * self.tile_w/2), (*render_color, 128))
                self.screen.blit(s, (0,0))
            else:
                pygame.gfxdraw.filled_circle(self.screen, iso_pos[0], iso_pos[1], int(entity.radius * self.tile_w/2), render_color)
                pygame.gfxdraw.aacircle(self.screen, iso_pos[0], iso_pos[1], int(entity.radius * self.tile_w/2), render_color)

            # Health bar
            if entity.health < entity.max_health:
                bar_width = int(entity.radius * self.tile_w)
                bar_height = 5
                bar_x = iso_pos[0] - bar_width // 2
                bar_y = iso_pos[1] - int(entity.radius * self.tile_w/2) - 15
                health_pct = entity.health / entity.max_health
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, int(bar_width * health_pct), bar_height))
        
        # Player attack arc
        if self.player.attack_timer > 0:
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            p_pos = self._to_iso(self.player.x, self.player.y)
            progress = 1.0 - self.player.attack_timer / 8.0
            angle_base = math.atan2(self.player.last_move_dir[1], self.player.last_move_dir[0])
            start_angle = angle_base - math.pi / 3
            end_angle = angle_base + math.pi / 3
            current_angle = start_angle + (end_angle - start_angle) * progress
            radius = int(2.0 * self.tile_w / 2)
            
            for i in range(4): # Draw a few trail segments
                trail_progress = max(0, progress - i * 0.05)
                trail_angle = start_angle + (end_angle - start_angle) * trail_progress
                p1 = (p_pos[0], p_pos[1])
                p2 = (p_pos[0] + radius * math.cos(trail_angle), p_pos[1] + radius * math.sin(trail_angle) * 0.5) # 0.5 for iso perspective
                alpha = 255 - i * 60
                pygame.draw.line(s, (255, 255, 100, alpha), p1, p2, 4)
            self.screen.blit(s, (0,0))

        # Render particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), int(p.size), p.color)

    def _render_ui(self):
        # Score
        score_surf = self.FONT.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Health
        health_surf = self.FONT.render(f"HEALTH: {self.player.health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_surf, (10, 40))

        # Cooldowns
        if self.player.attack_cooldown > 0:
            pygame.draw.rect(self.screen, (100,100,100), (10, 75, 100, 10))
            cd_pct = 1 - self.player.attack_cooldown / 20
            pygame.draw.rect(self.screen, (255,255,100), (10, 75, int(100 * cd_pct), 10))
        
        if self.player.dodge_cooldown > 0:
            pygame.draw.rect(self.screen, (100,100,100), (10, 90, 100, 10))
            cd_pct = 1 - self.player.dodge_cooldown / 45
            pygame.draw.rect(self.screen, (100,100,255), (10, 90, int(100 * cd_pct), 10))

        if self.game_over:
            text = "VICTORY!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            end_surf = self.FONT_LARGE.render(text, True, color)
            end_rect = end_surf.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(end_surf, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player.health,
            "monsters_left": len(self.monsters),
            "boss_active": self.boss is not None,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")