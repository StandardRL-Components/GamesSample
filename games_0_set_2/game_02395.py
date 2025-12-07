# Generated: 2025-08-28T04:40:28.604209
# Source Brief: brief_02395.md
# Brief Index: 2395

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from dataclasses import dataclass, field
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# --- Helper Data Classes for Game Objects ---

@dataclass
class Enemy:
    pos: pygame.Vector2
    health: float
    max_health: float
    speed: float
    path_index: int
    value: float  # Reward for killing
    distance_traveled: float = 0.0
    
@dataclass
class Tower:
    grid_pos: tuple[int, int]
    tower_type: int
    cooldown: int = 0
    
@dataclass
class Projectile:
    pos: pygame.Vector2
    target: Enemy
    speed: float
    damage: float
    
@dataclass
class Particle:
    pos: pygame.Vector2
    vel: pygame.Vector2
    radius: float
    lifespan: int
    color: tuple[int, int, int]


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to move the placement cursor. Press Shift to cycle tower types. Press Space to build."
    )

    game_description = (
        "An isometric tower defense game. Place towers to defend your base from waves of enemies. Survive all 10 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 8
    TILE_WIDTH_ISO, TILE_HEIGHT_ISO = 64, 32
    MAX_STEPS = 5000
    TOTAL_WAVES = 10
    
    # --- Colors ---
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (40, 45, 65)
    COLOR_PATH = (60, 68, 95)
    COLOR_BASE = (0, 100, 200)
    COLOR_BASE_DMG = (255, 0, 0)
    COLOR_CURSOR_VALID = (50, 255, 50, 150)
    COLOR_CURSOR_INVALID = (255, 50, 50, 150)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (40, 200, 80)
    COLOR_HEALTH_BAR_BG = (120, 40, 40)
    
    TOWER_SPECS = {
        0: {"name": "Cannon", "cost": 0, "range": 100, "damage": 15, "fire_rate": 45, "proj_speed": 8, "color": (0, 180, 255)},
        1: {"name": "Missile", "cost": 0, "range": 150, "damage": 40, "fire_rate": 90, "proj_speed": 6, "color": (255, 150, 0)},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Arial", 48, bold=True)

        self.origin_x = self.SCREEN_WIDTH / 2
        self.origin_y = 80

        self._define_path()
        self.path_pixels = [self._iso_to_screen(x, y) for x, y in self.path_coords]

        # Initialize state variables needed for the first render during validation.
        # These are duplicated from reset() to ensure __init__ can run validate_implementation().
        self.steps = 0
        self.score = 0
        self.base_health = 100
        self.game_over = False
        self.game_won = False
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        for x, y in self.path_coords:
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid[x, y] = -1 # Path is not buildable

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self.last_shift_state = 0
        self.last_space_state = 0

        self.current_wave = 0
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        self.time_until_next_wave = 150
        self.game_phase = "placement"

        self.validate_implementation()

    def _define_path(self):
        self.path_coords = []
        for i in range(self.GRID_WIDTH):
            self.path_coords.append((i, 2))
        for i in range(3, self.GRID_HEIGHT):
            self.path_coords.append((self.GRID_WIDTH - 1, i))
        self.path_coords.append((self.GRID_WIDTH, self.GRID_HEIGHT-1)) # Base location

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.origin_x + (grid_x - grid_y) * self.TILE_WIDTH_ISO / 2
        screen_y = self.origin_y + (grid_x + grid_y) * self.TILE_HEIGHT_ISO / 2
        return pygame.Vector2(screen_x, screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.base_health = 100
        self.game_over = False
        self.game_won = False
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        for x, y in self.path_coords:
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid[x, y] = -1 # Path is not buildable

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self.last_shift_state = 0
        self.last_space_state = 0

        self.current_wave = 0
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        self.time_until_next_wave = 150 # 5 seconds at 30fps
        self.game_phase = "placement"
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_action, shift_action = action
        
        reward = 0
        terminated = False

        self._handle_input(movement, space_action, shift_action)
        
        if not self.game_over and not self.game_won:
            self._update_game_state()
            
            killed_enemies = self._update_enemies()
            reward += killed_enemies * 0.1
            self.score += killed_enemies

            self._update_towers()
            self._update_projectiles()
        
        self._update_particles()
        
        if self.base_health <= 0 and not self.game_over:
            self.game_over = True
            reward = -100
        
        if self.game_won and not self.game_over:
            reward = 50
            self.game_over = True # End the game on win

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_action, shift_action):
        # --- Cursor Movement ---
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.GRID_HEIGHT - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.GRID_WIDTH - 1: self.cursor_pos[0] += 1
            
        # --- Cycle Tower (on press) ---
        if shift_action == 1 and self.last_shift_state == 0:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
        self.last_shift_state = shift_action

        # --- Place Tower (on press) ---
        if space_action == 1 and self.last_space_state == 0:
            cx, cy = self.cursor_pos
            if self.grid[cx, cy] == 0: # If cell is empty
                self.grid[cx, cy] = 1
                self.towers.append(Tower(grid_pos=(cx, cy), tower_type=self.selected_tower_type))
                # Sound: tower_place.wav
                pos = self._iso_to_screen(cx, cy)
                for _ in range(20):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 3)
                    vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                    self.particles.append(Particle(pos.copy(), vel, 3, 20, self.TOWER_SPECS[self.selected_tower_type]["color"]))
        self.last_space_state = space_action

    def _update_game_state(self):
        if self.game_phase == "placement":
            self.time_until_next_wave -= 1
            if self.time_until_next_wave <= 0:
                self._start_next_wave()
        elif self.game_phase == "wave":
            if not self.enemies and not self.enemies_to_spawn:
                if self.current_wave == self.TOTAL_WAVES:
                    self.game_won = True
                else:
                    self.game_phase = "placement"
                    self.time_until_next_wave = 300 # 10s
                    self.score += self.current_wave * 10 # Wave clear bonus
            else:
                self._spawn_enemies()

    def _start_next_wave(self):
        self.current_wave += 1
        self.game_phase = "wave"
        num_enemies = 5 + self.current_wave * 2
        base_health = 20 + self.current_wave * 10
        base_speed = 0.8 + self.current_wave * 0.05
        
        for i in range(num_enemies):
            health = base_health * self.np_random.uniform(0.9, 1.1)
            speed = base_speed * self.np_random.uniform(0.9, 1.1)
            self.enemies_to_spawn.append({"health": health, "speed": speed, "delay": i * 30})
        self.spawn_timer = 0

    def _spawn_enemies(self):
        self.spawn_timer += 1
        if self.enemies_to_spawn and self.spawn_timer >= self.enemies_to_spawn[0]["delay"]:
            spec = self.enemies_to_spawn.pop(0)
            start_pos = self.path_pixels[0].copy()
            new_enemy = Enemy(pos=start_pos, health=spec["health"], max_health=spec["health"], speed=spec["speed"], path_index=0, value=1)
            self.enemies.append(new_enemy)
            # Sound: enemy_spawn.wav

    def _update_enemies(self):
        killed_count = 0
        for enemy in self.enemies[:]:
            if enemy.path_index >= len(self.path_pixels) - 1:
                self.base_health -= 10
                self.enemies.remove(enemy)
                # Sound: base_damage.wav
                continue

            target_node_pos = self.path_pixels[enemy.path_index + 1]
            direction = (target_node_pos - enemy.pos).normalize()
            enemy.pos += direction * enemy.speed
            enemy.distance_traveled += enemy.speed

            if enemy.pos.distance_to(target_node_pos) < enemy.speed:
                enemy.path_index += 1
            
            if enemy.health <= 0:
                # Sound: enemy_explode.wav
                for _ in range(30):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                    self.particles.append(Particle(enemy.pos.copy(), vel, self.np_random.uniform(2, 4), 30, (255, 80, 80)))
                self.enemies.remove(enemy)
                killed_count += 1
        return killed_count

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower.tower_type]
            if tower.cooldown > 0:
                tower.cooldown -= 1
                continue

            tower_pos = self._iso_to_screen(tower.grid_pos[0], tower.grid_pos[1])
            
            # Find best target (furthest along path)
            best_target = None
            max_dist = -1
            for enemy in self.enemies:
                if tower_pos.distance_to(enemy.pos) <= spec["range"]:
                    if enemy.distance_traveled > max_dist:
                        max_dist = enemy.distance_traveled
                        best_target = enemy
            
            if best_target:
                # Sound: cannon_fire.wav or missile_launch.wav
                self.projectiles.append(Projectile(
                    pos=tower_pos.copy(),
                    target=best_target,
                    speed=spec["proj_speed"],
                    damage=spec["damage"]
                ))
                tower.cooldown = spec["fire_rate"]

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj.target not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            direction = (proj.target.pos - proj.pos).normalize()
            proj.pos += direction * proj.speed

            if proj.pos.distance_to(proj.target.pos) < proj.speed:
                proj.target.health -= proj.damage
                self.projectiles.remove(proj)
                # Sound: projectile_hit.wav
                for _ in range(10):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(0.5, 2)
                    vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                    self.particles.append(Particle(proj.pos.copy(), vel, self.np_random.uniform(1, 3), 15, (255, 255, 100)))

    def _update_particles(self):
        for p in self.particles[:]:
            p.pos += p.vel
            p.lifespan -= 1
            p.radius *= 0.95
            if p.lifespan <= 0 or p.radius < 0.5:
                self.particles.remove(p)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "current_wave": self.current_wave,
            "towers": len(self.towers),
            "enemies": len(self.enemies),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], self.COLOR_GRID)

        # Draw path
        for i in range(len(self.path_pixels) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_pixels[i], self.path_pixels[i+1], 10)
        
        # Draw base
        base_pos = self.path_pixels[-1]
        base_color = self.COLOR_BASE if self.base_health > 30 else self.COLOR_BASE_DMG
        pygame.gfxdraw.filled_circle(self.screen, int(base_pos.x), int(base_pos.y), 15, base_color)
        pygame.gfxdraw.aacircle(self.screen, int(base_pos.x), int(base_pos.y), 15, tuple(c*0.8 for c in base_color))

        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower.tower_type]
            pos = self._iso_to_screen(tower.grid_pos[0], tower.grid_pos[1])
            p1 = (pos.x, pos.y - 10)
            p2 = (pos.x - 10, pos.y)
            p3 = (pos.x, pos.y + 10)
            p4 = (pos.x + 10, pos.y)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], spec["color"])
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], tuple(c*0.7 for c in spec["color"]))

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_center = self._iso_to_screen(cx, cy)
        p1 = self._iso_to_screen(cx, cy)
        p2 = self._iso_to_screen(cx + 1, cy)
        p3 = self._iso_to_screen(cx + 1, cy + 1)
        p4 = self._iso_to_screen(cx, cy + 1)
        is_valid = self.grid[cx, cy] == 0
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], cursor_color)

        # Draw enemies
        for enemy in sorted(self.enemies, key=lambda e: e.pos.y):
            pygame.gfxdraw.filled_circle(self.screen, int(enemy.pos.x), int(enemy.pos.y), 6, (220, 50, 50))
            pygame.gfxdraw.aacircle(self.screen, int(enemy.pos.x), int(enemy.pos.y), 6, (150, 20, 20))
            # Health bar
            bar_width = 15
            health_pct = enemy.health / enemy.max_health
            pygame.draw.rect(self.screen, (100,0,0), (enemy.pos.x - bar_width/2, enemy.pos.y - 12, bar_width, 3))
            pygame.draw.rect(self.screen, (0,200,0), (enemy.pos.x - bar_width/2, enemy.pos.y - 12, bar_width * health_pct, 3))

        # Draw projectiles and particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p.lifespan / 15))))
            color = (*p.color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.radius), color)
        for proj in self.projectiles:
            spec = self.TOWER_SPECS[1] if proj.damage > 20 else self.TOWER_SPECS[0]
            pygame.gfxdraw.filled_circle(self.screen, int(proj.pos.x), int(proj.pos.y), 4, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, int(proj.pos.x), int(proj.pos.y), 4, (255,255,255))

    def _render_ui(self):
        # --- Top Bar ---
        bar_height = 30
        pygame.draw.rect(self.screen, (10, 12, 22), (0, 0, self.SCREEN_WIDTH, bar_height))
        
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 7))
        
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - 120, 7))

        # Base Health Bar
        health_bar_width = 200
        health_pct = max(0, self.base_health / 100)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, ((self.SCREEN_WIDTH - health_bar_width) / 2, 5, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, ((self.SCREEN_WIDTH - health_bar_width) / 2, 5, health_bar_width * health_pct, 20))
        health_text = self.font_small.render(f"BASE HEALTH: {self.base_health}%", True, self.COLOR_TEXT)
        text_rect = health_text.get_rect(center=(self.SCREEN_WIDTH / 2, 15))
        self.screen.blit(health_text, text_rect)

        # --- Bottom Bar (Tower Selection) ---
        bar_height_bottom = 40
        pygame.draw.rect(self.screen, (10, 12, 22), (0, self.SCREEN_HEIGHT - bar_height_bottom, self.SCREEN_WIDTH, bar_height_bottom))
        
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_info_text = self.font_small.render(
            f"Selected: {spec['name']} | DMG: {spec['damage']} | RNG: {spec['range']} | RATE: {60/spec['fire_rate']:.1f}/s",
            True, self.COLOR_TEXT
        )
        self.screen.blit(tower_info_text, (10, self.SCREEN_HEIGHT - 28))

        # --- Phase Text ---
        if self.game_phase == "placement" and not self.game_won:
            secs_left = math.ceil(self.time_until_next_wave / 30)
            phase_text = self.font_large.render(f"NEXT WAVE IN {secs_left}", True, self.COLOR_TEXT)
            text_rect = phase_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 80))
            self.screen.blit(phase_text, text_rect)
        
        if self.game_over:
            status_text = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            end_text = self.font_huge.render(status_text, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen for display
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Tower Defense")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()