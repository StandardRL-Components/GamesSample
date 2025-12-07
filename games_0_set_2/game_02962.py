
# Generated: 2025-08-27T21:57:19.993976
# Source Brief: brief_02962.md
# Brief Index: 2962

        
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


# Helper classes for game entities
class Enemy:
    def __init__(self, wave_number, path):
        self.path = path
        self.path_index = 0
        self.pos = np.array(self.path[0], dtype=float)
        
        scale_factor = 1.05 ** (wave_number - 1)
        self.max_health = int(20 * scale_factor)
        self.health = self.max_health
        self.speed = 1.0 * scale_factor
        self.bounty = 10
        self.radius = 8
        self.damage = 5

    def move(self):
        if self.path_index >= len(self.path) - 1:
            return True # Reached the end

        target_pos = np.array(self.path[self.path_index + 1], dtype=float)
        direction = target_pos - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = target_pos
            self.path_index += 1
        else:
            self.pos += (direction / distance) * self.speed
        
        return False

    def draw(self, surface):
        # Body
        pos_int = (int(self.pos[0]), int(self.pos[1]))
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], self.radius, (220, 50, 50))
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], self.radius, (255, 100, 100))
        
        # Health bar
        if self.health < self.max_health:
            bar_width = 20
            bar_height = 4
            health_pct = self.health / self.max_health
            fill_width = int(bar_width * health_pct)
            
            bar_x = pos_int[0] - bar_width // 2
            bar_y = pos_int[1] - self.radius - bar_height - 3
            
            pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(surface, (50, 200, 50), (bar_x, bar_y, fill_width, bar_height))

class Tower:
    def __init__(self, pos, tower_type):
        self.pos = pos
        self.type = tower_type
        self.cooldown = 0
        
        if self.type == 1: # Blue Square - Standard
            self.range = 120
            self.damage = 10
            self.fire_rate = 30 # frames per shot
            self.color = (50, 150, 255)
        elif self.type == 2: # Yellow Triangle - Machine Gun
            self.range = 160
            self.damage = 4
            self.fire_rate = 8
            self.color = (255, 220, 50)
        
    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1
            
    def can_fire(self):
        return self.cooldown == 0
        
    def find_target(self, enemies):
        valid_targets = [e for e in enemies if np.linalg.norm(np.array(self.pos) - e.pos) < self.range]
        if not valid_targets:
            return None
        # Target enemy closest to the base (highest path_index)
        return max(valid_targets, key=lambda e: e.path_index + np.linalg.norm(e.pos - e.path[e.path_index]))

    def draw(self, surface):
        center_x, center_y = self.pos
        if self.type == 1:
            size = 30
            rect = pygame.Rect(center_x - size/2, center_y - size/2, size, size)
            pygame.draw.rect(surface, self.color, rect, border_radius=3)
            pygame.draw.rect(surface, (200, 220, 255), rect, width=3, border_radius=3)
        elif self.type == 2:
            size = 36
            points = [
                (center_x, center_y - size/2),
                (center_x - size/2, center_y + size/2),
                (center_x + size/2, center_y + size/2)
            ]
            pygame.gfxdraw.aapolygon(surface, points, self.color)
            pygame.gfxdraw.filled_polygon(surface, points, self.color)

class Projectile:
    def __init__(self, start_pos, target_enemy, damage, color):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target_enemy
        self.damage = damage
        self.speed = 10.0
        self.color = color

    def update(self):
        direction = self.target.pos - self.pos
        distance = np.linalg.norm(direction)
        
        if distance < self.speed:
            self.target.health -= self.damage
            return True # Hit
        
        self.pos += (direction / distance) * self.speed
        return False

    def draw(self, surface):
        pos_int = (int(self.pos[0]), int(self.pos[1]))
        pygame.draw.rect(surface, self.color, (pos_int[0]-2, pos_int[1]-2, 4, 4))
        
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-1.5, 1.5)
        self.lifespan = random.randint(15, 30)
        self.color = color
        self.radius = random.uniform(2, 4)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius *= 0.95

    def draw(self, surface):
        if self.lifespan > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.radius), self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to move the placement cursor. "
        "Press 'Space' to build the selected tower. "
        "Press 'Shift' to switch between tower types."
    )

    game_description = (
        "A top-down tower defense game. Survive 15 waves of enemies by "
        "strategically placing towers. Earn resources by defeating enemies and "
        "clearing waves."
    )

    auto_advance = True
    
    # Game constants
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 80
    GRID_W, GRID_H = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
    MAX_WAVES = 15
    MAX_STEPS = 30 * 60 * 3 # 3 minutes at 30fps
    BASE_HEALTH_MAX = 50
    BUILD_TIME = 10 * 30 # 10 seconds at 30fps
    
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_GRID = (30, 40, 50)
    COLOR_BASE = (50, 200, 50)
    COLOR_CURSOR_VALID = (255, 255, 255, 50)
    COLOR_CURSOR_INVALID = (255, 50, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    
    # Game States
    STATE_BUILD = 0
    STATE_WAVE = 1
    STATE_GAME_OVER = 2
    STATE_VICTORY = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.path_coords = [(0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (3, 0), (4, 0), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (6, 4), (7, 4)]
        self.path_pixels = [(x * self.GRID_SIZE + self.GRID_SIZE//2, y * self.GRID_SIZE + self.GRID_SIZE//2) for x, y in self.path_coords]
        self.base_pos_grid = self.path_coords[-1]
        self.base_pos_pixels = self.path_pixels[-1]

        self.tower_types = [
            {"name": "Cannon", "cost": 50, "type": 1},
            {"name": "Gatling", "cost": 80, "type": 2},
        ]
        
        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False
        self.game_state = self.STATE_BUILD
        
        self.base_health = self.BASE_HEALTH_MAX
        self.wave_number = 1
        self.resources = 100
        self.build_timer = self.BUILD_TIME
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.enemies_to_spawn = []
        self.spawn_cooldown = 0
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_idx = 0
        self.unlocked_towers = 1
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        # 1. Handle player input
        self._handle_input(action)

        # 2. Update game logic based on state
        if self.game_state == self.STATE_BUILD:
            self.build_timer -= 1
            if self.build_timer <= 0:
                self._start_wave()
        
        elif self.game_state == self.STATE_WAVE:
            # Spawn enemies
            if self.spawn_cooldown > 0:
                self.spawn_cooldown -= 1
            elif self.enemies_to_spawn:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.spawn_cooldown = 30 # 1 second between spawns
            
            # Update entities
            reward += self._update_enemies()
            self._update_towers()
            self._update_projectiles()
            
            # Check for wave completion
            if not self.enemies and not self.enemies_to_spawn:
                reward += 1.0 # Wave clear bonus
                self.resources += 50 + self.wave_number * 5
                self.wave_number += 1
                
                if self.wave_number > self.MAX_WAVES:
                    self.game_state = self.STATE_VICTORY
                    reward += 100
                    terminated = True
                else:
                    self.game_state = self.STATE_BUILD
                    self.build_timer = self.BUILD_TIME
                    if self.wave_number > 5:
                        self.unlocked_towers = 2
        
        self._update_particles()
        
        # 3. Check for termination conditions
        if self.base_health <= 0 and not terminated:
            self.game_state = self.STATE_GAME_OVER
            reward -= 100
            terminated = True
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True # Time limit reached

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        if movement == 2: self.cursor_pos[1] += 1 # Down
        if movement == 3: self.cursor_pos[0] -= 1 # Left
        if movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # --- Place Tower (on key press) ---
        if space_held and not self.prev_space_held:
            self._try_place_tower()
        
        # --- Switch Tower (on key press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % self.unlocked_towers

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _try_place_tower(self):
        if tuple(self.cursor_pos) in self.path_coords: return # Cannot build on path
        if any(t.pos == self._grid_to_pixels(self.cursor_pos) for t in self.towers): return # Already a tower here
        
        tower_info = self.tower_types[self.selected_tower_idx]
        if self.resources >= tower_info["cost"]:
            self.resources -= tower_info["cost"]
            pos = self._grid_to_pixels(self.cursor_pos)
            self.towers.append(Tower(pos, tower_info["type"]))
            # sfx: build_tower.wav
            for _ in range(20):
                self.particles.append(Particle(pos[0], pos[1], (200, 200, 200)))

    def _start_wave(self):
        self.game_state = self.STATE_WAVE
        num_enemies = 3 + self.wave_number * 2
        self.enemies_to_spawn = [Enemy(self.wave_number, self.path_pixels) for _ in range(num_enemies)]
        self.spawn_cooldown = 0
        # sfx: wave_start.wav

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy.move():
                self.base_health -= enemy.damage
                reward -= 0.01 * enemy.damage
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                for _ in range(30):
                    self.particles.append(Particle(self.base_pos_pixels[0], self.base_pos_pixels[1], (255, 50, 50)))
        return reward
        
    def _update_towers(self):
        for tower in self.towers:
            tower.update()
            if tower.can_fire():
                target = tower.find_target(self.enemies)
                if target:
                    self.projectiles.append(Projectile(tower.pos, target, tower.damage, tower.color))
                    tower.cooldown = tower.fire_rate
                    # sfx: laser_shoot.wav

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            if p.update(): # Returns True on hit
                self.projectiles.remove(p)
                # sfx: enemy_hit.wav
                for _ in range(10):
                    self.particles.append(Particle(p.target.pos[0], p.target.pos[1], p.color))
                if p.target.health <= 0 and p.target in self.enemies:
                    self.resources += p.target.bounty
                    self.enemies.remove(p.target)
                    # sfx: enemy_die.wav
                    for _ in range(40):
                        self.particles.append(Particle(p.target.pos[0], p.target.pos[1], (255, 100, 100)))

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
            "steps": self.steps,
            "score": self.wave_number -1 # A simple score metric
        }
        
    def _grid_to_pixels(self, grid_pos):
        return (grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE//2, grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE//2)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_W):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.GRID_SIZE, 0), (x * self.GRID_SIZE, self.HEIGHT))
        for y in range(self.GRID_H):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.GRID_SIZE), (self.WIDTH, y * self.GRID_SIZE))
            
        # Draw path
        if len(self.path_pixels) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_pixels, self.GRID_SIZE)

        # Draw base
        base_rect = pygame.Rect(0, 0, self.GRID_SIZE-10, self.GRID_SIZE-10)
        base_rect.center = self.base_pos_pixels
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        
        # Draw entities
        for tower in self.towers: tower.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        for particle in self.particles: particle.draw(self.screen)

        # Draw cursor and tower range
        cursor_px = self._grid_to_pixels(self.cursor_pos)
        is_valid_pos = tuple(self.cursor_pos) not in self.path_coords and not any(t.pos == cursor_px for t in self.towers)
        cursor_color = self.COLOR_CURSOR_VALID if is_valid_pos else self.COLOR_CURSOR_INVALID
        
        cursor_surface = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        cursor_surface.fill(cursor_color)
        self.screen.blit(cursor_surface, (self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE))
        
        tower_info = self.tower_types[self.selected_tower_idx]
        tower_proto = Tower(cursor_px, tower_info["type"])
        pygame.gfxdraw.aacircle(self.screen, cursor_px[0], cursor_px[1], tower_proto.range, (255,255,255, 50))

    def _render_ui(self):
        # Wave number
        wave_text = self.font_small.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Resources
        res_text = self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (self.WIDTH - res_text.get_width() - 10, 10))

        # Base health bar
        health_bar_width = self.GRID_SIZE - 20
        health_pct = max(0, self.base_health / self.BASE_HEALTH_MAX)
        health_fill_width = int(health_bar_width * health_pct)
        bar_x = self.base_pos_pixels[0] - health_bar_width // 2
        bar_y = self.base_pos_pixels[1] - self.GRID_SIZE // 2 - 10
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, health_bar_width, 8))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, health_fill_width, 8))
        
        # Selected Tower Info
        tower_info = self.tower_types[self.selected_tower_idx]
        tower_name = tower_info["name"]
        tower_cost = tower_info["cost"]
        can_afford = self.resources >= tower_cost
        cost_color = (100, 255, 100) if can_afford else (255, 100, 100)

        sel_text1 = self.font_small.render(f"Selected: {tower_name}", True, self.COLOR_TEXT)
        sel_text2 = self.font_small.render(f"Cost: {tower_cost}", True, cost_color)
        self.screen.blit(sel_text1, (10, self.HEIGHT - 45))
        self.screen.blit(sel_text2, (10, self.HEIGHT - 25))

        # Game State Messages
        if self.game_state == self.STATE_BUILD:
            time_left = math.ceil(self.build_timer / 30)
            build_text = self.font_large.render(f"WAVE STARTING IN {time_left}", True, self.COLOR_TEXT)
            text_rect = build_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(build_text, text_rect)
        elif self.game_state == self.STATE_GAME_OVER:
            end_text = self.font_large.render("GAME OVER", True, (255, 50, 50))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
        elif self.game_state == self.STATE_VICTORY:
            end_text = self.font_large.render("VICTORY!", True, (50, 255, 50))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Action mapping for human play ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = [0, 0, 0] 

    print(env.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        
        # Reset actions
        action = [0, 0, 0]
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Buttons
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Transpose back for pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Wave: {info['wave']}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(3000)

        clock.tick(30)
        
    env.close()