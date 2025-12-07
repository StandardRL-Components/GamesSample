
# Generated: 2025-08-28T05:09:44.886603
# Source Brief: brief_02535.md
# Brief Index: 2535

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    def __init__(self, health, speed, path, path_offset):
        self.max_health = health
        self.health = health
        self.speed = speed
        self.path = path
        self.path_index = 0
        self.pos = np.array(self.path[0], dtype=float)
        self.radius = 8
        self.path_offset = path_offset
        self.pos += self.path_offset
        self.value = int(health / 10) # Score value when defeated

    def move(self):
        if self.path_index >= len(self.path) - 1:
            return True  # Reached the end

        target_pos = np.array(self.path[self.path_index + 1], dtype=float) + self.path_offset
        direction = target_pos - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = target_pos
            self.path_index += 1
        else:
            self.pos += (direction / distance) * self.speed
        return False

    def draw(self, surface):
        # Health bar
        bar_width = self.radius * 2
        bar_height = 4
        health_pct = self.health / self.max_health
        fill_width = int(bar_width * health_pct)
        
        # Enemy body
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, (200, 0, 50))
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, (255, 80, 120))
        
        # Health bar background
        pygame.draw.rect(surface, (50, 50, 50), (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius - 8), bar_width, bar_height))
        # Health bar fill
        pygame.draw.rect(surface, (0, 255, 0) if health_pct > 0.5 else (255, 255, 0), (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius - 8), fill_width, bar_height))


class Tower:
    TOWER_SPECS = {
        1: {"name": "Cannon", "color": (0, 150, 255), "range": 80, "damage": 12, "fire_rate": 1.0, "cost": 10},
        2: {"name": "Gatling", "color": (255, 255, 0), "range": 60, "damage": 5, "fire_rate": 3.0, "cost": 15},
        3: {"name": "Sniper", "color": (150, 0, 255), "range": 150, "damage": 30, "fire_rate": 0.5, "cost": 25},
        4: {"name": "Heavy", "color": (255, 100, 0), "range": 70, "damage": 40, "fire_rate": 0.4, "cost": 30},
    }

    def __init__(self, pos, tower_type):
        self.pos = pos
        self.type = tower_type
        spec = self.TOWER_SPECS[tower_type]
        self.range = spec["range"]
        self.damage = spec["damage"]
        self.fire_rate = spec["fire_rate"]
        self.cooldown = 0
        self.color = spec["color"]
        self.angle = 0

    def update(self, enemies, projectiles):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        target = self.find_target(enemies)
        if target:
            # Point at target
            dx = target.pos[0] - self.pos[0]
            dy = target.pos[1] - self.pos[1]
            self.angle = math.atan2(dy, dx)
            
            # Fire
            projectiles.append(Projectile(self.pos, target, self.damage, self.color))
            self.cooldown = 30 / self.fire_rate
            # Sound effect placeholder: # sfx_tower_fire()
            return (self.pos, self.angle) # Return muzzle flash info
        return None

    def find_target(self, enemies):
        for enemy in enemies:
            dist = np.linalg.norm(np.array(self.pos) - enemy.pos)
            if dist <= self.range:
                return enemy
        return None

    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        
        # Glow effect
        glow_color = (*self.color, 50)
        pygame.gfxdraw.filled_circle(surface, x, y, 15, glow_color)
        
        # Base
        pygame.gfxdraw.filled_circle(surface, x, y, 10, (50, 50, 60))
        pygame.gfxdraw.aacircle(surface, x, y, 10, (80, 80, 90))

        # Barrel
        end_x = x + 12 * math.cos(self.angle)
        end_y = y + 12 * math.sin(self.angle)
        pygame.draw.line(surface, self.color, (x, y), (end_x, end_y), 4)

class Projectile:
    def __init__(self, start_pos, target, damage, color):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target
        self.damage = damage
        self.color = color
        self.speed = 10.0

    def move(self):
        if self.target.health <= 0: # Target already dead
            return True

        direction = self.target.pos - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = self.target.pos
            return True  # Hit target
        else:
            self.pos += (direction / distance) * self.speed
        return False

    def draw(self, surface):
        pygame.draw.line(surface, self.color, (int(self.pos[0]), int(self.pos[1])), (int(self.pos[0]-2), int(self.pos[1]-2)), 2)


class Particle:
    def __init__(self, pos, vel, size, color, lifespan):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.size = size
        self.color = color
        self.lifespan = lifespan
        self.life = lifespan

    def update(self):
        self.pos += self.vel
        self.life -= 1
        return self.life <= 0

    def draw(self, surface):
        alpha = int(255 * (self.life / self.lifespan))
        current_size = int(self.size * (self.life / self.lifespan))
        if current_size > 0:
            color = (*self.color, alpha)
            temp_surf = pygame.Surface((current_size * 2, current_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (current_size, current_size), current_size)
            surface.blit(temp_surf, (int(self.pos[0] - current_size), int(self.pos[1] - current_size)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. Press SHIFT to cycle through tower types. Press SPACE to place the selected tower."
    )

    game_description = (
        "A minimalist tower defense game. Strategically place towers to defend your base from waves of incoming enemies."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 18, 22)
        self.COLOR_PATH = (30, 35, 45)
        self.COLOR_BASE = (0, 180, 100)
        self.COLOR_UI_TEXT = (220, 220, 220)

        self._define_path_and_grid()

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.enemies_to_spawn = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.max_base_health = 100
        self.current_wave = 0
        self.total_waves = 15
        self.wave_timer = 0
        self.selected_tower_type = 1
        self.placement_cursor_idx = 0
        self.prev_shift_held = False
        self.prev_space_held = False

        self.reset()
        self.validate_implementation()

    def _define_path_and_grid(self):
        self.path = [
            (-20, 100), (100, 100), (100, 300), (540, 300), (540, 160), (self.WIDTH + 20, 160)
        ]
        self.base_pos = (self.WIDTH, 160)
        
        self.placement_grid = []
        path_rects = []
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            rect = pygame.Rect(min(p1[0], p2[0]) - self.GRID_SIZE, min(p1[1], p2[1]) - self.GRID_SIZE,
                               abs(p1[0] - p2[0]) + 2 * self.GRID_SIZE, abs(p1[1] - p2[1]) + 2 * self.GRID_SIZE)
            path_rects.append(rect)

        for y in range(1, self.GRID_H - 1):
            for x in range(1, self.GRID_W - 1):
                pos = (x * self.GRID_SIZE + self.GRID_SIZE // 2, y * self.GRID_SIZE + self.GRID_SIZE // 2)
                cell_rect = pygame.Rect(pos[0] - self.GRID_SIZE//2, pos[1] - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE)
                on_path = any(cell_rect.colliderect(pr) for pr in path_rects)
                if not on_path:
                    self.placement_grid.append(pos)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.max_base_health
        self.current_wave = 0
        self.wave_timer = 120 # Time until first wave
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.enemies_to_spawn = []

        self.selected_tower_type = 1
        self.placement_cursor_idx = len(self.placement_grid) // 2
        self.prev_shift_held = True
        self.prev_space_held = True
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Time penalty
        
        # 1. Handle player actions
        reward += self._handle_actions(action)
        
        # 2. Update game state (towers, enemies, projectiles)
        muzzle_flashes = self._update_towers()
        self._create_muzzle_flashes(muzzle_flashes)
        
        reward += self._update_projectiles()
        reward += self._update_enemies()
        
        self._update_particles()

        # 3. Wave Management
        reward += self._manage_waves()
        
        # 4. Check for termination
        self.steps += 1
        terminated = self._check_termination()
        if terminated and self.base_health > 0 and self.current_wave > self.total_waves:
            reward += 100 # Victory bonus
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Cycle tower type on SHIFT press
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type % 4) + 1
            # Sound effect placeholder: # sfx_ui_select()
        self.prev_shift_held = shift_held

        # Move cursor
        if movement != 0:
            self._move_cursor(movement)

        # Place tower on SPACE press
        if space_held and not self.prev_space_held:
            cursor_pos = self.placement_grid[self.placement_cursor_idx]
            cost = Tower.TOWER_SPECS[self.selected_tower_type]["cost"]
            
            is_occupied = any(t.pos == cursor_pos for t in self.towers)
            
            if not is_occupied and self.score >= cost:
                self.towers.append(Tower(cursor_pos, self.selected_tower_type))
                self.score -= cost
                reward -= 0.1 # Small penalty for placing
                # Sound effect placeholder: # sfx_place_tower()
        self.prev_space_held = space_held

        return reward

    def _move_cursor(self, direction):
        current_pos = np.array(self.placement_grid[self.placement_cursor_idx])
        best_idx = self.placement_cursor_idx
        min_dist = float('inf')

        # Find the closest valid grid point in the desired direction
        for i, pos in enumerate(self.placement_grid):
            if i == self.placement_cursor_idx:
                continue
            
            diff = np.array(pos) - current_pos
            dist = np.linalg.norm(diff)
            
            if direction == 1 and diff[1] < -1 and abs(diff[0]) < abs(diff[1]): # Up
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            elif direction == 2 and diff[1] > 1 and abs(diff[0]) < abs(diff[1]): # Down
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            elif direction == 3 and diff[0] < -1 and abs(diff[1]) < abs(diff[0]): # Left
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            elif direction == 4 and diff[0] > 1 and abs(diff[1]) < abs(diff[0]): # Right
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        
        self.placement_cursor_idx = best_idx

    def _update_towers(self):
        muzzle_flashes = []
        for tower in self.towers:
            flash_info = tower.update(self.enemies, self.projectiles)
            if flash_info:
                muzzle_flashes.append(flash_info)
        return muzzle_flashes

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        for p in self.projectiles:
            if p.move():
                projectiles_to_remove.append(p)
                if p.target.health > 0: # Check if target is still alive
                    p.target.health -= p.damage
                    reward += 0.1 # Reward for hitting
                    self._create_explosion(p.pos, p.color, 5)
                    # Sound effect placeholder: # sfx_projectile_hit()
                    if p.target.health <= 0:
                        self.score += p.target.value
                        reward += 1.0 # Reward for kill
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []
        for e in self.enemies:
            if e.health <= 0:
                enemies_to_remove.append(e)
                self._create_explosion(e.pos, (255, 180, 0), 10)
                # Sound effect placeholder: # sfx_enemy_die()
                continue
            if e.move():
                enemies_to_remove.append(e)
                self.base_health -= 10
                reward -= 10.0 # Penalty for reaching base
                self._create_explosion(self.base_pos, (255,0,0), 20)
                # Sound effect placeholder: # sfx_base_damage()
        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        return reward
        
    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

    def _manage_waves(self):
        reward = 0
        if not self.enemies and not self.enemies_to_spawn and self.current_wave <= self.total_waves:
            if self.wave_timer > 0:
                self.wave_timer -= 1
            else:
                if self.current_wave > 0:
                    reward += 10.0 # Wave clear bonus
                self.current_wave += 1
                if self.current_wave <= self.total_waves:
                    self._spawn_wave()
                    self.wave_timer = 180 # Time between waves
        
        if self.enemies_to_spawn and self.steps % 15 == 0: # Spawn one enemy every 15 steps
            self.enemies.append(self.enemies_to_spawn.pop(0))
        return reward

    def _spawn_wave(self):
        num_enemies = 3 + self.current_wave * 2
        enemy_health = 20 * (1.05 ** (self.current_wave - 1))
        enemy_speed = 1.0 * (1.05 ** (self.current_wave - 1))
        
        for _ in range(num_enemies):
            offset = np.array([self.np_random.uniform(-8, 8), self.np_random.uniform(-8, 8)])
            self.enemies_to_spawn.append(Enemy(enemy_health, enemy_speed, self.path, offset))

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.base_health = 0
        if self.current_wave > self.total_waves and not self.enemies and not self.enemies_to_spawn:
            self.game_over = True # Victory
        if self.steps >= 2500: # Max episode length
            self.game_over = True
        return self.game_over

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Path
        for i in range(len(self.path) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path[i], self.path[i+1], 25)
        
        # Base
        pygame.gfxdraw.filled_trigon(self.screen, self.WIDTH, 145, self.WIDTH, 175, self.WIDTH-15, 160, self.COLOR_BASE)
        
        # Towers
        for tower in self.towers:
            tower.draw(self.screen)
        
        # Enemies
        for enemy in sorted(self.enemies, key=lambda e: e.pos[1]):
            enemy.draw(self.screen)
        
        # Projectiles
        for p in self.projectiles:
            p.draw(self.screen)
            
        # Particles
        for particle in self.particles:
            particle.draw(self.screen)
            
        # Placement cursor
        if not self.game_over:
            cursor_pos = self.placement_grid[self.placement_cursor_idx]
            spec = Tower.TOWER_SPECS[self.selected_tower_type]
            color = spec["color"]
            
            # Range indicator
            pygame.gfxdraw.aacircle(self.screen, cursor_pos[0], cursor_pos[1], spec["range"], (*color, 80))
            
            # Cursor itself
            pygame.draw.rect(self.screen, color, (cursor_pos[0] - self.GRID_SIZE//2, cursor_pos[1] - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE), 2)
    
    def _render_ui(self):
        # Top-left info
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        wave_text = self.font_small.render(f"WAVE: {min(self.current_wave, self.total_waves)}/{self.total_waves}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(wave_text, (10, 30))

        # Base health bar
        health_pct = self.base_health / self.max_base_health
        bar_width = 150
        pygame.draw.rect(self.screen, (50, 50, 50), (self.WIDTH - bar_width - 10, 10, bar_width, 20))
        fill_color = (0, 255, 0) if health_pct > 0.5 else ((255, 255, 0) if health_pct > 0.2 else (255, 0, 0))
        pygame.draw.rect(self.screen, fill_color, (self.WIDTH - bar_width - 10, 10, int(bar_width * health_pct), 20))
        health_text = self.font_small.render("BASE HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (self.WIDTH - bar_width - 10, 32))
        
        # Tower selection UI (bottom center)
        spec = Tower.TOWER_SPECS[self.selected_tower_type]
        box_w, box_h = 220, 50
        box_x, box_y = (self.WIDTH - box_w) // 2, self.HEIGHT - box_h - 10
        
        s = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        s.fill((*self.COLOR_PATH, 180))
        self.screen.blit(s, (box_x, box_y))
        pygame.draw.rect(self.screen, spec["color"], (box_x, box_y, box_w, box_h), 2)

        tower_name = self.font_large.render(spec["name"], True, spec["color"])
        self.screen.blit(tower_name, (box_x + 10, box_y + 12))
        
        cost_text = self.font_small.render(f"Cost: {spec['cost']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(cost_text, (box_x + 150, box_y + 5))
        damage_text = self.font_small.render(f"Dmg: {spec['damage']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(damage_text, (box_x + 150, box_y + 20))
        rate_text = self.font_small.render(f"Rate: {spec['fire_rate']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(rate_text, (box_x + 150, box_y + 35))

        # Game Over / Victory message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            message = "VICTORY!" if self.base_health > 0 else "GAME OVER"
            color = (0, 255, 100) if self.base_health > 0 else (255, 50, 50)
            msg_render = self.font_large.render(message, True, color)
            self.screen.blit(msg_render, (self.WIDTH/2 - msg_render.get_width()/2, self.HEIGHT/2 - msg_render.get_height()/2))
            
    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(2, 5)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos, vel, size, color, lifespan))

    def _create_muzzle_flashes(self, flashes):
        for pos, angle in flashes:
            for _ in range(3):
                speed = self.np_random.uniform(1, 3)
                spread = self.np_random.uniform(-0.4, 0.4)
                vel = [math.cos(angle + spread) * speed, math.sin(angle + spread) * speed]
                size = self.np_random.uniform(1, 3)
                lifespan = self.np_random.integers(5, 10)
                self.particles.append(Particle(pos, vel, (255, 255, 100), size, lifespan))

    def validate_implementation(self):
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

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use Pygame for human interaction
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    done = False
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0)

    while not done:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Map keys to MultiDiscrete action space
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for human playability
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()