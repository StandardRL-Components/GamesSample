
# Generated: 2025-08-27T13:38:45.385850
# Source Brief: brief_00438.md
# Brief Index: 438

        
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


class GameEnv(gym.Env):
    """
    A tower defense game where the player places towers to defend a base against waves of enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Arrow keys to move cursor, Space to place tower, Shift to cycle tower type."
    )
    game_description = (
        "Defend your base from waves of invading enemies by strategically placing towers."
    )

    # Frame advance setting
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 15000  # Approx 8.3 minutes at 30fps
        self.TOTAL_WAVES = 10

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_GRID = (25, 30, 40)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_CANT_AFFORD = (255, 50, 50)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game Objects & Path ---
        self.path = [
            (-20, 200), (100, 200), (100, 100), (300, 100), (300, 300),
            (500, 300), (500, 50), (self.WIDTH + 20, 50)
        ]
        self.placement_spots = self._generate_placement_spots()
        self.base_pos = (self.WIDTH - 40, 50)
        self.base_rect = pygame.Rect(self.WIDTH - 40, 30, 20, 40)
        
        # --- Tower Definitions ---
        self.TOWER_TYPES = {
            0: {"name": "Cannon", "cost": 100, "range": 80, "damage": 25, "fire_rate": 1.0, "color": (0, 150, 255)},
            1: {"name": "Rapid", "cost": 150, "range": 60, "damage": 10, "fire_rate": 3.0, "color": (255, 100, 0)},
        }
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.max_base_health = 0
        self.gold = 0
        self.wave_number = 0
        self.wave_timer = 0
        self.enemies_in_wave = 0
        self.enemies_spawned_in_wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_index = 0
        self.selected_tower_type = 0
        self.last_shift_state = 0
        self.last_space_state = 0
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_base_health = 100
        self.base_health = self.max_base_health
        self.gold = 250
        
        self.wave_number = 0
        self.wave_timer = 5 * self.FPS # 5 second delay before first wave
        self.enemies_in_wave = 0
        self.enemies_spawned_in_wave = 0
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_index = 0
        self.selected_tower_type = 0
        self.last_shift_state = 0
        self.last_space_state = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = -0.01  # Small penalty for time passing

        # --- Handle Input ---
        input_reward = self._handle_input(action)
        step_reward += input_reward

        # --- Game Logic Updates ---
        self._update_waves()
        
        # Update towers and get rewards from firing
        fire_reward = self._update_towers()
        step_reward += fire_reward
        
        # Update projectiles and get rewards from hits
        hit_reward, kill_reward = self._update_projectiles()
        step_reward += hit_reward + kill_reward
        
        # Update enemies and check for base damage
        base_damage_reward = self._update_enemies()
        step_reward += base_damage_reward
        
        self._update_particles()
        
        # --- Check for wave clear ---
        if self.enemies_spawned_in_wave == self.enemies_in_wave and not self.enemies:
            if self.wave_number > 0 and self.wave_timer < 0: # Wave was active
                step_reward += 10.0 # Wave clear bonus
                self.score += 10
                self.wave_timer = 10 * self.FPS # 10s until next wave
                if self.wave_number == self.TOTAL_WAVES:
                    self.game_over = True
                    step_reward += 100.0 # Victory bonus
                    self.score += 100

        # --- Check Termination Conditions ---
        terminated = False
        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
            terminated = True
            step_reward -= 100.0 # Loss penalty
            self.score -= 100
        elif self.wave_number == self.TOTAL_WAVES and not self.enemies and self.enemies_spawned_in_wave > 0:
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            
        self.score += step_reward

        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        if movement != 0:
            rows = len(self.placement_spots) // 5 # Assuming 5 columns
            cols = 5
            row = self.cursor_index // cols
            col = self.cursor_index % cols
            if movement == 1 and row > 0: self.cursor_index -= cols # Up
            elif movement == 2 and row < rows - 1: self.cursor_index += cols # Down
            elif movement == 3 and col > 0: self.cursor_index -= 1 # Left
            elif movement == 4 and col < cols - 1: self.cursor_index += 1 # Right
        
        # --- Cycle Tower Type ---
        if shift_pressed and not self.last_shift_state:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
        self.last_shift_state = shift_pressed

        # --- Place Tower ---
        if space_pressed and not self.last_space_state:
            spot = self.placement_spots[self.cursor_index]
            tower_def = self.TOWER_TYPES[self.selected_tower_type]
            if not spot['occupied'] and self.gold >= tower_def['cost']:
                self.gold -= tower_def['cost']
                spot['occupied'] = True
                new_tower = self.Tower(spot['pos'], tower_def)
                self.towers.append(new_tower)
                # Sound: PlaceTower.wav
        self.last_space_state = space_pressed
        
        return 0.0 # Input itself doesn't grant reward

    def _update_waves(self):
        self.wave_timer -= 1
        if self.wave_timer <= 0 and self.wave_number < self.TOTAL_WAVES:
            if self.enemies_spawned_in_wave == self.enemies_in_wave: # Start new wave
                self.wave_number += 1
                self.enemies_spawned_in_wave = 0
                difficulty_mod = 1 + (self.wave_number - 1) * 0.1
                self.enemies_in_wave = 5 + self.wave_number * 2
                enemy_health = int(50 * difficulty_mod)
                enemy_speed = 1.0 + 0.1 * (self.wave_number - 1)
                self.wave_timer = int(1.0 * self.FPS) # Time between spawns
                
            if self.enemies_spawned_in_wave < self.enemies_in_wave:
                # Spawn an enemy
                self.enemies_spawned_in_wave += 1
                difficulty_mod = 1 + (self.wave_number - 1) * 0.1
                health = int(50 * difficulty_mod)
                speed = 1.0 + 0.1 * (self.wave_number - 1)
                new_enemy = self.Enemy(self.path, health, speed)
                self.enemies.append(new_enemy)
                self.wave_timer = int(max(0.2, 1.0 - self.wave_number * 0.05) * self.FPS)
                # Sound: EnemySpawn.wav

    def _update_enemies(self):
        base_damage_reward = 0.0
        for enemy in reversed(self.enemies):
            reached_end = enemy.update()
            if reached_end:
                self.base_health -= enemy.health # Damage proportional to remaining health
                base_damage_reward -= 10.0 # Penalty for leak
                self.enemies.remove(enemy)
                # Sound: BaseDamage.wav
            elif enemy.health <= 0:
                self.enemies.remove(enemy)
        return base_damage_reward

    def _update_towers(self):
        for tower in self.towers:
            projectile = tower.update(self.enemies)
            if projectile:
                self.projectiles.append(projectile)
                # Sound: TowerFire.wav
        return 0.0 # No reward for just firing

    def _update_projectiles(self):
        hit_reward = 0.0
        kill_reward = 0.0
        for proj in reversed(self.projectiles):
            if proj.update():
                self.projectiles.remove(proj)
                continue
            
            for enemy in self.enemies:
                if proj.check_collision(enemy):
                    damage_dealt = enemy.take_damage(proj.damage)
                    hit_reward += 0.1
                    
                    if enemy.health <= 0:
                        kill_reward += 1.0
                        self.gold += 10 + self.wave_number
                        # Sound: EnemyDie.wav
                    
                    # Create impact particles
                    for _ in range(5):
                        self.particles.append(self.Particle(proj.pos, (255, 200, 0), self.rng))
                    
                    self.projectiles.remove(proj)
                    # Sound: ProjectileHit.wav
                    break
        return hit_reward, kill_reward

    def _update_particles(self):
        for particle in reversed(self.particles):
            if not particle.update():
                self.particles.remove(particle)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 30)

        # Draw placement spots
        for spot in self.placement_spots:
            if not spot['occupied']:
                pygame.draw.rect(self.screen, self.COLOR_GRID, spot['rect'], 1)

        # Draw cursor
        cursor_spot = self.placement_spots[self.cursor_index]
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        can_afford = self.gold >= tower_def['cost']
        cursor_color = self.COLOR_CURSOR if can_afford else self.COLOR_CANT_AFFORD
        pygame.draw.rect(self.screen, cursor_color, cursor_spot['rect'], 2)
        
        # Draw tower range indicator
        if not cursor_spot['occupied']:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(s, cursor_color + (50,), cursor_spot['pos'], tower_def['range'])
            self.screen.blit(s, (0, 0))

        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect)

        # Draw towers
        for tower in self.towers:
            tower.draw(self.screen)

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)

        # Draw projectiles
        for proj in self.projectiles:
            proj.draw(self.screen)

        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)

    def _render_ui(self):
        # --- Top Bar ---
        gold_text = self.font_small.render(f"Gold: {self.gold}", True, self.COLOR_TEXT)
        self.screen.blit(gold_text, (self.WIDTH - 100, 10))
        
        wave_str = f"Wave: {self.wave_number}/{self.TOTAL_WAVES}"
        if self.wave_timer > 0 and self.wave_number < self.TOTAL_WAVES and not (self.enemies_in_wave > 0 and self.enemies):
             wave_str += f" (Next in {self.wave_timer // self.FPS + 1}s)"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # --- Base Health Bar ---
        health_ratio = self.base_health / self.max_base_health
        bar_width = 150
        bar_height = 20
        bar_x, bar_y = 10, self.HEIGHT - 30
        pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
        health_val_text = self.font_small.render(f"{self.base_health}/{self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_val_text, (bar_x + 5, bar_y + 2))

        # --- Selected Tower Info ---
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        can_afford = self.gold >= tower_def['cost']
        text_color = self.COLOR_TEXT if can_afford else self.COLOR_CANT_AFFORD
        
        tower_name = self.font_small.render(f"Tower: {tower_def['name']}", True, text_color)
        tower_cost = self.font_small.render(f"Cost: {tower_def['cost']}", True, text_color)
        self.screen.blit(tower_name, (self.WIDTH - 150, self.HEIGHT - 50))
        self.screen.blit(tower_cost, (self.WIDTH - 150, self.HEIGHT - 30))

        # --- Game Over/Win Message ---
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            message = "VICTORY!" if self.base_health > 0 else "GAME OVER"
            color = self.COLOR_BASE if self.base_health > 0 else self.COLOR_CANT_AFFORD
            msg_text = self.font_large.render(message, True, color)
            text_rect = msg_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }
        
    def _generate_placement_spots(self):
        spots = []
        for y in range(50, self.HEIGHT - 50, 60):
            for x in range(150, self.WIDTH - 150, 60):
                pos = (x, y)
                if not self._is_on_path(pos, 35):
                    spots.append({'pos': pos, 'rect': pygame.Rect(x-15, y-15, 30, 30), 'occupied': False})
        return spots
    
    def _is_on_path(self, pos, tolerance):
        px, py = pos
        for i in range(len(self.path) - 1):
            p1 = self.path[i]
            p2 = self.path[i+1]
            x1, y1 = p1
            x2, y2 = p2
            
            # Bounding box check
            if not (min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance and \
                    min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance):
                continue
            
            # Line distance check
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0: continue
            
            t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
            t = max(0, min(1, t))
            closest_x, closest_y = x1 + t * dx, y1 + t * dy
            
            dist_sq = (px - closest_x)**2 + (py - closest_y)**2
            if dist_sq < tolerance**2:
                return True
        return False

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    class Enemy:
        def __init__(self, path, health, speed):
            self.path = path
            self.path_index = 0
            self.pos = pygame.Vector2(path[0])
            self.max_health = health
            self.health = health
            self.speed = speed
            self.radius = 10
            self.color = (220, 50, 50)

        def update(self):
            if self.path_index >= len(self.path) - 1:
                return True # Reached end

            target = pygame.Vector2(self.path[self.path_index + 1])
            direction = (target - self.pos)
            
            if direction.length_squared() == 0:
                self.path_index += 1
                return self.update()

            direction.normalize_ip()
            self.pos += direction * self.speed
            
            if (self.pos - target).length_squared() < self.speed**2:
                self.pos = target
                self.path_index += 1
            return False

        def draw(self, surface):
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
            # Health bar
            if self.health < self.max_health:
                bar_w = 20
                bar_h = 4
                bar_x = self.pos.x - bar_w / 2
                bar_y = self.pos.y - self.radius - 8
                health_ratio = self.health / self.max_health
                pygame.draw.rect(surface, (80,0,0), (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(surface, (0,200,0), (bar_x, bar_y, int(bar_w * health_ratio), bar_h))
        
        def take_damage(self, amount):
            dealt = min(self.health, amount)
            self.health -= dealt
            return dealt

    class Tower:
        def __init__(self, pos, type_def):
            self.pos = pygame.Vector2(pos)
            self.type = type_def
            self.cooldown = 0
            self.fire_rate_frames = int(30 / type_def['fire_rate'])
            self.target = None

        def update(self, enemies):
            if self.cooldown > 0:
                self.cooldown -= 1
            
            # Find new target if current is invalid
            if self.target and (self.target.health <= 0 or (self.pos - self.target.pos).length_squared() > self.type['range']**2):
                self.target = None
            
            if not self.target:
                for enemy in enemies:
                    if (self.pos - enemy.pos).length_squared() <= self.type['range']**2:
                        self.target = enemy
                        break
            
            # Fire if possible
            if self.target and self.cooldown <= 0:
                self.cooldown = self.fire_rate_frames
                return GameEnv.Projectile(self.pos, self.target, self.type['damage'])
            return None

        def draw(self, surface):
            color = self.type['color']
            rect = pygame.Rect(self.pos.x - 10, self.pos.y - 10, 20, 20)
            pygame.draw.rect(surface, color, rect)
            if self.cooldown > self.fire_rate_frames - 3: # Flash on fire
                pygame.draw.rect(surface, (255, 255, 255), rect.inflate(4, 4), 2)

    class Projectile:
        def __init__(self, start_pos, target, damage):
            self.pos = pygame.Vector2(start_pos)
            self.target = target
            self.damage = damage
            self.speed = 8.0
            self.color = (255, 255, 0)
            self.radius = 3

        def update(self):
            if self.target.health <= 0:
                return True # Target is dead, projectile fizzles
            
            direction = (self.target.pos - self.pos)
            if direction.length_squared() < self.speed**2:
                return False # Close enough to hit next frame
            
            direction.normalize_ip()
            self.pos += direction * self.speed
            return False

        def check_collision(self, enemy):
            return (self.pos - enemy.pos).length_squared() < (self.radius + enemy.radius)**2

        def draw(self, surface):
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)

    class Particle:
        def __init__(self, pos, color, rng):
            self.pos = pygame.Vector2(pos)
            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(1, 3)
            self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.lifespan = rng.integers(10, 20)
            self.color = color
            self.radius = rng.uniform(1, 4)

        def update(self):
            self.pos += self.vel
            self.lifespan -= 1
            self.vel *= 0.9 # Friction
            return self.lifespan > 0

        def draw(self, surface):
            alpha = int(255 * (self.lifespan / 20))
            color_with_alpha = self.color + (alpha,)
            s = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color_with_alpha, (self.radius, self.radius), self.radius)
            surface.blit(s, (self.pos.x - self.radius, self.pos.y - self.radius))

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It's a simple example of how to use the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    # --- Action mapping from keyboard to MultiDiscrete ---
    # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    # actions[1]: Space button (0=released, 1=held)
    # actions[2]: Shift button (0=released, 1=held)
    
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0.0

        clock.tick(env.FPS)
        
    env.close()