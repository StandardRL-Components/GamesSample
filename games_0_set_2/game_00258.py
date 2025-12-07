
# Generated: 2025-08-27T13:05:48.276859
# Source Brief: brief_00258.md
# Brief Index: 258

        
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
    def __init__(self, health, speed, value, path):
        self.pos = pygame.Vector2(path[0])
        self.health = health
        self.max_health = health
        self.speed = speed
        self.value = value
        self.path = path
        self.path_index = 0
        self.target_pos = pygame.Vector2(path[1]) if len(path) > 1 else self.pos
        self.is_alive = True

    def move(self):
        if self.path_index >= len(self.path) - 1:
            return True # Reached the end

        direction = self.target_pos - self.pos
        distance = direction.length()

        if distance < self.speed:
            self.pos = self.target_pos
            self.path_index += 1
            if self.path_index < len(self.path) - 1:
                self.target_pos = pygame.Vector2(self.path[self.path_index + 1])
        else:
            self.pos += direction.normalize() * self.speed
        
        return False

    def draw(self, surface):
        # Body
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), 8, (200, 0, 0))
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), 8, (255, 50, 50))
        
        # Health bar
        if self.health < self.max_health:
            bar_width = 16
            bar_height = 3
            health_ratio = self.health / self.max_health
            current_health_width = int(bar_width * health_ratio)
            
            bg_rect = pygame.Rect(self.pos.x - bar_width / 2, self.pos.y - 15, bar_width, bar_height)
            health_rect = pygame.Rect(self.pos.x - bar_width / 2, self.pos.y - 15, current_health_width, bar_height)
            
            pygame.draw.rect(surface, (100, 0, 0), bg_rect)
            pygame.draw.rect(surface, (0, 200, 0), health_rect)


class Tower:
    def __init__(self, pos, tower_type_info):
        self.pos = pos
        self.type_info = tower_type_info
        self.cooldown = 0
        self.target = None

    def update(self, dt):
        self.cooldown = max(0, self.cooldown - dt)

    def can_fire(self):
        return self.cooldown == 0

    def find_target(self, enemies):
        closest_enemy = None
        min_dist = self.type_info['range'] ** 2 # Use squared distance for efficiency
        
        for enemy in enemies:
            dist_sq = self.pos.distance_squared_to(enemy.pos)
            if dist_sq < min_dist:
                min_dist = dist_sq
                closest_enemy = enemy
        self.target = closest_enemy

    def draw(self, surface):
        color = self.type_info['color']
        size = 18
        rect = pygame.Rect(self.pos.x - size, self.pos.y - size, size * 2, size * 2)
        pygame.draw.rect(surface, color, rect, border_radius=4)
        pygame.draw.rect(surface, tuple(min(255, c+50) for c in color), rect, width=2, border_radius=4)


class Projectile:
    def __init__(self, start_pos, target, damage, speed):
        self.pos = pygame.Vector2(start_pos)
        self.target = target
        self.damage = damage
        self.speed = speed
        self.active = True

    def move(self):
        if not self.target or not self.target.is_alive:
            self.active = False
            return
        
        direction = self.target.pos - self.pos
        distance = direction.length()

        if distance < self.speed:
            self.pos = self.target.pos
            self.active = False # Hit
        else:
            self.pos += direction.normalize() * self.speed
    
    def draw(self, surface):
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), 3, (255, 255, 0))
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), 3, (255, 255, 150))


class Particle:
    def __init__(self, pos, color, start_radius, end_radius, lifespan):
        self.pos = pygame.Vector2(pos)
        self.color = color
        self.radius = start_radius
        self.max_radius = end_radius
        self.lifespan = lifespan
        self.life = lifespan

    def update(self):
        self.life -= 1
        progress = (self.lifespan - self.life) / self.lifespan
        self.radius = self.max_radius * math.sin(progress * math.pi / 2) # Ease out
        return self.life <= 0

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.lifespan))
            temp_surface = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surface, int(self.radius), int(self.radius), int(self.radius), (*self.color, alpha))
            surface.blit(temp_surface, (int(self.pos.x - self.radius), int(self.pos.y - self.radius)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to move the cursor. Press Shift to cycle tower types. Press Space to build a tower."
    )
    game_description = (
        "A top-down tower defense game. Place towers to defend your base from waves of incoming enemies."
    )
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 40
    GRID_W, GRID_H = SCREEN_WIDTH // GRID_SIZE, SCREEN_HEIGHT // GRID_SIZE
    
    COLOR_BG = (25, 25, 40)
    COLOR_PATH = (50, 50, 70)
    COLOR_GRID = (40, 40, 60)
    COLOR_BASE = (0, 150, 50)
    COLOR_TEXT = (220, 220, 240)
    
    MAX_STEPS = 30 * 180 # 3 minutes at 30fps
    MAX_WAVES = 10
    INITIAL_RESOURCES = 150
    INITIAL_BASE_HEALTH = 100
    WAVE_GRACE_PERIOD = 5 * 30 # 5 seconds

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
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.tower_types = {
            0: {"name": "Gatling", "cost": 50, "range": 100, "damage": 2, "fire_rate": 8, "color": (0, 150, 255)},
            1: {"name": "Cannon", "cost": 125, "range": 140, "damage": 15, "fire_rate": 1.5, "color": (255, 150, 0)},
        }
        
        self._define_path_and_grid()
        self.reset()
        
    def _define_path_and_grid(self):
        self.path_waypoints = [
            (-20, 220), (100, 220), (100, 100), (460, 100), (460, 300), (260, 300), (260, 180), (580, 180), (580, 420)
        ]
        self.base_pos = pygame.Vector2(580, 380) # End of path is near base
        
        self.buildable_grid = np.ones((self.GRID_W, self.GRID_H), dtype=bool)
        path_width = self.GRID_SIZE * 1.5
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                cell_center = pygame.Vector2(x * self.GRID_SIZE + self.GRID_SIZE / 2, y * self.GRID_SIZE + self.GRID_SIZE / 2)
                for i in range(len(self.path_waypoints) - 1):
                    p1 = pygame.Vector2(self.path_waypoints[i])
                    p2 = pygame.Vector2(self.path_waypoints[i+1])
                    # Check distance from line segment
                    l2 = p1.distance_squared_to(p2)
                    if l2 == 0.0:
                        if cell_center.distance_squared_to(p1) < path_width**2:
                           self.buildable_grid[x, y] = False
                           break
                    t = max(0, min(1, (cell_center - p1).dot(p2 - p1) / l2))
                    projection = p1 + t * (p2 - p1)
                    if cell_center.distance_squared_to(projection) < path_width**2:
                        self.buildable_grid[x, y] = False
                        break
                if self.base_pos.distance_to(cell_center) < self.GRID_SIZE * 1.5:
                    self.buildable_grid[x, y] = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0

        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.current_wave = 0
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.wave_spawning = False
        self.wave_complete = True
        self.wave_timer = 0
        self.enemies_to_spawn = []

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            return 
        
        self.wave_spawning = True
        self.wave_complete = False
        self.wave_timer = 0
        self.enemies_to_spawn.clear()
        
        num_enemies = 5 + self.current_wave * 2
        health = 20 * (1.1 ** (self.current_wave - 1))
        speed = 1.0 * (1.05 ** (self.current_wave - 1))
        value = 5 + self.current_wave

        for _ in range(num_enemies):
            self.enemies_to_spawn.append({"health": health, "speed": speed, "value": value})
        
        # Stagger spawns
        self.spawn_interval = max(10, 45 - self.current_wave * 2)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = 0
        
        self._handle_input(action)
        self._update_wave_spawner()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        if not self.wave_spawning and not self.enemies and not self.wave_complete:
            self.wave_complete = True
            self.wave_timer = self.WAVE_GRACE_PERIOD
            self.reward_this_step += 5 # Wave clear bonus

        if self.wave_complete and self.wave_timer > 0:
            self.wave_timer -= 1
            if self.wave_timer == 0 and self.current_wave < self.MAX_WAVES:
                self._start_next_wave()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.base_health <= 0:
                self.reward_this_step -= 100
            elif self.current_wave > self.MAX_WAVES:
                self.reward_this_step += 100
            self.game_over = True

        self.score += self.reward_this_step
        
        return self._get_observation(), self.reward_this_step, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_H - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_W - 1, self.cursor_pos[0] + 1)
        
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_types)
        
        if space_held and not self.prev_space_held:
            self._place_tower()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _place_tower(self):
        grid_x, grid_y = self.cursor_pos
        tower_info = self.tower_types[self.selected_tower_type]
        
        if self.resources >= tower_info['cost'] and self.buildable_grid[grid_x, grid_y]:
            # Check if a tower already exists here
            is_occupied = False
            for tower in self.towers:
                if tower.pos.x // self.GRID_SIZE == grid_x and tower.pos.y // self.GRID_SIZE == grid_y:
                    is_occupied = True
                    break
            
            if not is_occupied:
                self.resources -= tower_info['cost']
                pos = pygame.Vector2(grid_x * self.GRID_SIZE + self.GRID_SIZE / 2, grid_y * self.GRID_SIZE + self.GRID_SIZE / 2)
                self.towers.append(Tower(pos, tower_info))
                # sfx: build_tower.wav
                self.particles.append(Particle(pos, (200, 200, 255), 5, 25, 15))


    def _update_wave_spawner(self):
        if self.wave_spawning:
            self.wave_timer += 1
            if self.wave_timer >= self.spawn_interval and self.enemies_to_spawn:
                self.wave_timer = 0
                enemy_data = self.enemies_to_spawn.pop(0)
                self.enemies.append(Enemy(enemy_data['health'], enemy_data['speed'], enemy_data['value'], self.path_waypoints))
            
            if not self.enemies_to_spawn:
                self.wave_spawning = False

    def _update_towers(self):
        dt = 1.0 # Assuming step is one frame
        for tower in self.towers:
            tower.update(dt)
            if tower.can_fire():
                tower.find_target(self.enemies)
                if tower.target:
                    # sfx: tower_shoot.wav
                    self.projectiles.append(Projectile(tower.pos, tower.target, tower.type_info['damage'], 10))
                    tower.cooldown = 30 / tower.type_info['fire_rate']

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj.move()
            if not proj.active:
                if proj.target and proj.target.is_alive and proj.pos.distance_to(proj.target.pos) < proj.speed:
                    # Hit!
                    proj.target.health -= proj.damage
                    self.reward_this_step += 0.1
                    # sfx: enemy_hit.wav
                    self.particles.append(Particle(proj.pos, (255, 200, 0), 2, 8, 10))
                    if proj.target.health <= 0:
                        proj.target.is_alive = False
                self.projectiles.remove(proj)

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if not enemy.is_alive:
                self.reward_this_step += 1.0
                self.resources += enemy.value
                self.particles.append(Particle(enemy.pos, (255, 50, 50), 5, 20, 20))
                self.enemies.remove(enemy)
                # sfx: enemy_explode.wav
                continue

            if enemy.move():
                self.base_health -= 10
                self.base_health = max(0, self.base_health)
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
                self.particles.append(Particle(self.base_pos, (255, 0, 0), 10, 40, 30))

    def _update_particles(self):
        for p in self.particles[:]:
            if p.update():
                self.particles.remove(p)

    def _check_termination(self):
        return self.base_health <= 0 or self.current_wave > self.MAX_WAVES or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "resources": self.resources, "base_health": self.base_health}
    
    def _draw_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw path
        if len(self.path_waypoints) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, width=self.GRID_SIZE)
        
        # Draw base
        base_rect = pygame.Rect(self.base_pos.x - 20, self.base_pos.y - 20, 40, 40)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        
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
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_world_x = cursor_x * self.GRID_SIZE
        cursor_world_y = cursor_y * self.GRID_SIZE
        
        can_build = self.buildable_grid[cursor_x, cursor_y] and self.resources >= self.tower_types[self.selected_tower_type]['cost']
        cursor_color = (0, 255, 0, 100) if can_build else (255, 0, 0, 100)
        
        # Draw range indicator
        range_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        tower_range = self.tower_types[self.selected_tower_type]['range']
        pygame.gfxdraw.filled_circle(range_surf, cursor_world_x + self.GRID_SIZE//2, cursor_world_y + self.GRID_SIZE//2, tower_range, (255, 255, 255, 30))
        pygame.gfxdraw.aacircle(range_surf, cursor_world_x + self.GRID_SIZE//2, cursor_world_y + self.GRID_SIZE//2, tower_range, (255, 255, 255, 80))
        self.screen.blit(range_surf, (0,0))
        
        # Draw cursor box
        cursor_rect = pygame.Rect(cursor_world_x, cursor_world_y, self.GRID_SIZE, self.GRID_SIZE)
        s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        s.fill(cursor_color)
        self.screen.blit(s, (cursor_world_x, cursor_world_y))


    def _render_ui(self):
        # Top UI Panel
        panel_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 35)
        pygame.draw.rect(self.screen, (15, 15, 25), panel_rect)
        pygame.draw.line(self.screen, (80, 80, 100), (0, 35), (self.SCREEN_WIDTH, 35))

        # Wave
        self._draw_text(f"Wave: {self.current_wave}/{self.MAX_WAVES}", (10, 8), self.font_large, self.COLOR_TEXT)

        # Base Health
        health_text = f"Base Health: {self.base_health}"
        text_w = self.font_large.size(health_text)[0]
        self._draw_text(health_text, (self.SCREEN_WIDTH - text_w - 10, 8), self.font_large, self.COLOR_TEXT)
        
        # Bottom UI Panel
        panel_rect_bottom = pygame.Rect(0, self.SCREEN_HEIGHT - 35, self.SCREEN_WIDTH, 35)
        pygame.draw.rect(self.screen, (15, 15, 25), panel_rect_bottom)
        pygame.draw.line(self.screen, (80, 80, 100), (0, self.SCREEN_HEIGHT - 35), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 35))

        # Resources
        self._draw_text(f"Resources: ${self.resources}", (10, self.SCREEN_HEIGHT - 27), self.font_large, self.COLOR_TEXT)

        # Selected Tower
        tower_info = self.tower_types[self.selected_tower_type]
        tower_text = f"Selected: {tower_info['name']} (Cost: ${tower_info['cost']})"
        text_w = self.font_large.size(tower_text)[0]
        self._draw_text(tower_text, (self.SCREEN_WIDTH - text_w - 10, self.SCREEN_HEIGHT - 27), self.font_large, self.COLOR_TEXT)
        
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            msg = "VICTORY!" if self.base_health > 0 else "GAME OVER"
            self._draw_text(msg, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20), self.font_large, (255,255,0), center=True)
            self._draw_text(f"Final Score: {self.score:.1f}", (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20), self.font_large, self.COLOR_TEXT, center=True)

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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Gymnasium environments are not designed for direct human play without a wrapper.
    # This is a basic interactive loop for testing and visualization.
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a real screen for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Human Controls to Action Space Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        # --- End of Human Controls ---

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()