
# Generated: 2025-08-27T16:19:34.329589
# Source Brief: brief_01190.md
# Brief Index: 1190

        
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


# --- Helper Classes for Game Objects ---

class Enemy:
    def __init__(self, wave, path, np_random):
        self.path = path
        self.path_index = 0
        self.pos = pygame.Vector2(path[0])
        
        base_health = 20
        base_speed = 0.8
        base_value = 10

        # Scale difficulty by 5% per wave
        difficulty_multiplier = (1.05) ** (wave - 1)
        
        self.max_health = int(base_health * difficulty_multiplier)
        self.health = self.max_health
        self.speed = base_speed * difficulty_multiplier
        self.value = int(base_value * difficulty_multiplier)
        self.radius = 8
        self.color = (220, 50, 50)
        self.pulse_timer = np_random.uniform(0, 2 * math.pi)

    def update(self):
        self.pulse_timer += 0.2
        if self.path_index < len(self.path) - 1:
            target = pygame.Vector2(self.path[self.path_index + 1])
            direction = (target - self.pos).normalize()
            self.pos += direction * self.speed
            if self.pos.distance_to(target) < self.speed:
                self.pos = target
                self.path_index += 1
    
    def draw(self, surface):
        # Pulsing effect
        pulse_size = self.radius + 1.5 * (math.sin(self.pulse_timer) + 1)
        
        # Main body
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(pulse_size), self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(pulse_size), self.color)
        
        # Health bar
        if self.health < self.max_health:
            bar_width = 20
            bar_height = 4
            bar_x = self.pos.x - bar_width / 2
            bar_y = self.pos.y - self.radius - 12
            health_ratio = self.health / self.max_health
            
            pygame.draw.rect(surface, (50, 0, 0), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(surface, (255, 0, 0), (bar_x, bar_y, int(bar_width * health_ratio), bar_height))

class Tower:
    def __init__(self, pos, tower_type):
        self.pos = pygame.Vector2(pos)
        self.type_info = tower_type
        self.cooldown = 0
        self.target = None
        self.angle = -90

    def update(self, enemies, projectiles):
        if self.cooldown > 0:
            self.cooldown -= 1

        # Find new target if current one is gone or out of range
        if self.target and (self.target.health <= 0 or self.pos.distance_to(self.target.pos) > self.type_info['range']):
            self.target = None

        if not self.target:
            closest_enemy = None
            min_dist = self.type_info['range']
            for enemy in enemies:
                dist = self.pos.distance_to(enemy.pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy = enemy
            self.target = closest_enemy
        
        # Aim and fire
        if self.target:
            target_angle = math.degrees(math.atan2(self.target.pos.y - self.pos.y, self.target.pos.x - self.pos.x))
            self.angle = target_angle

            if self.cooldown == 0:
                projectiles.append(Projectile(self.pos, self.target, self.type_info))
                self.cooldown = self.type_info['rate']
                # sfx: tower_fire.wav

    def draw(self, surface):
        # Base
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), 10, self.type_info['color'])
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), 10, (255, 255, 255))

        # Barrel
        barrel_length = 15
        end_x = self.pos.x + barrel_length * math.cos(math.radians(self.angle))
        end_y = self.pos.y + barrel_length * math.sin(math.radians(self.angle))
        pygame.draw.line(surface, (255, 255, 255), (int(self.pos.x), int(self.pos.y)), (int(end_x), int(end_y)), 4)


class Projectile:
    def __init__(self, pos, target, tower_type):
        self.pos = pygame.Vector2(pos)
        self.target = target
        self.speed = tower_type['proj_speed']
        self.damage = tower_type['damage']
        self.color = (255, 255, 0)
        self.radius = 3

    def update(self):
        if self.target and self.target.health > 0:
            direction = (self.target.pos - self.pos).normalize()
            self.pos += direction * self.speed
            # Return True if hit
            if self.pos.distance_to(self.target.pos) < self.target.radius:
                return True
        # Return False if miss or target is gone
        return False

    def draw(self, surface):
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)

class Particle:
    def __init__(self, pos, color, np_random):
        self.pos = pygame.Vector2(pos)
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.color = color
        self.life = 20
        self.size = np_random.uniform(1, 4)

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95
        self.life -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.life > 0:
            pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), int(self.size))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle tower types. Press Space to build a tower."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers. "
        "Survive 10 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 3000 # Increased for 10 waves
    MAX_WAVES = 10
    
    COLOR_BG = (15, 20, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_PATH_BORDER = (60, 70, 80)
    COLOR_BASE = (50, 200, 50)
    COLOR_UI_TEXT = (200, 200, 255)
    COLOR_CURSOR = (0, 255, 255)
    
    TOWER_TYPES = [
        {
            'name': 'Gatling', 'cost': 25, 'range': 100, 'damage': 2, 'rate': 10, 
            'proj_speed': 8, 'color': (150, 150, 150)
        },
        {
            'name': 'Cannon', 'cost': 75, 'range': 150, 'damage': 15, 'rate': 60, 
            'proj_speed': 5, 'color': (100, 100, 120)
        },
    ]

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
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 48)
        
        self._define_level()
        
        self.reset()
        self.validate_implementation()

    def _define_level(self):
        self.path = [
            (-20, 100), (100, 100), (100, 300), 
            (540, 300), (540, 100), (self.SCREEN_WIDTH + 20, 100)
        ]
        self.base_pos = (self.SCREEN_WIDTH - 30, 100)
        self.base_rect = pygame.Rect(self.base_pos[0] - 10, self.base_pos[1] - 20, 20, 40)
        
        self.tower_grid_dims = (10, 5)
        self.tower_spots = []
        for r in range(self.tower_grid_dims[1]):
            row = []
            for c in range(self.tower_grid_dims[0]):
                y = 40 + r * 65
                if r > 2: y += 20
                x = 40 + c * 60
                
                # Exclude spots too close to the path
                on_path = False
                if (80 < y < 120) or (280 < y < 320):
                    on_path = True
                if (80 < x < 120 and 100 < y < 300) or (520 < x < 560 and 100 < y < 300):
                    on_path = True
                
                if not on_path:
                    row.append((x, y))
            self.tower_spots.append(row)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.base_health = 100
        self.max_base_health = 100
        self.money = 80
        
        self.current_wave = 0
        self.wave_cooldown = 150 # Time between waves
        self.spawn_timer = 0
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = deque(maxlen=200)
        
        self.cursor_pos = [0, 0]
        self.selected_tower_type_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_movement_action = 0

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        step_reward = 0
        
        if not self.game_over:
            self.steps += 1
            
            step_reward += self._handle_input(action)
            step_reward += self._update_game_state()

        self.score += step_reward
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
             if self.game_won:
                 step_reward += 100 # Win bonus
             else:
                 step_reward -= 100 # Loss penalty
             self.score += step_reward
             self.game_over = True
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Movement ---
        if movement != self.last_movement_action and movement != 0:
            if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 2: self.cursor_pos[0] = min(self.tower_grid_dims[1] - 1, self.cursor_pos[0] + 1)
            elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 4: self.cursor_pos[1] = min(self.tower_grid_dims[0] - 1, self.cursor_pos[1] + 1)
        self.last_movement_action = movement

        # --- Cycle Tower Type ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.TOWER_TYPES)
            # sfx: ui_cycle.wav

        # --- Place Tower ---
        if space_held and not self.prev_space_held:
            self._place_tower()
            # No reward for placing, cost is implicit penalty

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return 0

    def _place_tower(self):
        row, col = self.cursor_pos
        if not self.tower_spots[row] or col >= len(self.tower_spots[row]):
            return # Invalid spot
        
        pos = self.tower_spots[row][col]
        tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
        
        # Check cost and if spot is occupied
        if self.money >= tower_type['cost']:
            is_occupied = any(t.pos.distance_to(pygame.Vector2(pos)) < 1 for t in self.towers)
            if not is_occupied:
                self.money -= tower_type['cost']
                self.towers.append(Tower(pos, tower_type))
                # sfx: build_tower.wav
    
    def _update_game_state(self):
        reward = 0

        # Update Towers
        for tower in self.towers:
            tower.update(self.enemies, self.projectiles)

        # Update Projectiles and handle hits
        for proj in self.projectiles[:]:
            if proj.update():
                proj.target.health -= proj.damage
                reward += 0.1 # Reward for damaging
                for _ in range(5): self.particles.append(Particle(proj.pos, (255, 200, 0), self.np_random))
                self.projectiles.remove(proj)
                # sfx: enemy_hit.wav
            elif not proj.target or proj.target.health <= 0:
                self.projectiles.remove(proj)

        # Update Enemies
        for enemy in self.enemies[:]:
            enemy.update()
            if enemy.health <= 0:
                reward += 1.0 # Reward for kill
                self.money += enemy.value
                for _ in range(20): self.particles.append(Particle(enemy.pos, enemy.color, self.np_random))
                self.enemies.remove(enemy)
                # sfx: enemy_die.wav
            elif enemy.pos.x >= self.SCREEN_WIDTH:
                damage = 10
                self.base_health -= damage
                reward -= damage * 0.1 # Penalty for base damage
                self.enemies.remove(enemy)
                # sfx: base_damage.wav
        
        # Update Particles
        for p in list(self.particles):
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        # Update Waves
        if not self.enemies and self.enemies_spawned == self.enemies_in_wave:
            if self.current_wave >= self.MAX_WAVES:
                self.game_won = True
            else:
                self.wave_cooldown -= 1
                if self.wave_cooldown <= 0:
                    reward += 10 # Wave clear bonus
                    self._start_next_wave()
        else:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0 and self.enemies_spawned < self.enemies_in_wave:
                self.enemies.append(Enemy(self.current_wave, self.path, self.np_random))
                self.enemies_spawned += 1
                self.spawn_timer = 30 # Time between enemies in a wave

        return reward

    def _start_next_wave(self):
        self.current_wave += 1
        self.enemies_in_wave = 3 + self.current_wave * 2
        self.enemies_spawned = 0
        self.spawn_timer = 0
        self.wave_cooldown = 150

    def _check_termination(self):
        return self.base_health <= 0 or self.game_won or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "money": self.money, "wave": self.current_wave}

    def _render_game(self):
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH_BORDER, False, self.path, 44)
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 40)
        
        # Draw tower placement grid and cursor
        self._render_grid_and_cursor()

        # Draw Base
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), self.base_rect, 2)
        
        # Draw game objects
        for tower in self.towers: tower.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        for particle in self.particles: particle.draw(self.screen)

    def _render_grid_and_cursor(self):
        occupied_spots = {tuple(t.pos) for t in self.towers}
        
        for r_idx, row in enumerate(self.tower_spots):
            for c_idx, pos in enumerate(row):
                is_cursor_pos = (r_idx == self.cursor_pos[0] and c_idx == self.cursor_pos[1])
                is_occupied = pos in occupied_spots
                
                if is_occupied:
                    color = (100, 100, 100)
                elif is_cursor_pos:
                    color = self.COLOR_CURSOR
                else:
                    color = (50, 60, 70)
                
                size = 12 if is_cursor_pos else 8
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
        
        # Draw range indicator for selected tower at cursor
        cursor_row, cursor_col = self.cursor_pos
        if self.tower_spots[cursor_row] and cursor_col < len(self.tower_spots[cursor_row]):
            cursor_world_pos = self.tower_spots[cursor_row][cursor_col]
            selected_tower = self.TOWER_TYPES[self.selected_tower_type_idx]
            
            can_afford = self.money >= selected_tower['cost']
            color = (0, 255, 0, 50) if can_afford else (255, 0, 0, 50)
            
            range_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(range_surface, cursor_world_pos[0], cursor_world_pos[1], selected_tower['range'], color)
            self.screen.blit(range_surface, (0, 0))

    def _render_ui(self):
        # Top bar background
        pygame.draw.rect(self.screen, (0, 0, 0, 150), (0, 0, self.SCREEN_WIDTH, 30))
        
        # Money
        money_text = self.font_small.render(f"$ {self.money}", True, (255, 220, 0))
        self.screen.blit(money_text, (10, 5))
        
        # Base Health
        health_text = self.font_small.render(f"Base: {max(0, self.base_health)} / {self.max_base_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH / 2 - health_text.get_width() / 2, 5))

        # Wave
        wave_text = self.font_small.render(f"Wave: {min(self.current_wave, self.MAX_WAVES)} / {self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 5))
        
        # Selected Tower Info (Bottom-left)
        tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
        can_afford = self.money >= tower_type['cost']
        cost_color = (0, 255, 0) if can_afford else (255, 50, 50)

        info_bg_rect = pygame.Rect(5, self.SCREEN_HEIGHT - 65, 150, 60)
        pygame.draw.rect(self.screen, (0, 0, 0, 150), info_bg_rect, border_radius=5)
        
        name_surf = self.font_small.render(f"Build: {tower_type['name']}", True, self.COLOR_UI_TEXT)
        cost_surf = self.font_small.render(f"Cost: ${tower_type['cost']}", True, cost_color)
        self.screen.blit(name_surf, (15, self.SCREEN_HEIGHT - 60))
        self.screen.blit(cost_surf, (15, self.SCREEN_HEIGHT - 40))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (50, 255, 50) if self.game_won else (255, 50, 50)
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    action = env.action_space.sample() # Start with a no-op
    action.fill(0)

    while not terminated:
        movement, space, shift = 0, 0, 0
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
            
        action = np.array([movement, space, shift])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Money: {info['money']}")

    env.close()