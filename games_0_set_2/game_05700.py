
# Generated: 2025-08-28T05:48:20.339667
# Source Brief: brief_05700.md
# Brief Index: 5700

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place selected tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Isometric tower defense. Place towers to defend your base from waves of enemies."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 12
    TILE_WIDTH, TILE_HEIGHT = 48, 24
    MAX_STEPS = 5000  # Increased for longer games
    MAX_WAVES = 15

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_PATH = (60, 55, 70)
    COLOR_BASE = (0, 150, 255)
    COLOR_BASE_DMG = (255, 80, 80)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_GOLD = (255, 215, 0)
    
    CURSOR_VALID = (50, 255, 50, 150)
    CURSOR_INVALID = (255, 50, 50, 150)

    # --- Tower Definitions ---
    TOWER_SPECS = {
        "Gun": {
            "cost": 50, "range": 90, "damage": 5, "fire_rate": 10, "color": (0, 255, 150), "proj_speed": 8,
        },
        "Cannon": {
            "cost": 120, "range": 120, "damage": 25, "fire_rate": 45, "color": (255, 150, 0), "proj_speed": 6,
        },
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)
        
        # Isometric projection setup
        self.origin_x = self.SCREEN_WIDTH / 2
        self.origin_y = 80

        # Define the enemy path (grid coordinates)
        self.path_coords = self._define_path()
        self.path_pixels = [self._iso_to_screen(x, y) for x, y in self.path_coords]
        self.base_pos_grid = self.path_coords[-1]
        
        # Initialize state variables
        self.tower_types = list(self.TOWER_SPECS.keys())
        self.reset()

        # Final validation
        self.validate_implementation()
    
    def _define_path(self):
        # A winding path across the grid
        path = []
        for i in range(5): path.append((i, 2))
        for i in range(3, 8): path.append((4, i))
        for i in range(5, 10): path.append((i, 7))
        for i in range(6, 4, -1): path.append((9, i))
        for i in range(8, 2, -1): path.append((i, 5))
        path.append((3,5))
        path.append((3,4))
        return path

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.origin_x + (grid_x - grid_y) * self.TILE_WIDTH / 2
        screen_y = self.origin_y + (grid_x + grid_y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        
        self.base_health = 100
        self.gold = 150
        self.wave_number = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type_idx = 0
        
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Update Game Logic ---
        reward += self._update_towers()
        self._update_projectiles()
        wave_cleared, base_damage = self._update_enemies()
        self._update_particles()
        
        self.gold = max(0, self.gold)
        reward -= base_damage * 0.01

        if wave_cleared:
            reward += 1.0
            self._start_next_wave()

        # --- Termination ---
        self.steps += 1
        terminated = False
        if self.base_health <= 0:
            reward = -100
            terminated = True
            self.game_over_message = "Base Destroyed!"
        elif self.wave_number > self.MAX_WAVES:
            reward = 100
            terminated = True
            self.game_over_message = "You Win!"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over_message = "Time Limit Reached"

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # Cycle tower type (on press)
        if shift_held and not self.shift_pressed_last_frame:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.tower_types)
            # sfx: UI_switch
        self.shift_pressed_last_frame = shift_held

        # Place tower (on press)
        if space_held and not self.space_pressed_last_frame:
            self._place_tower()
        self.space_pressed_last_frame = space_held

    def _place_tower(self):
        x, y = self.cursor_pos
        tower_type_name = self.tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type_name]
        
        is_on_path = tuple(self.cursor_pos) in self.path_coords
        is_occupied = any(t['pos_grid'] == self.cursor_pos for t in self.towers)
        can_afford = self.gold >= spec['cost']

        if not is_on_path and not is_occupied and can_afford:
            self.gold -= spec['cost']
            screen_pos = self._iso_to_screen(x, y)
            self.towers.append({
                "pos_grid": [x, y],
                "pos_screen": screen_pos,
                "type": tower_type_name,
                "spec": spec,
                "cooldown": 0,
            })
            # sfx: build_tower

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return
            
        num_enemies = 2 + self.wave_number
        speed = 1.0 + (self.wave_number - 1) * 0.1
        health = 20 + (self.wave_number - 1) * 10
        
        for i in range(num_enemies):
            self.enemies.append({
                "pos": list(self.path_pixels[0]),
                "health": health,
                "max_health": health,
                "speed": speed,
                "path_idx": 0,
                "spawn_delay": i * 20, # Stagger spawn
                "id": self.steps + i
            })

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            target = None
            min_dist = float('inf')
            # Find closest enemy in range
            for enemy in self.enemies:
                if enemy['spawn_delay'] > 0: continue
                dist = math.hypot(tower['pos_screen'][0] - enemy['pos'][0], tower['pos_screen'][1] - enemy['pos'][1])
                if dist <= tower['spec']['range'] and dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower['cooldown'] = tower['spec']['fire_rate']
                # sfx: tower_fire
                self.projectiles.append({
                    "pos": list(tower['pos_screen']),
                    "target_pos": list(target['pos']),
                    "speed": tower['spec']['proj_speed'],
                    "damage": tower['spec']['damage'],
                    "color": tower['spec']['color']
                })
        return reward

    def _update_projectiles(self):
        for proj in self.projectiles.copy():
            target_x, target_y = proj['target_pos']
            dx = target_x - proj['pos'][0]
            dy = target_y - proj['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < proj['speed']:
                proj['pos'] = proj['target_pos']
            else:
                proj['pos'][0] += (dx / dist) * proj['speed']
                proj['pos'][1] += (dy / dist) * proj['speed']
            
            # Check for collision with any enemy (not just original target)
            hit = False
            for enemy in self.enemies:
                if enemy['spawn_delay'] > 0: continue
                e_dist = math.hypot(proj['pos'][0] - enemy['pos'][0], proj['pos'][1] - enemy['pos'][1])
                if e_dist < 8: # Hit radius
                    enemy['health'] -= proj['damage']
                    self._create_particles(proj['pos'], proj['color'], 5)
                    # sfx: projectile_hit
                    self.projectiles.remove(proj)
                    hit = True
                    break
            if hit:
                continue

            # If projectile reached target area and missed, remove it
            if dist < proj['speed']:
                self._create_particles(proj['pos'], (100,100,100), 2)
                self.projectiles.remove(proj)

    def _update_enemies(self):
        wave_cleared = False
        base_damage_total = 0
        
        # Check for wave clear condition
        if not self.enemies and self.wave_number <= self.MAX_WAVES:
            wave_cleared = True

        for enemy in self.enemies.copy():
            if enemy['spawn_delay'] > 0:
                enemy['spawn_delay'] -= 1
                continue

            if enemy['health'] <= 0:
                self.gold += 5 + self.wave_number
                self._create_particles(enemy['pos'], self.COLOR_ENEMY, 15)
                # sfx: enemy_die
                self.enemies.remove(enemy)
                self.score += 0.1 # Reward for kill
                continue

            path_idx = enemy['path_idx']
            if path_idx >= len(self.path_pixels) - 1:
                self.base_health -= 1
                base_damage_total += 1
                self.enemies.remove(enemy)
                self._create_particles(self.path_pixels[-1], self.COLOR_BASE_DMG, 20)
                # sfx: base_damage
                continue

            target_pos = self.path_pixels[path_idx + 1]
            dx = target_pos[0] - enemy['pos'][0]
            dy = target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < enemy['speed']:
                enemy['pos'] = list(target_pos)
                enemy['path_idx'] += 1
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']
        
        return wave_cleared, base_damage_total

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": random.randint(10, 20),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles.copy():
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

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
                screen_pos = self._iso_to_screen(x, y)
                is_path = (x, y) in self.path_coords
                color = self.COLOR_PATH if is_path else self.COLOR_GRID
                
                points = [
                    (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT / 2),
                    (screen_pos[0] + self.TILE_WIDTH / 2, screen_pos[1]),
                    (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT / 2),
                    (screen_pos[0] - self.TILE_WIDTH / 2, screen_pos[1]),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
        
        # Draw base
        base_screen_pos = self._iso_to_screen(*self.base_pos_grid)
        base_size = 20
        base_rect = pygame.Rect(base_screen_pos[0] - base_size/2, base_screen_pos[1] - base_size/2, base_size, base_size)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=4)
        
        # Draw towers
        for tower in self.towers:
            x, y = tower['pos_screen']
            color = tower['spec']['color']
            pygame.gfxdraw.filled_circle(self.screen, x, y, 8, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, 8, color)
        
        # Draw enemies
        for enemy in sorted(self.enemies, key=lambda e: e['pos'][1]):
            if enemy['spawn_delay'] > 0: continue
            x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 5, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, x, y, 5, self.COLOR_ENEMY)

        # Draw projectiles
        for proj in self.projectiles:
            x, y = int(proj['pos'][0]), int(proj['pos'][1])
            pygame.draw.line(self.screen, proj['color'], (x,y), (x-3, y-3), 2)
        
        # Draw particles
        for p in self.particles:
            x, y = int(p['pos'][0]), int(p['pos'][1])
            size = max(1, int(p['lifespan'] / 5))
            pygame.draw.rect(self.screen, p['color'], (x, y, size, size))
            
        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        screen_pos = self._iso_to_screen(cursor_x, cursor_y)
        spec = self.TOWER_SPECS[self.tower_types[self.selected_tower_type_idx]]
        
        is_on_path = tuple(self.cursor_pos) in self.path_coords
        is_occupied = any(t['pos_grid'] == self.cursor_pos for t in self.towers)
        can_afford = self.gold >= spec['cost']
        color = self.CURSOR_VALID if not is_on_path and not is_occupied and can_afford else self.CURSOR_INVALID

        points = [
            (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT / 2),
            (screen_pos[0] + self.TILE_WIDTH / 2, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT / 2),
            (screen_pos[0] - self.TILE_WIDTH / 2, screen_pos[1]),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Draw tower range for selected tower
        pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], spec['range'], color)

    def _render_ui(self):
        # Info panel background
        pygame.draw.rect(self.screen, (10, 15, 25, 200), (0, 0, self.SCREEN_WIDTH, 40))

        # Base Health
        health_text = self.font_m.render(f"Base HP: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        
        # Gold
        gold_text = self.font_m.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (180, 10))
        
        # Wave
        wave_text = self.font_m.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (320, 10))
        
        # Selected Tower
        tower_type_name = self.tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type_name]
        tower_text = self.font_s.render(f"Tower: {tower_type_name} (Cost: {spec['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (460, 12))

        # Game Over Message
        if self.game_over_message:
            text_surf = self.font_l.render(self.game_over_message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,180), text_rect.inflate(20,20))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "gold": self.gold,
            "base_health": self.base_health,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        
        # Test game-specific assertions
        assert self.base_health <= 100
        assert self.gold >= 0
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    running = True
    
    total_reward = 0
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        # --- Human Input ---
        movement, space_held, shift_held = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Waves Survived: {info['wave']-1}")
            pygame.time.wait(3000) # Pause before reset
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(30) # Limit to 30 FPS

    env.close()