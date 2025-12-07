import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to select a build location. Press Shift to cycle tower types. Press Space to build."
    )

    game_description = (
        "Isometric tower defense. Place towers to defend your base from waves of enemies. Survive for 500 steps to win."
    )

    auto_advance = False

    # --- Constants ---
    # Game parameters
    MAX_STEPS = 500
    INITIAL_BASE_HEALTH = 20
    INITIAL_SCRAP = 150
    ENEMY_SPAWN_INTERVAL = 30 # steps
    
    # Screen dimensions
    WIDTH, HEIGHT = 640, 400

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_PATH = (50, 60, 70)
    COLOR_SLOT = (70, 80, 90)
    COLOR_SLOT_HIGHLIGHT = (255, 255, 0)
    COLOR_BASE = (0, 150, 200)
    COLOR_BASE_DMG = (255, 80, 80)
    
    COLOR_ENEMY_BODY = (200, 50, 50)
    COLOR_ENEMY_TOP = (220, 70, 70)
    
    COLOR_HEALTH_BG = (80, 20, 20)
    COLOR_HEALTH_FG = (50, 200, 50)

    COLOR_TEXT = (220, 220, 220)
    COLOR_SCORE = (255, 200, 0)
    
    # Tower definitions
    TOWER_SPECS = {
        'Gatling': {
            'cost': 50, 'range': 90, 'damage': 1, 'fire_rate': 8, 'color': (100, 255, 100), 'proj_speed': 8, 'proj_color': (150, 255, 150)
        },
        'Cannon': {
            'cost': 120, 'range': 130, 'damage': 5, 'fire_rate': 45, 'color': (100, 100, 255), 'proj_speed': 6, 'proj_color': (150, 150, 255)
        }
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        self.tower_types = list(self.TOWER_SPECS.keys())

        # Isometric projection settings
        self.iso_tile_w = 48
        self.iso_tile_h = 24
        self.iso_origin_x = self.WIDTH // 2
        self.iso_origin_y = 80

        self._define_path_and_slots()
        # The reset is called here to initialize all game state variables
        # before any other method (like _get_observation) might be called.
        self.reset()
        
    def _define_path_and_slots(self):
        # Path for enemies (grid coordinates)
        path_coords = []
        for i in range(-5, 5): path_coords.append((i, 4))
        for i in range(4, -3, -1): path_coords.append((4, i))
        for i in range(4, -6, -1): path_coords.append((i, -3))
        self.enemy_path = [self._iso_to_screen(x, y) for x, y in path_coords]

        # Available slots for towers (grid coordinates)
        slot_coords = [
            (1, 2), (3, 2), (1, 0), (3, 0), 
            (1, -2), (3, -2), (5, -1), (-4, 2)
        ]
        self.tower_slots = [{'pos': self._iso_to_screen(x, y), 'tower': None} for x, y in slot_coords]

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.iso_origin_x + (grid_x - grid_y) * self.iso_tile_w / 2
        screen_y = self.iso_origin_y + (grid_x + grid_y) * self.iso_tile_h / 2
        return (screen_x, screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.scrap = self.INITIAL_SCRAP
        self.enemies_defeated = 0

        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        for slot in self.tower_slots:
            slot['tower'] = None
        self.towers = []

        self.selected_tower_slot_idx = 0
        self.selected_tower_type_idx = 0
        
        self.last_shift_press = False
        self.last_space_press = False
        self.last_move_press = False

        self.enemy_id_counter = 0
        self.last_spawn_step = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for each step to encourage efficiency

        # --- Handle Input ---
        # Cycle tower type on Shift press (rising edge)
        if shift_pressed and not self.last_shift_press:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.tower_types)
        self.last_shift_press = shift_pressed

        # Cycle tower slot on arrow key press (rising edge)
        if movement != 0 and not self.last_move_press:
            if movement in [1, 4]: # Up, Right
                self.selected_tower_slot_idx = (self.selected_tower_slot_idx + 1) % len(self.tower_slots)
            elif movement in [2, 3]: # Down, Left
                self.selected_tower_slot_idx = (self.selected_tower_slot_idx - 1 + len(self.tower_slots)) % len(self.tower_slots)
        self.last_move_press = (movement != 0)

        # Place tower on Space press (rising edge)
        if space_pressed and not self.last_space_press:
            self._place_tower()
        self.last_space_press = space_pressed

        # --- Game Logic Update ---
        self._spawn_enemies()
        
        for tower in self.towers:
            new_projectiles = self._update_tower(tower)
            self.projectiles.extend(new_projectiles)

        hit_rewards = self._update_projectiles()
        reward += hit_rewards

        kill_rewards = self._update_enemies()
        reward += kill_rewards
        
        self._update_particles()
        
        self.steps += 1
        
        # --- Termination Check ---
        terminated = False
        if self.base_health <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward += 100
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_tower(self):
        slot = self.tower_slots[self.selected_tower_slot_idx]
        if slot['tower'] is not None:
            return

        tower_type = self.tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type]
        
        if self.scrap >= spec['cost']:
            self.scrap -= spec['cost']
            new_tower = {
                'pos': slot['pos'], 'type': tower_type, 'spec': spec,
                'cooldown': 0, 'id': len(self.towers)
            }
            self.towers.append(new_tower)
            slot['tower'] = new_tower
            
    def _spawn_enemies(self):
        spawn_rate_modifier = 1 + (self.steps // 200)
        current_spawn_interval = self.ENEMY_SPAWN_INTERVAL / spawn_rate_modifier

        if self.steps - self.last_spawn_step >= current_spawn_interval:
            self.last_spawn_step = self.steps
            health_modifier = 1 + (self.steps // 500)
            
            new_enemy = {
                'id': self.enemy_id_counter,
                'path_idx': 0,
                'pos': self.enemy_path[0],
                'health': 3 * health_modifier,
                'max_health': 3 * health_modifier,
                'speed': self.np_random.uniform(0.8, 1.2),
                'sub_pos': 0.0
            }
            self.enemies.append(new_enemy)
            self.enemy_id_counter += 1

    def _update_tower(self, tower):
        if tower['cooldown'] > 0:
            tower['cooldown'] -= 1
            return []
        
        target = None
        max_path_idx = -1
        
        for enemy in self.enemies:
            dist = math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1])
            if dist <= tower['spec']['range']:
                if enemy['path_idx'] > max_path_idx:
                    max_path_idx = enemy['path_idx']
                    target = enemy
        
        if target:
            tower['cooldown'] = tower['spec']['fire_rate']
            proj = {
                'pos': list(tower['pos']),
                'target_id': target['id'],
                'spec': tower['spec']
            }
            return [proj]
        return []

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            target = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            
            if not target:
                continue

            direction = (target['pos'][0] - proj['pos'][0], target['pos'][1] - proj['pos'][1])
            dist = math.hypot(*direction)
            
            if dist < proj['spec']['proj_speed']:
                reward += 0.1
                target['health'] -= proj['spec']['damage']
                self._create_particles(target['pos'], proj['spec']['proj_color'], 5, 2)
            else:
                norm_dir = (direction[0] / dist, direction[1] / dist)
                proj['pos'][0] += norm_dir[0] * proj['spec']['proj_speed']
                proj['pos'][1] += norm_dir[1] * proj['spec']['proj_speed']
                projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep
        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_keep = []
        for enemy in self.enemies:
            if enemy['health'] <= 0:
                reward += 1.0
                self.score += 10
                self.scrap += 25
                self.enemies_defeated += 1
                self._create_particles(enemy['pos'], self.COLOR_ENEMY_TOP, 20, 4)
                continue

            enemy['sub_pos'] += enemy['speed']
            if int(enemy['sub_pos']) >= 1:
                enemy['path_idx'] += 1
                enemy['sub_pos'] -= 1.0

            if enemy['path_idx'] >= len(self.enemy_path) - 1:
                self.base_health -= 1
                self._create_particles(self.enemy_path[-1], self.COLOR_BASE_DMG, 30, 5)
                continue

            p1 = self.enemy_path[enemy['path_idx']]
            p2 = self.enemy_path[enemy['path_idx'] + 1]
            enemy['pos'] = (
                p1[0] + (p2[0] - p1[0]) * enemy['sub_pos'],
                p1[1] + (p2[1] - p1[1]) * enemy['sub_pos']
            )
            enemies_to_keep.append(enemy)

        self.enemies = enemies_to_keep
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['vel'][1] += 0.05 # Gravity

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 25),
                'max_life': 25,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "base_health": self.base_health, "scrap": self.scrap}

    def _draw_iso_rect(self, surface, color, grid_x, grid_y, w=1, h=1):
        points = [
            self._iso_to_screen(grid_x, grid_y),
            self._iso_to_screen(grid_x + w, grid_y),
            self._iso_to_screen(grid_x + w, grid_y + h),
            self._iso_to_screen(grid_x, grid_y + h),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_iso_cube(self, surface, pos, size, top_color, side_color):
        x, y = pos
        half_w, half_h = self.iso_tile_w / 2, self.iso_tile_h / 2
        
        top_points = [
            (x, y - size), (x + half_w, y - size + half_h),
            (x, y - size + self.iso_tile_h), (x - half_w, y - size + half_h)
        ]
        side1_points = [
            (x, y), (x, y - size + self.iso_tile_h),
            (x - half_w, y - size + half_h), (x - half_w, y + half_h)
        ]
        side2_points = [
            (x, y), (x, y - size + self.iso_tile_h),
            (x + half_w, y - size + half_h), (x + half_w, y + half_h)
        ]
        
        pygame.gfxdraw.filled_polygon(surface, side1_points, side_color)
        pygame.gfxdraw.filled_polygon(surface, side2_points, side_color)
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, side1_points, side_color)
        pygame.gfxdraw.aapolygon(surface, side2_points, side_color)

    def _render_game(self):
        # Draw path
        for i in range(len(self.enemy_path) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.enemy_path[i], self.enemy_path[i+1], int(self.iso_tile_w * 0.8))

        # Draw tower slots
        for i, slot in enumerate(self.tower_slots):
            color = self.COLOR_SLOT_HIGHLIGHT if i == self.selected_tower_slot_idx else self.COLOR_SLOT
            pygame.gfxdraw.filled_circle(self.screen, int(slot['pos'][0]), int(slot['pos'][1]), 15, (*color, 50))
            pygame.gfxdraw.aacircle(self.screen, int(slot['pos'][0]), int(slot['pos'][1]), 15, color)

        # Draw base
        side_color_base = tuple(int(c * 0.7) for c in self.COLOR_BASE)
        self._draw_iso_cube(self.screen, self.enemy_path[-1], 20, self.COLOR_BASE, side_color_base)
        
        # Draw towers
        for tower in self.towers:
            side_color_tower = tuple(int(c * 0.7) for c in tower['spec']['color'])
            self._draw_iso_cube(self.screen, tower['pos'], 12, tower['spec']['color'], side_color_tower)

        # Draw enemies
        for enemy in sorted(self.enemies, key=lambda e: e['pos'][1]):
            self._draw_iso_cube(self.screen, enemy['pos'], 8, self.COLOR_ENEMY_TOP, self.COLOR_ENEMY_BODY)
            # Health bar
            bar_w = 20
            h_ratio = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (enemy['pos'][0] - bar_w/2, enemy['pos'][1] - 25, bar_w, 4))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (enemy['pos'][0] - bar_w/2, enemy['pos'][1] - 25, bar_w * h_ratio, 4))

        # Draw projectiles
        for proj in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 3, proj['spec']['proj_color'])
            pygame.gfxdraw.aacircle(self.screen, int(proj['pos'][0]), int(proj['pos'][1]), 3, proj['spec']['proj_color'])

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), max(0, size), color)

    def _render_ui(self):
        # Top Left: Health and Scrap
        base_health_text = self.font_medium.render(f"Base HP: {self.base_health}/{self.INITIAL_BASE_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(base_health_text, (10, 10))
        scrap_text = self.font_medium.render(f"Scrap: {self.scrap}", True, self.COLOR_SCORE)
        self.screen.blit(scrap_text, (10, 35))

        # Top Right: Score and Step
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        step_text = self.font_medium.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(step_text, (self.WIDTH - step_text.get_width() - 10, 35))
        
        # Bottom Center: Selected Tower Info
        tower_type = self.tower_types[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type]
        name_text = self.font_medium.render(f"Build: {tower_type}", True, spec['color'])
        cost_text = self.font_small.render(f"Cost: {spec['cost']} | Rng: {spec['range']} | Dmg: {spec['damage']}", True, self.COLOR_TEXT)
        self.screen.blit(name_text, (self.WIDTH/2 - name_text.get_width()/2, self.HEIGHT - 50))
        self.screen.blit(cost_text, (self.WIDTH/2 - cost_text.get_width()/2, self.HEIGHT - 25))

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY" if self.base_health > 0 else "GAME OVER"
            color = (0, 255, 0) if self.base_health > 0 else (255, 0, 0)
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # The main block is for human play and visualization,
    # so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    running = True
    
    total_reward = 0
    terminated = False
    
    print(env.game_description)
    print(env.user_guide)
    
    # Use a variable to track rising edge for key presses
    last_keys = pygame.key.get_pressed()

    while running:
        movement = 0 # no-op
        space_pressed = 0
        shift_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
        
        if not terminated:
            current_keys = pygame.key.get_pressed()
            
            # Use rising edge detection for single actions
            if current_keys[pygame.K_UP] and not last_keys[pygame.K_UP]: movement = 1
            elif current_keys[pygame.K_DOWN] and not last_keys[pygame.K_DOWN]: movement = 2
            elif current_keys[pygame.K_LEFT] and not last_keys[pygame.K_LEFT]: movement = 3
            elif current_keys[pygame.K_RIGHT] and not last_keys[pygame.K_RIGHT]: movement = 4
            
            if current_keys[pygame.K_SPACE] and not last_keys[pygame.K_SPACE]: space_pressed = 1
            if (current_keys[pygame.K_LSHIFT] or current_keys[pygame.K_RSHIFT]) and not (last_keys[pygame.K_LSHIFT] or last_keys[pygame.K_RSHIFT]):
                shift_pressed = 1

            action = [movement, space_pressed, shift_pressed]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        last_keys = pygame.key.get_pressed()

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()