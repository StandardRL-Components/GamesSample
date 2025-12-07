
# Generated: 2025-08-28T00:56:03.648455
# Source Brief: brief_03942.md
# Brief Index: 3942

        
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

    user_guide = (
        "Controls: Use arrow keys to select a build location. Press Shift to cycle tower types. Press Space to build."
    )

    game_description = (
        "Defend your base from waves of invading enemies by strategically placing towers in an isometric world."
    )

    auto_advance = True

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_PATH = (45, 50, 66)
    COLOR_GRID = (35, 40, 56)
    COLOR_GRID_VALID = (60, 80, 110)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_BASE = (0, 200, 100)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_HEALTH_BAR_BG = (80, 20, 20)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TOWER_GUN = (0, 150, 255)
    COLOR_TOWER_CANNON = (255, 150, 0)
    COLOR_PROJECTILE_GUN = (100, 200, 255)
    COLOR_PROJECTILE_CANNON = (255, 200, 100)
    
    # --- Game Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 30000 # Extended to allow for 20 waves
    MAX_WAVES = 20
    INITIAL_BASE_HEALTH = 50
    INITIAL_RESOURCES = 250
    
    # --- Isometric Grid ---
    TILE_WIDTH_HALF = 28
    TILE_HEIGHT_HALF = 14
    GRID_ORIGIN_X = SCREEN_WIDTH // 2
    GRID_ORIGIN_Y = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        
        self._define_world()
        self.reset()
        self.validate_implementation()

    def _define_world(self):
        # Define the path enemies will follow (in grid coordinates)
        self.path_coords = [
            (-5, 0), (-4, 0), (-3, 0), (-2, 0), (-1, 0), (0, 0), (0, 1), (0, 2),
            (1, 2), (2, 2), (3, 2), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5),
            (6, 5), (6, 4), (6, 3), (6, 2), (6, 1), (7, 1), (8, 1)
        ]
        self.base_pos_grid = (8, 0) # Grid coordinate of the base

        # Define valid tower placement spots
        self.placement_spots = sorted([
            (-2, 1), (-1, 1), (1, 1), (2, 1),
            (1, 3), (2, 3), (4, 3), (5, 3),
            (4, 4), (5, 4), (5, 2),
            (7, 2), (7, 0)
        ], key=lambda p: (p[1], p[0]))
        
        # Define tower types
        self.tower_types = [
            {"name": "Gun", "cost": 100, "range": 100, "damage": 5, "fire_rate": 10, "color": self.COLOR_TOWER_GUN, "proj_color": self.COLOR_PROJECTILE_GUN},
            {"name": "Cannon", "cost": 200, "range": 150, "damage": 25, "fire_rate": 45, "color": self.COLOR_TOWER_CANNON, "proj_color": self.COLOR_PROJECTILE_CANNON},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.reward_this_step = 0
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.current_wave = 0
        self.wave_timer = 150 # Time until first wave starts
        self.wave_spawning = False
        self.enemies_to_spawn_this_wave = 0
        self.enemies_spawned_this_wave = 0
        self.spawn_cooldown = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_index = 0
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        
        self._handle_input(action)
        self._update_game_state()
        
        self.steps += 1
        terminated = self._check_termination()
        reward = self.reward_this_step
        
        if terminated:
            if self.victory:
                reward += 100
            else:
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement: Change cursor index ---
        if movement == 1: self.cursor_index = max(0, self.cursor_index - 1) # Up (prev in list)
        elif movement == 2: self.cursor_index = min(len(self.placement_spots) - 1, self.cursor_index + 1) # Down (next in list)
        # Left/Right could be implemented to jump rows, but up/down is simpler and effective.
        
        # --- Action 1: Place Tower (on key press) ---
        if space_held and not self.prev_space_held:
            self._place_tower()
        
        # --- Action 2: Cycle Tower Type (on key press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_types)
            # sfx: UI_switch
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _place_tower(self):
        spot_grid_pos = self.placement_spots[self.cursor_index]
        is_occupied = any(t['grid_pos'] == spot_grid_pos for t in self.towers)
        
        tower_data = self.tower_types[self.selected_tower_type]
        
        if not is_occupied and self.resources >= tower_data['cost']:
            self.resources -= tower_data['cost']
            
            new_tower = {
                "grid_pos": spot_grid_pos,
                "type_index": self.selected_tower_type,
                "cooldown": 0,
                "target": None
            }
            self.towers.append(new_tower)
            # sfx: tower_place
            self._create_particles(self._grid_to_screen(spot_grid_pos), 20, tower_data['color'])

    def _update_game_state(self):
        self._update_waves()
        self._update_towers()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()

    def _update_waves(self):
        all_enemies_defeated = not self.enemies and self.wave_spawning and self.enemies_spawned_this_wave == self.enemies_to_spawn_this_wave
        
        if all_enemies_defeated and self.current_wave <= self.MAX_WAVES:
            self.reward_this_step += 1.0 # Wave complete reward
            self.score += 50 * self.current_wave
            self.wave_spawning = False
            self.wave_timer = 200 # Time between waves
            if self.current_wave == self.MAX_WAVES:
                self.victory = True
                self.game_over = True
        
        if not self.wave_spawning and not self.game_over:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                self.wave_spawning = True
                self.enemies_to_spawn_this_wave = 2 + self.current_wave * 2
                self.enemies_spawned_this_wave = 0
                self.spawn_cooldown = 0

        if self.wave_spawning and self.enemies_spawned_this_wave < self.enemies_to_spawn_this_wave:
            self.spawn_cooldown -= 1
            if self.spawn_cooldown <= 0:
                self._spawn_enemy()
                self.spawn_cooldown = 30 - self.current_wave # Spawn faster in later waves

    def _spawn_enemy(self):
        start_pos = self._grid_to_screen(self.path_coords[0])
        base_health = 15 + (self.current_wave - 1) * 5 * 1.05**(self.current_wave - 1)
        base_speed = 0.8 + (self.current_wave - 1) * 0.05 * 1.02**(self.current_wave - 1)
        
        enemy = {
            "pos": list(start_pos),
            "health": base_health,
            "max_health": base_health,
            "speed": base_speed,
            "waypoint_index": 1,
            "id": self.steps + self.enemies_spawned_this_wave,
        }
        self.enemies.append(enemy)
        self.enemies_spawned_this_wave += 1
        # sfx: enemy_spawn

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            if enemy["waypoint_index"] >= len(self.path_coords):
                self.base_health -= 5
                self.reward_this_step -= 0.01
                self.enemies.remove(enemy)
                # sfx: base_damage
                self._create_particles(self._grid_to_screen(self.base_pos_grid), 30, self.COLOR_ENEMY)
                continue

            target_pos = self._grid_to_screen(self.path_coords[enemy["waypoint_index"]])
            direction = np.array(target_pos) - np.array(enemy["pos"])
            distance = np.linalg.norm(direction)
            
            if distance < enemy["speed"]:
                enemy["waypoint_index"] += 1
            else:
                move_vec = (direction / distance) * enemy["speed"]
                enemy["pos"][0] += move_vec[0]
                enemy["pos"][1] += move_vec[1]

    def _update_towers(self):
        for tower in self.towers:
            tower_data = self.tower_types[tower['type_index']]
            tower_pos = self._grid_to_screen(tower['grid_pos'])
            
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            # Invalidate target if it's dead or out of range
            if tower['target']:
                target_enemy = next((e for e in self.enemies if e['id'] == tower['target']), None)
                if not target_enemy or np.linalg.norm(np.array(target_enemy['pos']) - tower_pos) > tower_data['range']:
                    tower['target'] = None
                
            # Find a new target if needed
            if not tower['target']:
                for enemy in self.enemies:
                    if np.linalg.norm(np.array(enemy['pos']) - tower_pos) <= tower_data['range']:
                        tower['target'] = enemy['id']
                        break
            
            # Fire if a valid target exists
            if tower['target']:
                target_enemy = next((e for e in self.enemies if e['id'] == tower['target']), None)
                if target_enemy:
                    self._fire_projectile(tower, target_enemy)
                    tower['cooldown'] = tower_data['fire_rate']

    def _fire_projectile(self, tower, target_enemy):
        tower_data = self.tower_types[tower['type_index']]
        start_pos = self._grid_to_screen(tower['grid_pos'])
        
        projectile = {
            "pos": list(start_pos),
            "target_id": target_enemy['id'],
            "speed": 8,
            "damage": tower_data['damage'],
            "color": tower_data['proj_color']
        }
        self.projectiles.append(projectile)
        # sfx: tower_fire_gun or tower_fire_cannon

    def _update_projectiles(self):
        for p in reversed(self.projectiles):
            target_enemy = next((e for e in self.enemies if e['id'] == p['target_id']), None)
            
            if not target_enemy:
                self.projectiles.remove(p)
                continue

            target_pos = target_enemy['pos']
            direction = np.array(target_pos) - np.array(p['pos'])
            distance = np.linalg.norm(direction)

            if distance < p['speed']:
                target_enemy['health'] -= p['damage']
                self._create_particles(p['pos'], 5, p['color'])
                # sfx: enemy_hit
                if target_enemy['health'] <= 0:
                    self._create_particles(target_enemy['pos'], 30, self.COLOR_ENEMY)
                    # sfx: enemy_explode
                    self.enemies.remove(target_enemy)
                    self.score += 10
                    self.resources += 25
                    self.reward_this_step += 0.1
                self.projectiles.remove(p)
            else:
                move_vec = (direction / distance) * p['speed']
                p['pos'][0] += move_vec[0]
                p['pos'][1] += move_vec[1]
                
    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        if self.victory:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_world()
        self._render_entities()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_world(self):
        # Render placement spots and path
        all_grid_tiles = self.placement_spots + self.path_coords + [self.base_pos_grid]
        for x, y in sorted(list(set(all_grid_tiles)), key=lambda p: (p[1]+p[0], p[1]-p[0])):
            color = self.COLOR_GRID
            if (x,y) in self.path_coords: color = self.COLOR_PATH
            elif (x,y) in self.placement_spots: color = self.COLOR_GRID_VALID
            self._draw_iso_tile((x,y), color)
        
        # Render base
        self._draw_iso_cube(self.base_pos_grid, (self.TILE_WIDTH_HALF*2, self.TILE_HEIGHT_HALF*4), self.COLOR_BASE)

        # Render cursor
        cursor_grid_pos = self.placement_spots[self.cursor_index]
        self._draw_iso_tile(cursor_grid_pos, self.COLOR_CURSOR, outline=True)

    def _render_entities(self):
        # Combine towers and enemies for z-sorting
        render_list = []
        for t in self.towers:
            render_list.append({'type': 'tower', 'data': t})
        for e in self.enemies:
            render_list.append({'type': 'enemy', 'data': e})

        # Sort by screen y-position for correct layering
        def sort_key(item):
            if item['type'] == 'tower':
                return self._grid_to_screen(item['data']['grid_pos'])[1]
            else: # enemy
                return item['data']['pos'][1]
        render_list.sort(key=sort_key)

        for item in render_list:
            if item['type'] == 'tower':
                self._render_tower(item['data'])
            else:
                self._render_enemy(item['data'])
        
        # Render projectiles and particles on top of everything
        for p in self.projectiles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), 3)
        
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * (255 / 30))))
            p_color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, p_color, (2,2), 2)
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))
            
    def _render_tower(self, tower):
        tower_data = self.tower_types[tower['type_index']]
        size = (self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF * 2)
        self._draw_iso_cube(tower['grid_pos'], size, tower_data['color'])
        
        # Range indicator when placing
        if self.placement_spots[self.cursor_index] == tower['grid_pos']:
            pos = self._grid_to_screen(tower['grid_pos'])
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), tower_data['range'], (*tower_data['color'], 80))

    def _render_enemy(self, enemy):
        pos = enemy['pos']
        # Simple diamond shape
        points = [
            (pos[0], pos[1] - 8),
            (pos[0] + 8, pos[1]),
            (pos[0], pos[1] + 8),
            (pos[0] - 8, pos[1]),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Health bar
        bar_width = 20
        health_pct = enemy['health'] / enemy['max_health']
        health_bar_rect_bg = pygame.Rect(pos[0] - bar_width / 2, pos[1] - 20, bar_width, 4)
        health_bar_rect_fg = pygame.Rect(pos[0] - bar_width / 2, pos[1] - 20, bar_width * health_pct, 4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, health_bar_rect_bg)
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, health_bar_rect_fg)

    def _render_ui(self):
        # Top bar background
        pygame.draw.rect(self.screen, (15, 18, 26), (0, 0, self.SCREEN_WIDTH, 30))
        
        # UI Text
        texts = [
            f"HP: {max(0, self.base_health)}/{self.INITIAL_BASE_HEALTH}",
            f"WAVE: {self.current_wave}/{self.MAX_WAVES}",
            f"RESOURCES: {self.resources}",
            f"SCORE: {self.score}",
        ]
        for i, text in enumerate(texts):
            surf = self.font_main.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surf, (10 + i * 150, 5))

        # Tower selection UI
        tower_data = self.tower_types[self.selected_tower_type]
        select_text = f"Build: {tower_data['name']} (Cost: {tower_data['cost']})"
        surf = self.font_small.render(select_text, True, tower_data['color'])
        self.screen.blit(surf, (self.SCREEN_WIDTH - surf.get_width() - 10, 380))

        # Game Over / Victory Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_BASE if self.victory else self.COLOR_ENEMY
            
            text_surf = pygame.font.SysFont("Consolas", 60, bold=True).render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "resources": self.resources
        }

    # --- Helper Functions ---
    def _grid_to_screen(self, grid_pos):
        x, y = grid_pos
        screen_x = self.GRID_ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.GRID_ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return screen_x, screen_y

    def _draw_iso_tile(self, grid_pos, color, outline=False):
        x, y = self._grid_to_screen(grid_pos)
        points = [
            (x, y - self.TILE_HEIGHT_HALF),
            (x + self.TILE_WIDTH_HALF, y),
            (x, y + self.TILE_HEIGHT_HALF),
            (x - self.TILE_WIDTH_HALF, y)
        ]
        if outline:
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, [(p[0]+1, p[1]) for p in points], color)
            pygame.gfxdraw.aapolygon(self.screen, [(p[0]-1, p[1]) for p in points], color)
        else:
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_iso_cube(self, grid_pos, size, color):
        w, h = size
        x, y = self._grid_to_screen(grid_pos)
        y -= h # Adjust for height
        
        top_color = color
        side_color_left = tuple(max(0, c - 30) for c in color)
        side_color_right = tuple(max(0, c - 60) for c in color)
        
        # Top
        points_top = [(x, y), (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF), 
                      (x, y + self.TILE_HEIGHT_HALF * 2), (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)]
        pygame.gfxdraw.filled_polygon(self.screen, points_top, top_color)
        pygame.gfxdraw.aapolygon(self.screen, points_top, top_color)
        
        # Left side
        points_left = [(x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF), (x, y + self.TILE_HEIGHT_HALF * 2),
                       (x, y + self.TILE_HEIGHT_HALF * 2 + h), (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF + h)]
        pygame.gfxdraw.filled_polygon(self.screen, points_left, side_color_left)
        pygame.gfxdraw.aapolygon(self.screen, points_left, side_color_left)
        
        # Right side
        points_right = [(x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF), (x, y + self.TILE_HEIGHT_HALF * 2),
                        (x, y + self.TILE_HEIGHT_HALF * 2 + h), (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF + h)]
        pygame.gfxdraw.filled_polygon(self.screen, points_right, side_color_right)
        pygame.gfxdraw.aapolygon(self.screen, points_right, side_color_right)
    
    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": random.randint(15, 30),
                "color": color
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    
    # Create a display for manual play
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    action = [0, 0, 0] # no-op, no-space, no-shift
    
    while running:
        # --- Manual Control Mapping ---
        movement = 0 # no-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # If not using keydown events, you can use get_pressed for held keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]: space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Display the observation ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before auto-resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
        
        env.clock.tick(30) # Run at 30 FPS
        
    pygame.quit()