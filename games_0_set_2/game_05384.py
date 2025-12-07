import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle through tower types. Press Space to build a tower."
    )

    game_description = (
        "An isometric tower defense game. Defend your castle from 10 waves of creeps "
        "by strategically placing towers. Earn gold by defeating creeps and surviving waves."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()
        self.dt = 0.0

        # --- Game Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 22, 14
        self.TILE_WIDTH, self.TILE_HEIGHT = 40, 20
        self.ISO_OFFSET_X = self.screen_size[0] / 2
        self.ISO_OFFSET_Y = 100

        self.MAX_CASTLE_HEALTH = 1000
        self.STARTING_GOLD = 250
        self.MAX_WAVES = 10
        self.INTER_WAVE_TIME = 5 * 30  # 5 seconds in frames
        self.MAX_STEPS = 30 * 60 * 5 # 5 minutes at 30fps

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_BUILDABLE = (25, 35, 45)
        self.COLOR_CASTLE = (100, 180, 255)
        self.COLOR_CASTLE_DMG = (255, 100, 100)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_UI_BG = (30, 40, 55, 180)

        # --- Fonts ---
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 60)

        # --- Game Assets ---
        self._define_game_assets()

        # --- State Initialization ---
        # The reset method is called here to initialize the state
        # but since it returns values, we just call it without assignment
        # The actual first state will be returned by the external call to env.reset()
        self.reset()
        
        # --- Validation ---
        self.validate_implementation()

    def _define_game_assets(self):
        self.PATH_COORDS = [
            (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (6, 3),
            (7, 3), (8, 3), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (8, 8),
            (7, 8), (7, 9), (7, 10), (8, 10), (9, 10), (10, 10), (11, 10), (12, 10),
            (13, 10), (14, 10), (14, 9), (14, 8), (15, 8), (16, 8), (17, 8), (18, 8),
            (19, 8), (20, 8), (21, 8)
        ]
        self.CASTLE_POS = (21, 8)
        self.BUILDABLE_TILES = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if (x, y) not in self.PATH_COORDS:
                    self.BUILDABLE_TILES.add((x, y))

        self.TOWER_TYPES = [
            {"name": "Cannon", "cost": 100, "damage": 25, "range": 3.0, "fire_rate": 1.0, "color": (0, 150, 255), "proj_speed": 400},
            {"name": "Missile", "cost": 200, "damage": 75, "range": 4.5, "fire_rate": 0.4, "color": (255, 100, 0), "proj_speed": 300},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.castle_health = self.MAX_CASTLE_HEALTH
        self.gold = self.STARTING_GOLD
        self.wave_number = 0
        self.wave_in_progress = False
        self.inter_wave_timer = 90 # 3 second start delay

        self.creeps_to_spawn = []
        self.spawn_timer = 0.0
        self.creeps = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.dt = self.clock.tick(30) / 1000.0
        self.steps += 1
        reward = 0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward += self._handle_input(action)
        reward += self._update_waves()
        
        self._update_towers()
        reward += self._update_creeps()
        reward += self._update_projectiles()
        self._update_particles()
        
        termination_reward = self._check_termination_conditions()
        reward += termination_reward

        self.score += reward
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Cycle Tower Type (on press) ---
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)

        # --- Place Tower (on press) ---
        if space_held and not self.last_space_held:
            self._place_tower()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return 0

    def _place_tower(self):
        pos = tuple(self.cursor_pos)
        tower_spec = self.TOWER_TYPES[self.selected_tower_type]
        
        is_buildable = pos in self.BUILDABLE_TILES
        is_unoccupied = not any(t['pos'] == pos for t in self.towers)
        can_afford = self.gold >= tower_spec['cost']

        if is_buildable and is_unoccupied and can_afford:
            self.gold -= tower_spec['cost']
            self.towers.append({
                "pos": pos,
                "type": self.selected_tower_type,
                "cooldown": 0,
                "target": None,
            })
            self._create_particles(pos, (200, 200, 255), 15, 2.0)

    def _update_waves(self):
        if self.wave_in_progress:
            if not self.creeps and not self.creeps_to_spawn:
                self.wave_in_progress = False
                self.inter_wave_timer = self.INTER_WAVE_TIME
                self.gold += 100 + self.wave_number * 10
                if self.wave_number >= self.MAX_WAVES:
                    self.victory = True
                return 100
        else: # Between waves
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0 and not self.victory:
                self._start_new_wave()
        
        if self.creeps_to_spawn:
            self.spawn_timer -= self.dt
            if self.spawn_timer <= 0:
                self.creeps.append(self.creeps_to_spawn.pop(0))
                self.spawn_timer = 0.5
        return 0.1

    def _start_new_wave(self):
        self.wave_number += 1
        self.wave_in_progress = True
        
        num_creeps = 5 + self.wave_number * 2
        base_health = 50 + self.wave_number * 25
        base_speed = 3.0 + self.wave_number * 0.2
        
        for _ in range(num_creeps):
            creep = {
                "path_index": 0,
                "pos": list(self.PATH_COORDS[0]),
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed,
                "dist_to_next": 0.0
            }
            self.creeps_to_spawn.append(creep)

    def _update_creeps(self):
        reward = 0
        for creep in self.creeps[:]:
            path_idx = creep['path_index']
            if path_idx >= len(self.PATH_COORDS) - 1:
                self.creeps.remove(creep)
                self.castle_health -= creep['health']
                reward -= 5
                reward -= creep['health'] * 0.01
                continue

            start_pos = self.PATH_COORDS[path_idx]
            end_pos = self.PATH_COORDS[path_idx + 1]
            
            dist_to_move = creep['speed'] * self.dt
            vec = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
            
            creep['pos'][0] += vec[0] * dist_to_move
            creep['pos'][1] += vec[1] * dist_to_move
            
            if np.dot( (creep['pos'][0] - end_pos[0], creep['pos'][1] - end_pos[1]), vec ) >= 0:
                creep['pos'] = list(end_pos)
                creep['path_index'] += 1
        return reward

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_TYPES[tower['type']]
            tower['cooldown'] = max(0, tower['cooldown'] - self.dt)

            if tower['cooldown'] > 0:
                continue

            best_target = None
            max_dist = -1
            for creep in self.creeps:
                dist = math.hypot(creep['pos'][0] - tower['pos'][0], creep['pos'][1] - tower['pos'][1])
                if dist <= spec['range']:
                    creep_path_dist = creep['path_index'] + (1 - creep.get('dist_to_next', 0.0))
                    if creep_path_dist > max_dist:
                        max_dist = creep_path_dist
                        best_target = creep
            
            if best_target:
                tower['cooldown'] = 1.0 / spec['fire_rate']
                self.projectiles.append({
                    "start_pos": tower['pos'],
                    "pos": self._iso_to_screen(*tower['pos']),
                    "target": best_target,
                    "spec": spec
                })

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            if p['target'] not in self.creeps:
                self.projectiles.remove(p)
                continue

            target_screen_pos = self._iso_to_screen(*p['target']['pos'])
            target_screen_pos = (target_screen_pos[0], target_screen_pos[1] - self.TILE_HEIGHT)
            
            direction = (target_screen_pos[0] - p['pos'][0], target_screen_pos[1] - p['pos'][1])
            dist = math.hypot(*direction)
            
            if dist < 10:
                p['target']['health'] -= p['spec']['damage']
                self._create_particles(p['target']['pos'], p['spec']['color'], 10, 1.5)
                if p['target']['health'] <= 0:
                    self.gold += 5
                    self._create_particles(p['target']['pos'], (255, 50, 50), 20, 3.0)
                    self.creeps.remove(p['target'])
                    reward += 1
                self.projectiles.remove(p)
                continue

            norm_dir = (direction[0] / dist, direction[1] / dist)
            p['pos'] = (p['pos'][0] + norm_dir[0] * p['spec']['proj_speed'] * self.dt,
                        p['pos'][1] + norm_dir[1] * p['spec']['proj_speed'] * self.dt)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0] * self.dt
            p['pos'][1] += p['vel'][1] * self.dt
            p['vel'][1] += 50 * self.dt
            p['life'] -= self.dt
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination_conditions(self):
        if self.castle_health <= 0:
            self.game_over = True
            self.castle_health = 0
            return -100
        if self.victory:
            self.game_over = True
            return 200
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return -50
        return 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_pos = self._iso_to_screen(x, y)
                tile_points = [
                    (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT / 2),
                    (screen_pos[0] + self.TILE_WIDTH / 2, screen_pos[1]),
                    (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT / 2),
                    (screen_pos[0] - self.TILE_WIDTH / 2, screen_pos[1]),
                ]
                color = self.COLOR_PATH if (x, y) in self.PATH_COORDS else self.COLOR_BUILDABLE
                pygame.draw.polygon(self.screen, color, tile_points)

        castle_screen_pos = self._iso_to_screen(*self.CASTLE_POS)
        health_perc = self.castle_health / self.MAX_CASTLE_HEALTH
        c_start = np.array(self.COLOR_CASTLE_DMG)
        c_end = np.array(self.COLOR_CASTLE)
        interpolated = c_start * (1 - health_perc) + c_end * health_perc
        castle_color = tuple(np.clip(interpolated, 0, 255).astype(int))
        castle_rect = pygame.Rect(castle_screen_pos[0]-20, castle_screen_pos[1]-30, 40, 40)
        pygame.draw.rect(self.screen, castle_color, castle_rect, border_radius=4)
        pygame.draw.rect(self.screen, (255,255,255), castle_rect, 2, border_radius=4)

        render_queue = []
        for t in self.towers: render_queue.append(("tower", t))
        for c in self.creeps: render_queue.append(("creep", c))
        
        render_queue.sort(key=lambda item: item[1]['pos'][0] + item[1]['pos'][1])

        for item_type, item in render_queue:
            if item_type == "tower": self._render_tower(item)
            elif item_type == "creep": self._render_creep(item)

        for p in self.projectiles: self._render_projectile(p)
        for p in self.particles: self._render_particle(p)

        self._render_cursor()

    def _render_tower(self, tower):
        spec = self.TOWER_TYPES[tower['type']]
        screen_pos = self._iso_to_screen(*tower['pos'])
        pygame.draw.circle(self.screen, spec['color'], (int(screen_pos[0]), int(screen_pos[1] - self.TILE_HEIGHT/2)), 10)
        pygame.draw.circle(self.screen, (255,255,255), (int(screen_pos[0]), int(screen_pos[1] - self.TILE_HEIGHT/2)), 10, 2)
        
    def _render_creep(self, creep):
        screen_pos = self._iso_to_screen(*creep['pos'])
        x, y = int(screen_pos[0]), int(screen_pos[1] - self.TILE_HEIGHT)
        
        body_points = [(x, y - 5), (x + 7, y), (x, y + 5), (x - 7, y)]
        pygame.gfxdraw.aapolygon(self.screen, body_points, (255, 50, 50))
        pygame.gfxdraw.filled_polygon(self.screen, body_points, (255, 50, 50))

        health_perc = creep['health'] / creep['max_health']
        bar_width = 16
        pygame.draw.rect(self.screen, (50, 0, 0), (x - bar_width/2, y - 15, bar_width, 4))
        pygame.draw.rect(self.screen, (0, 200, 0), (x - bar_width/2, y - 15, bar_width * health_perc, 4))

    def _render_projectile(self, projectile):
        pygame.draw.circle(self.screen, projectile['spec']['color'], (int(projectile['pos'][0]), int(projectile['pos'][1])), 4)
        pygame.gfxdraw.aacircle(self.screen, int(projectile['pos'][0]), int(projectile['pos'][1]), 4, (255,255,255))
        
    def _render_particle(self, particle):
        life_perc = max(0, particle['life'] / particle['max_life'])
        size = int(particle['size'] * life_perc)
        if size > 0:
            color = (int(particle['color'][0]*life_perc), int(particle['color'][1]*life_perc), int(particle['color'][2]*life_perc))
            pygame.draw.circle(self.screen, color, (int(particle['pos'][0]), int(particle['pos'][1])), size)

    def _render_cursor(self):
        pos = tuple(self.cursor_pos)
        tower_spec = self.TOWER_TYPES[self.selected_tower_type]
        is_buildable = pos in self.BUILDABLE_TILES
        is_unoccupied = not any(t['pos'] == pos for t in self.towers)
        can_afford = self.gold >= tower_spec['cost']

        valid_placement = is_buildable and is_unoccupied and can_afford
        color = (0, 255, 0, 100) if valid_placement else (255, 0, 0, 100)

        screen_pos = self._iso_to_screen(*self.cursor_pos)
        tile_points = [
            (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT / 2),
            (screen_pos[0] + self.TILE_WIDTH / 2, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT / 2),
            (screen_pos[0] - self.TILE_WIDTH / 2, screen_pos[1]),
        ]
        
        s = pygame.Surface(self.screen_size, pygame.SRCALPHA)
        pygame.draw.polygon(s, color, tile_points)
        self.screen.blit(s, (0,0))
        
        range_px = tower_spec['range'] * (self.TILE_WIDTH / 2)
        pygame.gfxdraw.aacircle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(range_px), (255,255,255,100))

    def _render_ui(self):
        panel = pygame.Surface((self.screen_size[0], 50), pygame.SRCALPHA)
        panel.fill(self.COLOR_UI_BG)
        self.screen.blit(panel, (0, 0))

        health_text = self.font_main.render(f"Castle HP:", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 15))
        health_perc = self.castle_health / self.MAX_CASTLE_HEALTH
        c_start = np.array(self.COLOR_CASTLE_DMG)
        c_end = np.array(self.COLOR_CASTLE)
        interpolated = c_start * (1 - health_perc) + c_end * health_perc
        health_color = tuple(np.clip(interpolated, 0, 255).astype(int))
        pygame.draw.rect(self.screen, (50,50,50), (100, 15, 150, 20))
        pygame.draw.rect(self.screen, health_color, (100, 15, 150 * health_perc, 20))

        gold_text = self.font_main.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (270, 15))

        wave_str = f"Wave: {self.wave_number}/{self.MAX_WAVES}"
        if not self.wave_in_progress and not self.victory:
            wave_str = f"Next wave in {int(self.inter_wave_timer/30)+1}s"
        wave_text = self.font_main.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (380, 15))

        tower_spec = self.TOWER_TYPES[self.selected_tower_type]
        sel_text = self.font_main.render(f"Build: {tower_spec['name']} ({tower_spec['cost']}g)", True, tower_spec['color'])
        self.screen.blit(sel_text, (490, 15))

        if self.game_over:
            s = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            s.fill((0,0,0,150))
            self.screen.blit(s, (0,0))
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = (100, 255, 100) if self.victory else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_size[0]/2, self.screen_size[1]/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave_number,
            "castle_health": self.castle_health
        }

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = (grid_x - grid_y) * (self.TILE_WIDTH / 2) + self.ISO_OFFSET_X
        screen_y = (grid_x + grid_y) * (self.TILE_HEIGHT / 2) + self.ISO_OFFSET_Y
        return screen_x, screen_y

    def _create_particles(self, grid_pos, color, count, max_life):
        screen_pos = self._iso_to_screen(*grid_pos)
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(20, 80)
            self.particles.append({
                "pos": list(screen_pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed - 50],
                "life": random.uniform(max_life * 0.5, max_life),
                "max_life": max_life,
                "size": random.uniform(2, 5),
                "color": color
            })

    def close(self):
        pygame.quit()
        
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
        assert trunc == False
        assert isinstance(info, dict)

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    display_screen = pygame.display.set_mode(env.screen_size)
    pygame.display.set_caption("Tower Defense")
    
    action = env.action_space.sample()
    action.fill(0)

    # Un-comment the following line to see the user guide and description
    # print(f"User Guide: {env.user_guide}\nDescription: {env.game_description}")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        action.fill(0)
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()