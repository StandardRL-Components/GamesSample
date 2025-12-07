
# Generated: 2025-08-27T22:54:57.813260
# Source Brief: brief_03288.md
# Brief Index: 3288

        
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
        "Controls: Arrows to move cursor, Space to place/cycle tower. Defend the base!"
    )

    game_description = (
        "An isometric tower defense game. Place towers on strategic nodes to stop waves of enemies from reaching your base. Survive 20 waves to win."
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
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 64)

        # --- Game Constants ---
        self.GRID_SIZE = (5, 4)
        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps
        self.MAX_WAVES = 20
        self.WAVE_PREP_TIME = 300 # frames (10 seconds)

        # --- Colors ---
        self.COLOR_BG = (30, 40, 50)
        self.COLOR_PATH = (50, 60, 70)
        self.COLOR_SLOT = (0, 100, 150)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_HEALTH_BG = (80, 80, 80)
        self.COLOR_HEALTH = (50, 200, 50)
        
        # --- Tower Definitions ---
        self.TOWER_TYPES = {
            0: {"name": "Cannon", "cost": 100, "range": 80, "damage": 25, "fire_rate": 45, "color": (0, 200, 0), "proj_speed": 5},
            1: {"name": "Missile", "cost": 250, "range": 150, "damage": 80, "fire_rate": 120, "color": (0, 150, 255), "proj_speed": 3},
        }

        # --- World Geometry ---
        self.origin = pygame.Vector2(self.screen_size[0] // 2, 60)
        self.tile_w_half = 32
        self.tile_h_half = 16
        
        self._define_path_and_slots()

        # --- State variables will be initialized in reset() ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.rng = None
        self.space_was_held = False
        self.frame_reward = 0.0

        self.reset()
        self.validate_implementation()
    
    def _define_path_and_slots(self):
        raw_path = [(-2, 2), (2, 2), (2, -2), (6, -2), (6, 2)]
        self.path_waypoints = [self._to_iso(p[0], p[1]) for p in raw_path]

        raw_slots = [
            (0, 0), (1, 1), (3, 1), (3, -1),
            (5, -1), (5, 1), (7, 0), (1, -3), (4, -3)
        ]
        self.tower_slots = [{"pos": self._to_iso(p[0], p[1]), "tower": None} for p in raw_slots]
        self.cursor_index = 0

    def _to_iso(self, x, y):
        iso_x = self.origin.x + (x - y) * self.tile_w_half
        iso_y = self.origin.y + (x + y) * self.tile_h_half
        return pygame.Vector2(iso_x, iso_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.frame_reward = 0.0
        
        self.resources = 250
        self.wave_number = 0
        self.wave_timer = self.WAVE_PREP_TIME // 2
        self.wave_active = False
        self.wave_spawner = []

        self.enemies = []
        self.projectiles = []
        self.particles = []
        for slot in self.tower_slots:
            slot["tower"] = None
        
        self.towers = []
        
        self.cursor_index = 0
        self.selected_tower_type = 0
        self.space_was_held = True # Prevent placing on first frame
        
        self._start_next_wave()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.frame_reward = 0
        self.steps += 1
        
        self._handle_input(action)
        self._update_game_logic()
        
        reward = self.frame_reward
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 100
            else: # Loss or timeout
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

        # --- Cursor Movement ---
        # Map grid to a 1D list for easier traversal
        if movement == 1: self.cursor_index = max(0, self.cursor_index - 1) # Up (conceptually prev)
        elif movement == 2: self.cursor_index = min(len(self.tower_slots) - 1, self.cursor_index + 1) # Down (conceptually next)
        # Left/Right are mapped to prev/next as well for simplicity
        elif movement == 3: self.cursor_index = max(0, self.cursor_index - 1) 
        elif movement == 4: self.cursor_index = min(len(self.tower_slots) - 1, self.cursor_index + 1)

        # --- Place Tower ---
        if space_held and not self.space_was_held:
            slot = self.tower_slots[self.cursor_index]
            tower_def = self.TOWER_TYPES[self.selected_tower_type]
            
            if slot["tower"] is None and self.resources >= tower_def["cost"]:
                self.resources -= tower_def["cost"]
                new_tower = {
                    "pos": slot["pos"],
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    "health": 100 # Not used in brief, but good practice
                }
                self.towers.append(new_tower)
                slot["tower"] = new_tower
                # sfx: place_tower.wav
                
            # Cycle tower type regardless of placement success
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
            
        self.space_was_held = space_held

    def _update_game_logic(self):
        self._update_waves()
        self._update_towers()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()

    def _start_next_wave(self):
        if self.wave_number >= self.MAX_WAVES:
            self.win = True
            return

        self.wave_number += 1
        self.wave_timer = self.WAVE_PREP_TIME
        self.wave_active = False
        
        num_enemies = 2 + self.wave_number
        health = 40 + (self.wave_number - 1) * 20 * (1.05 ** (self.wave_number - 1))
        speed = 0.8 + (self.wave_number - 1) * 0.05 * (1.02 ** (self.wave_number - 1))
        
        self.wave_spawner = []
        for i in range(num_enemies):
            spawn_delay = i * 30 # Spawn one per second
            self.wave_spawner.append({
                "delay": spawn_delay,
                "health": health,
                "speed": speed,
            })
        self.wave_spawner.sort(key=lambda x: x['delay'], reverse=True)

    def _update_waves(self):
        if not self.wave_active:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.wave_active = True
        
        if self.wave_active:
            if self.wave_spawner:
                self.wave_spawner[-1]['delay'] -= 1
                if self.wave_spawner[-1]['delay'] <= 0:
                    enemy_data = self.wave_spawner.pop()
                    self.enemies.append({
                        "pos": self.path_waypoints[0].copy(),
                        "max_health": enemy_data['health'],
                        "health": enemy_data['health'],
                        "speed": enemy_data['speed'],
                        "waypoint_index": 1,
                    })
            elif not self.enemies: # Wave is over
                self.frame_reward += 1.0
                self.score += 10 * self.wave_number
                self._start_next_wave()

    def _update_towers(self):
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                tower_def = self.TOWER_TYPES[tower['type']]
                target = None
                min_dist = tower_def['range']
                for enemy in self.enemies:
                    dist = tower['pos'].distance_to(enemy['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    self.projectiles.append({
                        "pos": tower['pos'].copy(),
                        "type": tower['type'],
                        "target": target,
                        "speed": tower_def['proj_speed'],
                        "damage": tower_def['damage'],
                    })
                    tower['cooldown'] = tower_def['fire_rate']
                    # sfx: fire_cannon.wav or fire_missile.wav

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['waypoint_index'] >= len(self.path_waypoints):
                self.game_over = True
                # sfx: base_destroyed.wav
                return

            target_pos = self.path_waypoints[enemy['waypoint_index']]
            direction = (target_pos - enemy['pos'])
            
            if direction.length() < enemy['speed']:
                enemy['pos'] = target_pos
                enemy['waypoint_index'] += 1
            else:
                enemy['pos'] += direction.normalize() * enemy['speed']

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            direction = (proj['target']['pos'] - proj['pos'])
            if direction.length() < proj['speed']:
                proj['target']['health'] -= proj['damage']
                self._create_particles(proj['pos'], 5, self.COLOR_ENEMY)
                self.projectiles.remove(proj)
                # sfx: explosion.wav
                if proj['target']['health'] <= 0:
                    self._create_particles(proj['target']['pos'], 15, self.COLOR_ENEMY)
                    self.enemies.remove(proj['target'])
                    self.score += 5
                    self.resources += 20
                    self.frame_reward += 0.1
            else:
                proj['pos'] += direction.normalize() * proj['speed']
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(self.rng.uniform(-1, 1), self.rng.uniform(-1, 1)),
                'life': self.rng.integers(10, 20),
                'color': color
            })

    def _check_termination(self):
        return self.game_over or self.win or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 30)
        
        # Tower Slots and Cursor
        cursor_pos = self.tower_slots[self.cursor_index]['pos']
        for i, slot in enumerate(self.tower_slots):
            color = self.COLOR_CURSOR if i == self.cursor_index else self.COLOR_SLOT
            pygame.gfxdraw.filled_circle(self.screen, int(slot['pos'].x), int(slot['pos'].y), 10, (*color, 100))
            pygame.gfxdraw.aacircle(self.screen, int(slot['pos'].x), int(slot['pos'].y), 10, color)

        # Draw range indicator for selected tower at cursor
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        pygame.gfxdraw.aacircle(self.screen, int(cursor_pos.x), int(cursor_pos.y), tower_def['range'], (*self.COLOR_CURSOR, 50))

        # Towers
        for tower in self.towers:
            tower_def = self.TOWER_TYPES[tower['type']]
            pos = (int(tower['pos'].x), int(tower['pos'].y))
            pygame.draw.circle(self.screen, (50,50,50), pos, 12)
            pygame.draw.circle(self.screen, tower_def['color'], pos, 10)

        # Projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'].x), int(proj['pos'].y))
            pygame.draw.circle(self.screen, self.COLOR_CURSOR, pos, 3)

        # Enemies (render after projectiles to be on top)
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, 8)
            # Health bar
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            bar_w = 20
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (pos[0] - bar_w/2, pos[1] - 20, bar_w, 5))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (pos[0] - bar_w/2, pos[1] - 20, bar_w * health_pct, 5))

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 2, color)

    def _render_ui(self):
        # Top-left info
        wave_text = self.font_m.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        resources_text = self.font_m.render(f"Resources: ${self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(resources_text, (10, 40))

        # Top-right info
        score_text = self.font_m.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.screen_size[0] - score_text.get_width() - 10, 10))

        # Bottom-center info
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        tower_info = f"Next: {tower_def['name']} (${tower_def['cost']})"
        tower_text = self.font_m.render(tower_info, True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (self.screen_size[0]/2 - tower_text.get_width()/2, self.screen_size[1] - 40))
        
        # Wave timer
        if not self.wave_active and not self.win:
            timer_sec = math.ceil(self.wave_timer / 30)
            timer_text = self.font_l.render(f"Next wave in {timer_sec}", True, self.COLOR_TEXT)
            self.screen.blit(timer_text, (self.screen_size[0]/2 - timer_text.get_width()/2, self.screen_size[1]/2 - 50))

        # Game Over / Win message
        if self.game_over or self.win:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_l.render(message, True, color)
            self.screen.blit(end_text, (self.screen_size[0]/2 - end_text.get_width()/2, self.screen_size[1]/2 - end_text.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "enemies_left": len(self.enemies) + len(self.wave_spawner),
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Mapping from Pygame keys to action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated:
        # --- Human Input ---
        movement_action = 0 # No-op
        space_action = 0 # Released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        for key, action_val in key_map.items():
            if keys[key]:
                movement_action = action_val
                break # Prioritize first key found
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_ESCAPE]:
            terminated = True
            
        action = [movement_action, space_action, 0] # Shift is not used
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(30) # Run at 30 FPS
        pygame.display.set_caption(f"Score: {info['score']} | Wave: {info['wave']} | FPS: {clock.get_fps():.1f}")

    env.close()