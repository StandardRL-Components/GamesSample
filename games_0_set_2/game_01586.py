
# Generated: 2025-08-27T17:36:47.371682
# Source Brief: brief_01586.md
# Brief Index: 1586

        
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
        "Controls: Arrow keys to select a tower plot. Shift to cycle tower type. Space to build."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric tower defense game. Place towers to defend your base from waves of enemies."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 15000 # ~8 minutes
        self.MAX_WAVES = 15
        self.INITIAL_GOLD = 100
        self.INITIAL_BASE_HEALTH = 100
        self.WAVE_PREP_TIME = 300 # 10 seconds

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PATH = (40, 50, 70)
        self.COLOR_PLOT = (60, 75, 100)
        self.COLOR_PLOT_SELECTED = (255, 255, 0)
        self.COLOR_BASE = (0, 100, 200)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_HEALTH_BAR_BG = (70, 20, 20)
        self.COLOR_HEALTH_BAR_FG = (200, 40, 40)
        self.COLOR_PLAYER_HEALTH_FG = (40, 200, 40)
        
        # Tower Types
        self.TOWER_TYPES = [
            {
                "name": "Cannon", "cost": 50, "damage": 25, "range": 2.5, "fire_rate": 60,
                "color": (0, 150, 255), "proj_color": (150, 220, 255), "proj_speed": 5
            },
            {
                "name": "Missile", "cost": 120, "damage": 80, "range": 4.0, "fire_rate": 120,
                "color": (255, 100, 0), "proj_color": (255, 180, 100), "proj_speed": 4
            }
        ]
        
        # Game Grid & Path
        self.GRID_SIZE = (12, 12)
        self.TILE_WIDTH, self.TILE_HEIGHT = 60, 30
        self.ORIGIN = (self.WIDTH // 2, 60)
        self.ENEMY_PATH = [
            (-1, 5), (0, 5), (1, 5), (2, 5), (2, 4), (2, 3), (2, 2), (3, 2), (4, 2),
            (5, 2), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 7), (8, 7),
            (9, 7), (9, 6), (9, 5), (10, 5), (11, 5), (12, 5)
        ]
        self.BASE_POS = self.ENEMY_PATH[-1]
        self.TOWER_PLOTS = [
            (1, 3), (3, 4), (4, 4), (5, 4), (4, 0),
            (7, 5), (8, 5), (7, 9), (8, 9), (10, 7)
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.base_health = 0
        self.gold = 0
        self.current_wave = 0
        self.wave_timer = 0
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.selected_plot_idx = 0
        self.selected_tower_idx = 0
        self.prev_shift_held = False
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.base_health = self.INITIAL_BASE_HEALTH
        self.gold = self.INITIAL_GOLD
        self.current_wave = 0
        self.wave_timer = self.WAVE_PREP_TIME
        self.enemies_in_wave = 0
        self.enemies_spawned = 0
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.selected_plot_idx = 0
        self.selected_tower_idx = 0
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        step_reward = 0
        
        self._handle_actions(action)
        step_reward += self._update_game_state()

        self.score += step_reward
        self.steps += 1
        
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.victory:
                self.score += 100
                step_reward += 100
            else: # Loss by health or timeout
                self.score -= 100
                step_reward -= 100
            self.game_over = True

        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement: Select tower plot
        if movement == 1: self.selected_plot_idx = (self.selected_plot_idx + 1) % len(self.TOWER_PLOTS)
        if movement == 2: self.selected_plot_idx = (self.selected_plot_idx - 1 + len(self.TOWER_PLOTS)) % len(self.TOWER_PLOTS)
        
        # Shift: Cycle tower type (on press, not hold)
        if shift_held and not self.prev_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.TOWER_TYPES)
        self.prev_shift_held = shift_held

        # Space: Place tower
        if space_held:
            plot_pos = self.TOWER_PLOTS[self.selected_plot_idx]
            tower_type = self.TOWER_TYPES[self.selected_tower_idx]
            is_occupied = any(t['pos'] == plot_pos for t in self.towers)

            if self.gold >= tower_type['cost'] and not is_occupied:
                self.gold -= tower_type['cost']
                self.towers.append({
                    "pos": plot_pos, "type_idx": self.selected_tower_idx,
                    "cooldown": 0, "target": None
                })
                # sfx: build_tower

    def _update_game_state(self):
        reward = 0
        
        # Wave Management
        self.wave_timer -= 1
        if self.wave_timer <= 0 and self.current_wave < self.MAX_WAVES:
            if self.enemies_spawned == 0 and not self.enemies: # New wave starts
                self.current_wave += 1
                self.enemies_in_wave = 2 + self.current_wave * 2
                self.enemies_spawned = 0
                self.wave_timer = self.rng.integers(15, 30) # time between enemies
            
            if self.enemies_spawned < self.enemies_in_wave:
                self._spawn_enemy()
                self.enemies_spawned += 1
                self.wave_timer = self.rng.integers(15, 45 - self.current_wave)
            
            elif not self.enemies: # Wave cleared
                reward += 5
                gold_bonus = 25 + self.current_wave * 5
                self.gold += gold_bonus
                reward += gold_bonus * 0.1
                self.wave_timer = self.WAVE_PREP_TIME
                if self.current_wave >= self.MAX_WAVES:
                    self.victory = True

        reward += self._update_towers()
        reward += self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        return reward

    def _spawn_enemy(self):
        health = 40 + (self.current_wave - 1) * 20 * (1.1 ** (self.current_wave - 1))
        speed = 0.02 + (self.current_wave - 1) * 0.002
        self.enemies.append({
            "path_idx": 0, "progress": 0.0, "max_health": health, "health": health,
            "speed": speed, "pos": self.ENEMY_PATH[0],
            "screen_pos": self._iso_transform(*self.ENEMY_PATH[0])
        })

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_TYPES[tower['type_idx']]
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            
            # Find new target if needed
            if tower['target'] is None or tower['target'] not in self.enemies:
                tower['target'] = None
                in_range_enemies = []
                for enemy in self.enemies:
                    dist = math.hypot(tower['pos'][0] - enemy['pos'][0], tower['pos'][1] - enemy['pos'][1])
                    if dist <= spec['range']:
                        in_range_enemies.append(enemy)
                if in_range_enemies:
                    tower['target'] = max(in_range_enemies, key=lambda e: e['path_idx'] + e['progress'])

            # Fire if ready and has target
            if tower['cooldown'] == 0 and tower['target'] is not None:
                tower['cooldown'] = spec['fire_rate']
                self.projectiles.append({
                    "start_pos": self._iso_transform(*tower['pos'], z=20),
                    "target_enemy": tower['target'], "type_idx": tower['type_idx']
                })
                # sfx: fire_cannon or fire_missile

        return 0

    def _update_enemies(self):
        reward = 0
        for enemy in reversed(self.enemies):
            enemy['progress'] += enemy['speed']
            
            if enemy['progress'] >= 1.0:
                enemy['progress'] = 0.0
                enemy['path_idx'] += 1
                if enemy['path_idx'] >= len(self.ENEMY_PATH) - 1:
                    self.enemies.remove(enemy)
                    self.base_health = max(0, self.base_health - 10)
                    reward -= 1.0 # Larger penalty for reaching base
                    # sfx: base_hit
                    self._create_particles(self._iso_transform(*self.BASE_POS), (255, 0, 0), 20)
                    continue

            p1 = self.ENEMY_PATH[enemy['path_idx']]
            p2 = self.ENEMY_PATH[enemy['path_idx'] + 1]
            enemy['pos'] = (
                p1[0] + (p2[0] - p1[0]) * enemy['progress'],
                p1[1] + (p2[1] - p1[1]) * enemy['progress']
            )
            enemy['screen_pos'] = self._iso_transform(*enemy['pos'])
        return reward

    def _update_projectiles(self):
        for proj in reversed(self.projectiles):
            spec = self.TOWER_TYPES[proj['type_idx']]
            target_enemy = proj['target_enemy']
            
            if target_enemy not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = target_enemy['screen_pos'][0], target_enemy['screen_pos'][1] - 15
            start_pos = proj['start_pos']
            
            dx, dy = target_pos[0] - start_pos[0], target_pos[1] - start_pos[1]
            dist = math.hypot(dx, dy)

            if dist < spec['proj_speed']:
                target_enemy['health'] -= spec['damage']
                self._create_particles(target_pos, spec['proj_color'], 10)
                self.projectiles.remove(proj)
                # sfx: enemy_hit
            else:
                proj['start_pos'] = (
                    start_pos[0] + (dx / dist) * spec['proj_speed'],
                    start_pos[1] + (dy / dist) * spec['proj_speed']
                )

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['vel'] = (p['vel'][0] * 0.95, p['vel'][1] * 0.95 + 0.1) # friction and gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.victory:
            return True
        if self.base_health <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game_world()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game_world(self):
        # Draw path and plots first
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                if (x, y) in self.ENEMY_PATH:
                    self._draw_iso_tile((x, y), self.COLOR_PATH)
                if (x, y) in self.TOWER_PLOTS:
                    is_selected = (x, y) == self.TOWER_PLOTS[self.selected_plot_idx]
                    color = self.COLOR_PLOT_SELECTED if is_selected else self.COLOR_PLOT
                    self._draw_iso_tile((x, y), color, border_color=self.COLOR_BG)

        # Draw base
        self._draw_iso_cube(self.BASE_POS, (2, 2, 2), self.COLOR_BASE)
        
        # Collect and sort all dynamic objects for painter's algorithm
        render_queue = []
        for tower in self.towers:
            render_queue.append({'type': 'tower', 'obj': tower})
        for enemy in self.enemies:
            render_queue.append({'type': 'enemy', 'obj': enemy})
        
        # Sort by y-coordinate (grid y + x for isometric depth)
        render_queue.sort(key=lambda item: item['obj']['pos'][0] + item['obj']['pos'][1])

        # Render sorted objects
        for item in render_queue:
            if item['type'] == 'tower':
                self._render_tower(item['obj'])
            elif item['type'] == 'enemy':
                self._render_enemy(item['obj'])
        
        # Render projectiles and particles on top
        for proj in self.projectiles:
            spec = self.TOWER_TYPES[proj['type_idx']]
            pos = (int(proj['start_pos'][0]), int(proj['start_pos'][1]))
            pygame.draw.circle(self.screen, spec['proj_color'], pos, 4)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, spec['proj_color'])

        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

    def _render_tower(self, tower):
        spec = self.TOWER_TYPES[tower['type_idx']]
        self._draw_iso_cube(tower['pos'], (1, 1, 1.5), spec['color'])

    def _render_enemy(self, enemy):
        screen_pos = enemy['screen_pos']
        x, y = int(screen_pos[0]), int(screen_pos[1])
        
        # Body
        self._draw_iso_cube(enemy['pos'], (0.6, 0.6, 0.4), self.COLOR_HEALTH_BAR_FG)
        
        # Health bar
        bar_w, bar_h = 30, 4
        health_pct = enemy['health'] / enemy['max_health']
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (x - bar_w // 2, y - 25, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (x - bar_w // 2, y - 25, int(bar_w * health_pct), bar_h))

    def _render_ui(self):
        # Gold
        gold_text = self.font_large.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (10, 10))

        # Base Health
        health_text = self.font_large.render("BASE HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH - health_text.get_width() - 10, 10))
        health_pct = self.base_health / self.INITIAL_BASE_HEALTH
        bar_w, bar_h = health_text.get_width(), 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (self.WIDTH - bar_w - 10, 35, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_HEALTH_FG, (self.WIDTH - bar_w - 10, 35, int(bar_w * health_pct), bar_h))

        # Wave Info
        if self.wave_timer > 0 and self.enemies_spawned == 0:
            wave_info = f"WAVE {self.current_wave + 1} IN {self.wave_timer / self.FPS:.1f}s"
        else:
            wave_info = f"WAVE {self.current_wave}/{self.MAX_WAVES}"
        wave_text = self.font_large.render(wave_info, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH // 2 - wave_text.get_width() // 2, self.HEIGHT - 40))

        # Tower Selection UI
        ts = self.TOWER_TYPES[self.selected_tower_idx]
        name_text = self.font_large.render(f"{ts['name']}", True, ts['color'])
        cost_text = self.font_small.render(f"Cost: {ts['cost']}", True, self.COLOR_GOLD)
        dmg_text = self.font_small.render(f"Dmg: {ts['damage']}", True, self.COLOR_TEXT)
        rng_text = self.font_small.render(f"Rng: {ts['range']}", True, self.COLOR_TEXT)
        
        ui_x = self.WIDTH - 150
        self.screen.blit(name_text, (ui_x, self.HEIGHT - 100))
        self.screen.blit(cost_text, (ui_x, self.HEIGHT - 75))
        self.screen.blit(dmg_text, (ui_x, self.HEIGHT - 60))
        self.screen.blit(rng_text, (ui_x, self.HEIGHT - 45))
        
        # Game Over/Victory Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.victory else "GAME OVER"
            msg_text = self.font_huge.render(msg, True, self.COLOR_GOLD if self.victory else self.COLOR_HEALTH_BAR_FG)
            self.screen.blit(msg_text, (self.WIDTH // 2 - msg_text.get_width() // 2, self.HEIGHT // 2 - msg_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.current_wave,
            "base_health": self.base_health,
        }

    # --- Helper Functions ---
    def _iso_transform(self, x, y, z=0):
        screen_x = self.ORIGIN[0] + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = self.ORIGIN[1] + (x + y) * (self.TILE_HEIGHT / 2) - z
        return screen_x, screen_y

    def _draw_iso_tile(self, pos, color, border_color=None):
        x, y = pos
        points = [
            self._iso_transform(x, y),
            self._iso_transform(x + 1, y),
            self._iso_transform(x + 1, y + 1),
            self._iso_transform(x, y + 1)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        if border_color:
            pygame.gfxdraw.aapolygon(self.screen, points, border_color)

    def _draw_iso_cube(self, pos, size, color):
        x, y = pos
        w, d, h = size
        h_px = h * self.TILE_HEIGHT
        
        top_points = [
            self._iso_transform(x, y, z=h_px),
            self._iso_transform(x + w, y, z=h_px),
            self._iso_transform(x + w, y + d, z=h_px),
            self._iso_transform(x, y + d, z=h_px)
        ]
        
        c_light = tuple(min(255, int(c * 1.1)) for c in color)
        c_dark = tuple(max(0, int(c * 0.7)) for c in color)

        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)
        
        # Left face
        left_points = [
            self._iso_transform(x, y), self._iso_transform(x, y, z=h_px),
            self._iso_transform(x, y + d, z=h_px), self._iso_transform(x, y + d)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, left_points, c_dark)

        # Right face
        right_points = [
            self._iso_transform(x, y), self._iso_transform(x, y, z=h_px),
            self._iso_transform(x + w, y, z=h_px), self._iso_transform(x + w, y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, right_points, c_light)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = 1 + self.rng.random() * 3
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.rng.integers(15, 30),
                "max_life": 30,
                "color": color
            })

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Player Controls ---
    # This maps keyboard keys to the MultiDiscrete action space
    key_to_action = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0], # Not used in this mapping, but kept for consistency
        pygame.K_RIGHT: [4, 0, 0], # Not used in this mapping, but kept for consistency
        pygame.K_SPACE: [0, 1, 0],
        pygame.K_LSHIFT: [0, 0, 1],
        pygame.K_RSHIFT: [0, 0, 1],
    }

    # --- Game Loop ---
    obs, info = env.reset()
    terminated = False
    
    # We need a separate pygame screen for human display
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running and not terminated:
        # Default action is no-op
        action = np.array([0, 0, 0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Aggregate held keys for MultiDiscrete action
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1 # Cycle forward
        if keys[pygame.K_DOWN]: action[0] = 2 # Cycle backward
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Gold: {info['gold']}, Wave: {info['wave']}")

        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)
        
        if terminated:
            print("Game Over!")
            print(f"Final Score: {info['score']:.2f}, Survived {info['wave']-1} waves.")
            pygame.time.wait(3000) # Pause for 3 seconds before closing

    env.close()