
# Generated: 2025-08-27T18:26:08.870046
# Source Brief: brief_01826.md
# Brief Index: 1826

        
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
        "Controls: Arrows to move cursor, Shift to cycle tower types, Space to build."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this isometric tower defense game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)


        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_PATH = (45, 50, 62)
        self.COLOR_PATH_BORDER = (65, 72, 89)
        self.COLOR_BASE = (60, 179, 113)
        self.COLOR_BASE_DMG = (255, 99, 71)
        self.COLOR_SPOT = (80, 88, 105, 100)
        self.COLOR_CURSOR = (255, 215, 0)
        self.COLOR_ENEMY = (220, 20, 60)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_UI_BG = (40, 42, 54, 200)

        # Isometric projection settings
        self.TILE_W_HALF = 32
        self.TILE_H_HALF = 16
        self.ORIGIN_X = self.screen_width // 2
        self.ORIGIN_Y = 80

        # Game Layout
        self._define_layout()

        # Tower definitions
        self.TOWER_TYPES = [
            {"name": "Cannon", "cost": 100, "range": 100, "cooldown": 60, "damage": 25, "color": (0, 191, 255)},
            {"name": "Sniper", "cost": 250, "range": 200, "cooldown": 120, "damage": 100, "color": (147, 112, 219)},
        ]

        # Action state tracking
        self.prev_movement = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()
        self.validate_implementation()


    def _define_layout(self):
        self.path_iso = [
            (0, -2), (1, -2), (2, -2), (3, -2), (3, -1), (3, 0), (3, 1),
            (2, 1), (1, 1), (0, 1), (-1, 1), (-2, 1), (-2, 2), (-2, 3),
            (-2, 4), (-1, 4), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4)
        ]
        self.path_screen = [self._iso_to_screen(p[0], p[1]) for p in self.path_iso]

        self.tower_spots_iso = [
            (2, -1), (4, -1), (2, 0), (4, 0), (1, 2), (0, 2), (-1, 2),
            (-3, 2), (-3, 3), (-1, 3), (0, 3), (1, 3), (2, 3)
        ]
        self.tower_spots_screen = [self._iso_to_screen(p[0], p[1]) for p in self.tower_spots_iso]
        self.base_pos_screen = self._iso_to_screen(6, 4)

    def _iso_to_screen(self, iso_x, iso_y):
        screen_x = self.ORIGIN_X + (iso_x - iso_y) * self.TILE_W_HALF
        screen_y = self.ORIGIN_Y + (iso_x + iso_y) * self.TILE_H_HALF
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.step_reward = 0.0

        self.base_health = 100
        self.resources = 250
        self.wave_number = 0
        self.wave_in_progress = False
        self.time_to_next_wave = 150 # 5 seconds at 30fps

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.spawn_queue = []

        self.cursor_pos_index = 0
        self.selected_tower_type = 0

        self.prev_movement = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.step_reward = 0.0
        self.steps += 1

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        if not self.game_over:
            self._handle_input(movement, space_held, shift_held)
            self._update_game_state()

        terminated = self._check_termination()
        reward = self.step_reward

        if terminated:
            if self.game_won:
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

    def _handle_input(self, movement, space_held, shift_held):
        # Handle cursor movement (on change, not hold)
        if movement != 0 and movement != self.prev_movement:
            if movement == 1: # Up
                self.cursor_pos_index = (self.cursor_pos_index - 1) % len(self.tower_spots_screen)
            elif movement == 2: # Down
                self.cursor_pos_index = (self.cursor_pos_index + 1) % len(self.tower_spots_screen)
            elif movement == 3: # Left
                self.cursor_pos_index = (self.cursor_pos_index - 1) % len(self.tower_spots_screen)
            elif movement == 4: # Right
                self.cursor_pos_index = (self.cursor_pos_index + 1) % len(self.tower_spots_screen)
        self.prev_movement = movement

        # Handle tower type cycling (on press)
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_TYPES)
        self.prev_shift_held = shift_held

        # Handle tower placement (on press)
        if space_held and not self.prev_space_held:
            self._place_tower()
        self.prev_space_held = space_held

    def _place_tower(self):
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        if self.resources >= tower_def["cost"]:
            pos = self.tower_spots_screen[self.cursor_pos_index]
            is_occupied = any(t['pos'] == pos for t in self.towers)
            if not is_occupied:
                self.resources -= tower_def["cost"]
                self.towers.append({
                    "pos": pos,
                    "type_idx": self.selected_tower_type,
                    "cooldown_timer": 0,
                    "target": None,
                    **tower_def
                })
                # sfx: build_tower

    def _update_game_state(self):
        self._update_waves()
        self._update_towers()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()

    def _update_waves(self):
        if not self.wave_in_progress and not self.spawn_queue:
            self.time_to_next_wave -= 1
            if self.time_to_next_wave <= 0:
                self.wave_number += 1
                if self.wave_number > 15:
                    self.game_won = True
                    self.game_over = True
                    return
                self.step_reward += 1.0
                self._start_wave()
                self.wave_in_progress = True

        if self.spawn_queue:
            self.spawn_queue[0]['timer'] -= 1
            if self.spawn_queue[0]['timer'] <= 0:
                self.enemies.append(self.spawn_queue.pop(0)['enemy'])
                # sfx: enemy_spawn

        if self.wave_in_progress and not self.enemies and not self.spawn_queue:
            self.wave_in_progress = False
            self.time_to_next_wave = 300 # 10 seconds

    def _start_wave(self):
        num_enemies = 2 + self.wave_number
        base_health = 20 + self.wave_number * 5
        base_speed = 0.8 + self.wave_number * 0.05
        for i in range(num_enemies):
            enemy_health = base_health * (1 + self.np_random.uniform(-0.1, 0.1))
            enemy_speed = base_speed * (1 + self.np_random.uniform(-0.1, 0.1))
            self.spawn_queue.append({
                'timer': i * 30, # spawn delay
                'enemy': {
                    'pos': list(self.path_screen[0]),
                    'path_idx': 1,
                    'health': enemy_health,
                    'max_health': enemy_health,
                    'speed': enemy_speed,
                    'id': self.np_random.integers(1_000_000),
                }
            })

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown_timer'] > 0:
                tower['cooldown_timer'] -= 1
                continue

            # Retarget if current target is gone or out of range
            if tower['target']:
                target_enemy = next((e for e in self.enemies if e['id'] == tower['target']), None)
                if not target_enemy or math.dist(tower['pos'], target_enemy['pos']) > tower['range']:
                    tower['target'] = None
            
            # Find a new target if needed
            if not tower['target']:
                in_range_enemies = [e for e in self.enemies if math.dist(tower['pos'], e['pos']) <= tower['range']]
                if in_range_enemies:
                    # Target enemy furthest along the path
                    in_range_enemies.sort(key=lambda e: e['path_idx'], reverse=True)
                    tower['target'] = in_range_enemies[0]['id']

            # Fire if target is valid and cooldown is ready
            if tower['target']:
                target_enemy = next((e for e in self.enemies if e['id'] == tower['target']), None)
                if target_enemy and tower['cooldown_timer'] <= 0:
                    self.projectiles.append({
                        'pos': list(tower['pos']),
                        'target_id': tower['target'],
                        'speed': 8,
                        'damage': tower['damage']
                    })
                    tower['cooldown_timer'] = tower['cooldown']
                    # sfx: tower_shoot

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['path_idx'] >= len(self.path_screen):
                self.base_health -= 10
                self._create_particles(enemy['pos'], self.COLOR_BASE_DMG, 20)
                self.enemies.remove(enemy)
                # sfx: base_damage
                continue

            target_pos = self.path_screen[enemy['path_idx']]
            direction = (target_pos[0] - enemy['pos'][0], target_pos[1] - enemy['pos'][1])
            distance = math.hypot(*direction)

            if distance < enemy['speed']:
                enemy['pos'] = list(target_pos)
                enemy['path_idx'] += 1
            else:
                norm_dir = (direction[0] / distance, direction[1] / distance)
                enemy['pos'][0] += norm_dir[0] * enemy['speed']
                enemy['pos'][1] += norm_dir[1] * enemy['speed']

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            target_enemy = next((e for e in self.enemies if e['id'] == p['target_id']), None)
            if not target_enemy:
                self.projectiles.remove(p)
                continue

            target_pos = target_enemy['pos']
            direction = (target_pos[0] - p['pos'][0], target_pos[1] - p['pos'][1])
            distance = math.hypot(*direction)

            if distance < p['speed']:
                target_enemy['health'] -= p['damage']
                self._create_particles(p['pos'], self.COLOR_PROJECTILE, 10)
                self.projectiles.remove(p)
                # sfx: enemy_hit
                if target_enemy['health'] <= 0:
                    self.score += 10
                    self.resources += 20
                    self.step_reward += 0.1
                    self._create_particles(target_enemy['pos'], self.COLOR_ENEMY, 30)
                    self.enemies.remove(target_enemy)
                    # sfx: enemy_die
            else:
                norm_dir = (direction[0] / distance, direction[1] / distance)
                p['pos'][0] += norm_dir[0] * p['speed']
                p['pos'][1] += norm_dir[1] * p['speed']

    def _update_particles(self):
        for particle in self.particles[:]:
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['lifespan'] -= 1
            if particle['lifespan'] <= 0:
                self.particles.remove(particle)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'lifespan': self.np_random.integers(10, 20)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_layout()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_layout(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_screen, width=self.TILE_H_HALF * 2)
        pygame.draw.lines(self.screen, self.COLOR_PATH_BORDER, False, self.path_screen, width=1)
        # Base
        base_color = self.COLOR_BASE if self.base_health > 30 else self.COLOR_BASE_DMG
        pygame.gfxdraw.filled_circle(self.screen, self.base_pos_screen[0], self.base_pos_screen[1], 16, base_color)
        pygame.gfxdraw.aacircle(self.screen, self.base_pos_screen[0], self.base_pos_screen[1], 16, self.COLOR_PATH_BORDER)
        # Tower spots
        for pos in self.tower_spots_screen:
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_SPOT)

    def _render_towers(self):
        for tower in self.towers:
            pygame.gfxdraw.filled_circle(self.screen, tower['pos'][0], tower['pos'][1], 10, tower['color'])
            pygame.gfxdraw.aacircle(self.screen, tower['pos'][0], tower['pos'][1], 10, (255,255,255))
            # Cooldown indicator
            if tower['cooldown_timer'] > 0:
                angle = 360 * (tower['cooldown_timer'] / tower['cooldown'])
                rect = pygame.Rect(tower['pos'][0] - 10, tower['pos'][1] - 10, 20, 20)
                pygame.draw.arc(self.screen, (255,255,255,100), rect, 0, math.radians(angle), 3)

    def _render_enemies(self):
        for enemy in self.enemies:
            x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 7, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, x, y, 7, (255,255,255))
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, (255,0,0), (x - 10, y - 15, 20, 3))
            pygame.draw.rect(self.screen, (0,255,0), (x - 10, y - 15, int(20 * health_ratio), 3))

    def _render_projectiles(self):
        for p in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(p['pos'][0]), int(p['pos'][1])), 4)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 20.0))
            color = p['color'] + (alpha,)
            size = int(max(1, 4 * (p['lifespan'] / 20.0)))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_cursor(self):
        pos = self.tower_spots_screen[self.cursor_pos_index]
        # Pulsing glow effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        radius = int(12 + pulse * 4)
        alpha = int(100 + pulse * 100)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_CURSOR + (alpha,))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_CURSOR)
        
    def _render_ui(self):
        # Info Panel
        panel_rect = pygame.Rect(5, 5, 220, 95)
        pygame.gfxdraw.box(self.screen, panel_rect, self.COLOR_UI_BG)
        
        # Text rendering helper
        def draw_text(text, pos, font=self.font_small, color=self.COLOR_UI_TEXT):
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        draw_text(f"SCORE:    {self.score:06d}", (15, 10))
        draw_text(f"RESOURCES:{self.resources:06d}", (15, 30))
        draw_text(f"BASE HP:  {max(0, self.base_health)}%", (15, 50))
        
        wave_text = f"WAVE {self.wave_number}/15"
        if not self.wave_in_progress and self.wave_number < 15 and not self.game_over:
            wave_text += f" (in {self.time_to_next_wave//30 + 1}s)"
        draw_text(wave_text, (15, 70))

        # Selected Tower Panel
        tower_panel_rect = pygame.Rect(self.screen_width - 205, 5, 200, 75)
        pygame.gfxdraw.box(self.screen, tower_panel_rect, self.COLOR_UI_BG)
        tower_def = self.TOWER_TYPES[self.selected_tower_type]
        draw_text("SELECTED TOWER", (self.screen_width - 195, 10), color=self.COLOR_CURSOR)
        draw_text(f"{tower_def['name'].upper()}", (self.screen_width - 195, 30), font=self.font_large, color=tower_def['color'])
        draw_text(f"Cost: {tower_def['cost']}", (self.screen_width - 195, 55))

    def _render_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message = "VICTORY!" if self.game_won else "GAME OVER"
        text_surf = self.font_huge.render(message, True, self.COLOR_CURSOR)
        text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.steps >= 10000: # Increased from 1000 to allow for a full game
            self.game_over = True
            return True
        return False

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Isometric Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        # Display user guide
        guide_surf = env.font_small.render(env.user_guide, True, env.COLOR_UI_TEXT)
        screen.blit(guide_surf, (5, env.screen_height - 20))

        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS

    env.close()