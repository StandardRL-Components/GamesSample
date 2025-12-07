
# Generated: 2025-08-28T06:25:58.441249
# Source Brief: brief_05900.md
# Brief Index: 5900

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the placement cursor. Press space to build a tower at the cursor's location."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in this isometric tower defense game. Survive all 15 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 12
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 20, 10
    MAX_STEPS = 20000 # Increased for longer gameplay
    MAX_WAVES = 15

    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 50, 60)
    COLOR_PATH = (60, 70, 80)
    COLOR_BASE = (0, 150, 50)
    COLOR_BASE_DMG = (200, 50, 50)
    COLOR_TOWER = (50, 150, 255)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_PROJECTILE = (255, 200, 0)
    COLOR_CURSOR_VALID = (255, 255, 255, 100)
    COLOR_CURSOR_INVALID = (255, 0, 0, 100)
    COLOR_TEXT = (230, 230, 230)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_HEALTH_BAR_FG = (0, 200, 0)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_huge = pygame.font.SysFont("monospace", 48, bold=True)

        self.iso_offset_x = self.SCREEN_WIDTH // 2
        self.iso_offset_y = 100

        self.path_grid_coords = self._define_path()
        self.path_pixel_coords = [self._to_iso(x, y) for x, y in self.path_grid_coords]
        self.path_lengths = self._calculate_path_lengths()
        
        self.rng = np.random.default_rng()

        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def _define_path(self):
        path = []
        for x in range(self.GRID_WIDTH):
            path.append((x, 2))
        for y in range(3, self.GRID_HEIGHT - 2):
            path.append((self.GRID_WIDTH - 1, y))
        for x in range(self.GRID_WIDTH - 2, -1, -1):
            path.append((x, self.GRID_HEIGHT - 3))
        return path

    def _calculate_path_lengths(self):
        lengths = [0]
        total_length = 0
        for i in range(1, len(self.path_pixel_coords)):
            p1 = self.path_pixel_coords[i-1]
            p2 = self.path_pixel_coords[i]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            total_length += dist
            lengths.append(total_length)
        return lengths

    def _get_pos_on_path(self, path_dist):
        total_path_length = self.path_lengths[-1]
        dist = path_dist * total_path_length

        if dist <= 0: return self.path_pixel_coords[0]
        if dist >= total_path_length: return self.path_pixel_coords[-1]

        for i in range(1, len(self.path_lengths)):
            if dist <= self.path_lengths[i]:
                segment_start_dist = self.path_lengths[i-1]
                segment_end_dist = self.path_lengths[i]
                segment_len = segment_end_dist - segment_start_dist
                
                p1 = self.path_pixel_coords[i-1]
                p2 = self.path_pixel_coords[i]
                
                interp = (dist - segment_start_dist) / segment_len
                x = p1[0] + (p2[0] - p1[0]) * interp
                y = p1[1] + (p2[1] - p1[1]) * interp
                return (x, y)
        return self.path_pixel_coords[-1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = 100
        self.towers_remaining = 12
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 0
        self.wave_state = "INTER_WAVE" # INTER_WAVE, WAVE_IN_PROGRESS
        self.inter_wave_timer = 90 # 3 seconds at 30fps
        self.enemies_to_spawn_in_wave = 0
        self.spawn_timer = 0
        
        self.last_space_held = False
        
        self.base_grid_pos = self.path_grid_coords[-1]

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Handle Input ---
        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        
        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Place tower action
        is_valid_spot = self._is_valid_tower_spot(self.cursor_pos)
        if space_action and not self.last_space_held and self.towers_remaining > 0 and is_valid_spot:
            # sfx: place_tower.wav
            self.towers.append({
                "pos": list(self.cursor_pos),
                "cooldown": 0,
                "range_sq": 10000, # squared range for efficiency
            })
            self.towers_remaining -= 1
        self.last_space_held = space_action

        # --- Update Game State ---
        self._update_waves()
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # Small penalty for inaction
        if self.towers_remaining > 0 and movement == 0 and not space_action:
            reward -= 0.001

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        if self.game_over:
            if self.win:
                reward += 100 # Win bonus
            else:
                reward -= 10 # Lose penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_valid_tower_spot(self, pos):
        if tuple(pos) in self.path_grid_coords: return False
        for tower in self.towers:
            if tower["pos"] == pos: return False
        return True

    def _update_waves(self):
        if self.wave_state == "INTER_WAVE" and not self.game_over:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0:
                self.current_wave += 1
                if self.current_wave > self.MAX_WAVES:
                    self.win = True
                    self.game_over = True
                    return

                self.wave_state = "WAVE_IN_PROGRESS"
                self.enemies_to_spawn_in_wave = 5 + self.current_wave * 2
                self.spawn_timer = 0
                # sfx: wave_start.wav
        
        elif self.wave_state == "WAVE_IN_PROGRESS":
            if self.enemies_to_spawn_in_wave > 0:
                self.spawn_timer -= 1
                if self.spawn_timer <= 0:
                    self._spawn_enemy()
                    self.enemies_to_spawn_in_wave -= 1
                    self.spawn_timer = 30 # Spawn every 1 second
            
            elif len(self.enemies) == 0:
                self.wave_state = "INTER_WAVE"
                self.inter_wave_timer = 150 # 5 seconds
                self.score += self.current_wave * 10
                # sfx: wave_cleared.wav

    def _spawn_enemy(self):
        speed = 0.002 + self.current_wave * 0.0002
        health = 3 + self.current_wave
        self.enemies.append({
            "path_dist": 0.0,
            "health": health,
            "max_health": health,
            "speed": speed,
            "pixel_pos": self.path_pixel_coords[0],
            "id": self.rng.integers(1, 1_000_000)
        })

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = self._find_target(tower)
                if target:
                    # sfx: tower_shoot.wav
                    self.projectiles.append({
                        "start_pos": self._to_iso(*tower['pos']),
                        "target_enemy": target,
                        "progress": 0.0,
                        "speed": 0.1
                    })
                    tower['cooldown'] = 45 # 1.5 second cooldown
        return reward

    def _find_target(self, tower):
        tower_pos = self._to_iso(*tower['pos'])
        best_target = None
        max_dist = -1 # Find enemy furthest along the path
        for enemy in self.enemies:
            dist_sq = (enemy['pixel_pos'][0] - tower_pos[0])**2 + (enemy['pixel_pos'][1] - tower_pos[1])**2
            if dist_sq <= tower['range_sq']:
                if enemy['path_dist'] > max_dist:
                    max_dist = enemy['path_dist']
                    best_target = enemy
        return best_target

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            proj['progress'] += proj['speed']
            if proj['progress'] >= 1.0 or proj['target_enemy']['health'] <= 0:
                if proj['target_enemy']['health'] > 0:
                    # sfx: projectile_hit.wav
                    proj['target_enemy']['health'] -= 1
                    reward += 0.1 # Hit reward
                    self._create_hit_particles(proj['target_enemy']['pixel_pos'])
                self.projectiles.remove(proj)
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            enemy['path_dist'] += enemy['speed']
            enemy['pixel_pos'] = self._get_pos_on_path(enemy['path_dist'])
            
            if enemy['health'] <= 0:
                # sfx: enemy_destroy.wav
                self.score += 5
                reward += 1 # Kill reward
                self._create_explosion_particles(enemy['pixel_pos'])
                self.enemies.remove(enemy)
                continue

            if enemy['path_dist'] >= 1.0:
                # sfx: base_damage.wav
                self.base_health -= 10
                self.enemies.remove(enemy)
                if self.base_health <= 0:
                    self.base_health = 0
                    self.game_over = True

        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion_particles(self, pos):
        for _ in range(15):
            angle = self.rng.random() * 2 * math.pi
            speed = 1 + self.rng.random() * 2
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.rng.integers(10, 20),
                "color": self.COLOR_ENEMY,
                "radius": self.rng.integers(2, 4)
            })

    def _create_hit_particles(self, pos):
        for _ in range(5):
            angle = self.rng.random() * 2 * math.pi
            speed = 0.5 + self.rng.random() * 1.5
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.rng.integers(5, 10),
                "color": self.COLOR_PROJECTILE,
                "radius": self.rng.integers(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "towers_remaining": self.towers_remaining,
        }

    def _to_iso(self, x, y):
        sx = (x - y) * self.TILE_WIDTH_HALF + self.iso_offset_x
        sy = (x + y) * self.TILE_HEIGHT_HALF + self.iso_offset_y
        return int(sx), int(sy)

    def _draw_iso_poly(self, surface, color, points, offset=(0,0)):
        iso_points = [self._to_iso(p[0]+offset[0], p[1]+offset[1]) for p in points]
        pygame.gfxdraw.aapolygon(surface, iso_points, color)
        pygame.gfxdraw.filled_polygon(surface, iso_points, color)

    def _draw_iso_tile(self, surface, color, pos):
        x, y = pos
        points = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        iso_points = [self._to_iso(p[0], p[1]) for p in points]
        pygame.gfxdraw.filled_polygon(surface, iso_points, color)
        pygame.gfxdraw.aapolygon(surface, iso_points, (0,0,0,30))

    def _render_game(self):
        # Draw grid and path
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color = self.COLOR_PATH if (x,y) in self.path_grid_coords else self.COLOR_GRID
                self._draw_iso_tile(self.screen, color, (x,y))

        # Draw base
        base_pos_px = self._to_iso(*self.base_grid_pos)
        base_points = [
            (self.base_grid_pos[0], self.base_grid_pos[1]),
            (self.base_grid_pos[0] + 1, self.base_grid_pos[1]),
            (self.base_grid_pos[0] + 1, self.base_grid_pos[1] + 1),
            (self.base_grid_pos[0], self.base_grid_pos[1] + 1)
        ]
        self._draw_iso_poly(self.screen, self.COLOR_BASE, base_points)
        
        # Draw towers
        for tower in self.towers:
            x, y = tower['pos']
            tower_points = [(x+0.5, y+0.1), (x+0.1, y+0.9), (x+0.9, y+0.9)]
            self._draw_iso_poly(self.screen, self.COLOR_TOWER, tower_points)

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pixel_pos'][0]), int(enemy['pixel_pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_ENEMY)
            
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_w = 10
            bar_h = 2
            bar_x = pos[0] - bar_w // 2
            bar_y = pos[1] - 12
            pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, int(bar_w * health_ratio), bar_h))


        # Draw projectiles
        for proj in self.projectiles:
            start_pos = proj['start_pos']
            if proj['target_enemy']['health'] <= 0: continue
            target_pos = proj['target_enemy']['pixel_pos']
            
            x = start_pos[0] + (target_pos[0] - start_pos[0]) * proj['progress']
            y = start_pos[1] + (target_pos[1] - start_pos[1]) * proj['progress']
            
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 3, self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life'])) if 'max_life' in p else 255
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)
            
        # Draw cursor
        is_valid = self._is_valid_tower_spot(self.cursor_pos)
        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        self._draw_iso_tile(self.screen, cursor_color, self.cursor_pos)

    def _render_ui(self):
        # Score and Wave
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Base Health
        base_pos_px = self._to_iso(*self.base_grid_pos)
        bar_w, bar_h = 80, 10
        bar_x = base_pos_px[0] - bar_w // 2
        bar_y = base_pos_px[1] - 30
        health_ratio = self.base_health / 100
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (bar_x, bar_y, int(bar_w * health_ratio), bar_h), border_radius=2)
        health_text = self.font_small.render("BASE", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (bar_x + bar_w//2 - health_text.get_width()//2, bar_y - 15))


        # Towers remaining
        towers_text = self.font_large.render(f"TOWERS: {self.towers_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(towers_text, (self.SCREEN_WIDTH // 2 - towers_text.get_width() // 2, self.SCREEN_HEIGHT - 40))
        
        # Inter-wave timer
        if self.wave_state == "INTER_WAVE" and not self.game_over:
            timer_sec = math.ceil(self.inter_wave_timer / 30)
            wave_msg = f"WAVE {self.current_wave + 1} STARTING IN {timer_sec}"
            timer_text = self.font_large.render(wave_msg, True, self.COLOR_TEXT)
            self.screen.blit(timer_text, (self.SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 40))

        # Game Over / Win
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font_huge.render(end_text_str, True, self.COLOR_TEXT)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - 50))
            
            final_score_text = self.font_large.render(f"FINAL SCORE: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(final_score_text, (self.SCREEN_WIDTH // 2 - final_score_text.get_width() // 2, self.SCREEN_HEIGHT // 2 + 20))

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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action
        mov_action = 0 # none
        if keys[pygame.K_UP]: mov_action = 1
        elif keys[pygame.K_DOWN]: mov_action = 2
        elif keys[pygame.K_LEFT]: mov_action = 3
        elif keys[pygame.K_RIGHT]: mov_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([mov_action, space_action, shift_action])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset() # Auto-restart
            pygame.time.wait(2000)

        clock.tick(30) # Run at 30 FPS

    env.close()