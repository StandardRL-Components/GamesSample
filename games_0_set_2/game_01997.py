
# Generated: 2025-08-28T03:20:48.570226
# Source Brief: brief_01997.md
# Brief Index: 1997

        
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
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle tower types. Press Space to build a tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down tower defense game. Place towers to defend your base "
        "from waves of incoming enemies and survive for 20 waves."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.Font(None, 24)
        self.font_m = pygame.font.Font(None, 32)
        self.font_l = pygame.font.Font(None, 48)

        # Game Constants
        self.GRID_SIZE = 40
        self.GRID_W = self.screen_width // self.GRID_SIZE
        self.GRID_H = self.screen_height // self.GRID_SIZE
        self.MAX_STEPS = 5000  # Increased to allow for longer games
        self.MAX_WAVES = 20
        self.WAVE_PREP_TIME = 300 # 10 seconds at 30fps
        self.ENEMY_SPAWN_INTERVAL = 15 # 0.5 seconds

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PATH = (60, 65, 70)
        self.COLOR_BASE = (0, 150, 200)
        self.COLOR_BASE_DMG = (255, 100, 100)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_GOLD = (255, 200, 0)
        self.CURSOR_VALID = (50, 200, 50, 100)
        self.CURSOR_INVALID = (200, 50, 50, 100)
        
        # Enemy Path (Grid Coordinates)
        self.path = [
            (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
            (6, 4), (6, 3), (5, 3), (4, 3), (3, 3), (2, 3),
            (2, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1),
            (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
            (9, 7), (10, 7), (11, 7), (12, 7), (13, 7), (14, 7), (15, 7)
        ]
        self.path_pixels = [(x * self.GRID_SIZE + self.GRID_SIZE // 2, y * self.GRID_SIZE + self.GRID_SIZE // 2) for x, y in self.path]

        # Tower Definitions
        self.TOWER_STATS = {
            0: {"name": "Gatling", "cost": 50, "range": 80, "damage": 2, "fire_rate": 5, "color": (0, 200, 255)}, # High RoF, low dmg
            1: {"name": "Cannon", "cost": 125, "range": 150, "damage": 25, "fire_rate": 30, "color": (255, 150, 0)}, # Low RoF, high dmg
        }
        
        # Initialize state variables
        self.reset()
        
        # Implementation Validation
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 100
        self.gold = 150
        
        # Wave Management
        self.current_wave = 0
        self.wave_timer = self.WAVE_PREP_TIME
        self.enemies_to_spawn = []
        self.enemy_spawn_timer = 0

        # Entities
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        # Player Control State
        self.cursor_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.selected_tower_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Player Actions
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # 2. Update Game Logic
        self._update_wave_spawner()
        self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        self.steps += 1
        
        # 3. Check for Termination
        if self.base_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
        elif self.current_wave > self.MAX_WAVES and not self.enemies:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward += 100

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

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
        if movement != 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1 # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1 # Right
            self.cursor_pos = (
                np.clip(self.cursor_pos[0] + dx, 0, self.GRID_W - 1),
                np.clip(self.cursor_pos[1] + dy, 0, self.GRID_H - 1)
            )

        # Cycle tower type on key press (rising edge)
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_STATS)
            # SFX: UI_SWITCH

        # Place tower on key press (rising edge)
        if space_held and not self.prev_space_held:
            self._place_tower()
            # SFX: UI_CONFIRM / UI_ERROR

    def _place_tower(self):
        stats = self.TOWER_STATS[self.selected_tower_type]
        if self.gold < stats['cost']:
            return # Not enough gold

        if self.cursor_pos in self.path:
            return # Can't build on path

        if any(t['grid_pos'] == self.cursor_pos for t in self.towers):
            return # Tile occupied

        self.gold -= stats['cost']
        self.towers.append({
            "grid_pos": self.cursor_pos,
            "pixel_pos": (self.cursor_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, self.cursor_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2),
            "type": self.selected_tower_type,
            "cooldown": 0,
            "angle": 0
        })
        # SFX: BUILD_TOWER

    def _prepare_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            return

        num_enemies = 3 + self.current_wave * 2
        base_health = 10 + self.current_wave * 5
        base_speed = 1.0 + self.current_wave * 0.05
        
        for _ in range(num_enemies):
            health = int(base_health * self.np_random.uniform(0.8, 1.2))
            speed = base_speed * self.np_random.uniform(0.9, 1.1)
            self.enemies_to_spawn.append({'health': health, 'max_health': health, 'speed': speed})

    def _update_wave_spawner(self):
        if self.enemies or self.enemies_to_spawn:
            # Spawn next enemy in current wave
            if self.enemies_to_spawn:
                self.enemy_spawn_timer -= 1
                if self.enemy_spawn_timer <= 0:
                    self.enemy_spawn_timer = self.ENEMY_SPAWN_INTERVAL
                    enemy_stats = self.enemies_to_spawn.pop(0)
                    self.enemies.append({
                        "pos": list(self.path_pixels[0]),
                        "path_index": 0,
                        "health": enemy_stats['health'],
                        "max_health": enemy_stats['max_health'],
                        "speed": enemy_stats['speed'],
                        "id": self.np_random.integers(1, 1e9)
                    })
                    # SFX: ENEMY_SPAWN
            return 0 # No wave completion reward yet

        # If all enemies are gone, start timer for next wave
        self.wave_timer -= 1
        if self.wave_timer <= 0:
            self.wave_timer = self.WAVE_PREP_TIME
            self._prepare_next_wave()
            if self.current_wave > 1 and self.current_wave <= self.MAX_WAVES:
                 return 1.0 # Reward for completing a wave
        return 0

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            stats = self.TOWER_STATS[tower['type']]
            target = None
            min_dist = stats['range'] ** 2

            # Find closest enemy in range
            for enemy in self.enemies:
                dist_sq = (tower['pixel_pos'][0] - enemy['pos'][0])**2 + (tower['pixel_pos'][1] - enemy['pos'][1])**2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    target = enemy
            
            if target:
                # SFX: TOWER_FIRE_GATLING / TOWER_FIRE_CANNON
                tower['cooldown'] = stats['fire_rate']
                self.projectiles.append({
                    "start_pos": list(tower['pixel_pos']),
                    "pos": list(tower['pixel_pos']),
                    "target_id": target['id'],
                    "damage": stats['damage'],
                    "speed": 15,
                    "type": tower['type']
                })
                # Aim tower at target
                dx = target['pos'][0] - tower['pixel_pos'][0]
                dy = target['pos'][1] - tower['pixel_pos'][1]
                tower['angle'] = math.atan2(dy, dx)


    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            
            if not target:
                self.projectiles.remove(proj)
                continue

            # Move towards target
            target_pos = target['pos']
            dx = target_pos[0] - proj['pos'][0]
            dy = target_pos[1] - proj['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < proj['speed']:
                # Hit target
                target['health'] -= proj['damage']
                self._create_particles(target['pos'], 5, self.TOWER_STATS[proj['type']]['color'])
                # SFX: ENEMY_HIT
                if target['health'] <= 0:
                    self.gold += 5 + self.current_wave
                    reward += 0.1
                    self.enemies.remove(target)
                    self._create_particles(target['pos'], 20, self.COLOR_ENEMY)
                    # SFX: ENEMY_DESTROYED
                self.projectiles.remove(proj)
            else:
                # Move
                proj['pos'][0] += (dx / dist) * proj['speed']
                proj['pos'][1] += (dy / dist) * proj['speed']
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy['path_index'] >= len(self.path_pixels) - 1:
                # Reached base
                self.base_health -= 10
                self.base_health = max(0, self.base_health)
                reward -= 1.0 # -0.1 per health point lost -> -1 for 10 health
                self.enemies.remove(enemy)
                self._create_particles(enemy['pos'], 30, self.COLOR_BASE_DMG)
                # SFX: BASE_DAMAGE
                continue

            target_pos = self.path_pixels[enemy['path_index'] + 1]
            dx = target_pos[0] - enemy['pos'][0]
            dy = target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)

            if dist < enemy['speed']:
                enemy['pos'] = list(target_pos)
                enemy['path_index'] += 1
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']
        return reward

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": self.np_random.integers(10, 20),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
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
        for x in range(0, self.screen_width, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y))
        
        # Draw path and base
        for x, y in self.path:
            pygame.draw.rect(self.screen, self.COLOR_PATH, (x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
        base_pos = self.path[-1]
        pygame.draw.rect(self.screen, self.COLOR_BASE, (base_pos[0] * self.GRID_SIZE, base_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
        
        # Draw towers
        for t in self.towers:
            stats = self.TOWER_STATS[t['type']]
            pos_int = (int(t['pixel_pos'][0]), int(t['pixel_pos'][1]))
            pygame.draw.rect(self.screen, stats['color'], (t['grid_pos'][0]*self.GRID_SIZE+4, t['grid_pos'][1]*self.GRID_SIZE+4, self.GRID_SIZE-8, self.GRID_SIZE-8))
            # Draw barrel
            barrel_len = self.GRID_SIZE // 2
            end_x = pos_int[0] + barrel_len * math.cos(t['angle'])
            end_y = pos_int[1] + barrel_len * math.sin(t['angle'])
            pygame.draw.line(self.screen, (200,200,200), pos_int, (int(end_x), int(end_y)), 4)

        # Draw enemies
        for e in self.enemies:
            pos_int = (int(e['pos'][0]), int(e['pos'][1]))
            radius = self.GRID_SIZE // 3
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_ENEMY)
            # Health bar
            health_pct = e['health'] / e['max_health']
            bar_w = int(radius * 2 * health_pct)
            pygame.draw.rect(self.screen, (0, 255, 0), (pos_int[0] - radius, pos_int[1] - radius - 8, bar_w, 4))
            pygame.draw.rect(self.screen, (255, 0, 0), (pos_int[0] - radius + bar_w, pos_int[1] - radius - 8, radius*2 - bar_w, 4))

        # Draw projectiles
        for p in self.projectiles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            color = self.TOWER_STATS[p['type']]['color']
            pygame.draw.circle(self.screen, color, pos_int, 4)

        # Draw particles
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20))))
            p_color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, p_color, (2, 2), 2)
            self.screen.blit(temp_surf, (pos_int[0] - 2, pos_int[1] - 2))

        # Draw cursor and tower range
        self._render_cursor()

    def _render_cursor(self):
        c_pos = self.cursor_pos
        stats = self.TOWER_STATS[self.selected_tower_type]
        
        # Check validity
        is_valid = True
        if self.gold < stats['cost'] or c_pos in self.path or any(t['grid_pos'] == c_pos for t in self.towers):
            is_valid = False

        # Draw range indicator
        cursor_pixel_pos = (c_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, c_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2)
        range_color = (255, 255, 255, 30)
        range_surf = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(range_surf, cursor_pixel_pos[0], cursor_pixel_pos[1], stats['range'], range_color)
        self.screen.blit(range_surf, (0, 0))

        # Draw cursor box
        color = self.CURSOR_VALID if is_valid else self.CURSOR_INVALID
        s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        s.fill(color)
        self.screen.blit(s, (c_pos[0] * self.GRID_SIZE, c_pos[1] * self.GRID_SIZE), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.rect(self.screen, (255, 255, 255), (c_pos[0] * self.GRID_SIZE, c_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE), 1)

    def _render_ui(self):
        # Top Bar
        bar_surf = pygame.Surface((self.screen_width, 30), pygame.SRCALPHA)
        bar_surf.fill((10, 10, 10, 200))
        self.screen.blit(bar_surf, (0,0))
        
        # Health
        health_text = self.font_m.render(f"❤️ {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 5))
        
        # Gold
        gold_text = self.font_m.render(f"💰 {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (130, 5))
        
        # Wave
        wave_str = f"Wave: {self.current_wave}/{self.MAX_WAVES}"
        if not self.enemies and not self.enemies_to_spawn and self.current_wave <= self.MAX_WAVES:
            wave_str += f" (Next in {self.wave_timer//30 + 1}s)"
        wave_text = self.font_m.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (250, 5))

        # Selected Tower
        stats = self.TOWER_STATS[self.selected_tower_type]
        tower_text = self.font_m.render(f"Build: {stats['name']} (${stats['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(tower_text, (460, 5))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WON!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            
            end_text = self.font_l.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "gold": self.gold,
            "wave": self.current_wave,
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Wave: {info['wave']}")
            obs, info = env.reset()
            total_reward = 0
            # Add a small delay before restarting
            pygame.time.wait(2000)

        clock.tick(30) # Run at 30 FPS

    env.close()