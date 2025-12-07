# Generated: 2025-08-28T02:37:22.262345
# Source Brief: brief_01757.md
# Brief Index: 1757

        
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



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor, Space to place selected tower, Shift to cycle tower type."
    )

    game_description = (
        "A top-down tower defense game. Place towers to defend your base from waves of enemies."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2500
        self.MAX_WAVES = 10
        self.WAVE_PREP_TIME = 90  # 3 seconds at 30fps

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PATH = (45, 45, 55)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_BASE_STROKE = (80, 220, 95)
        self.COLOR_SLOT = (60, 60, 70)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_ENEMY = (215, 60, 60)
        self.COLOR_ENEMY_STROKE = (255, 100, 100)
        self.COLOR_EXPLOSION = (255, 200, 0)
        self.COLOR_TEXT = (230, 230, 230)
        self.TOWER_COLORS = [
            (50, 150, 255), # Basic
            (255, 100, 200), # Fast
            (150, 100, 255)  # Long-range
        ]
        
        # EXACT spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_wave = pygame.font.SysFont("monospace", 24, bold=True)

        # Game assets (defined in code)
        self._define_game_assets()

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.gold = 0
        self.current_wave = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.explosions = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.wave_timer = 0
        self.wave_complete = True
        self.last_space_held = False
        self.last_shift_held = False
        
        # self.reset() is called by the validation function
        
        # self.validate_implementation()

    def _define_game_assets(self):
        # Path for enemies
        self.path = [
            (100, -20), (100, 120), (540, 120), (540, 280), (100, 280), (100, 420)
        ]
        self.path_segments = []
        for i in range(len(self.path) - 1):
            p1 = pygame.Vector2(self.path[i])
            p2 = pygame.Vector2(self.path[i+1])
            dist = p1.distance_to(p2)
            self.path_segments.append({'start': p1, 'end': p2, 'length': dist})
        
        # Tower placement slots
        self.tower_slots = [
            (200, 180), (440, 180), (200, 220), (440, 220)
        ]
        
        # Tower definitions
        self.TOWER_SPECS = {
            0: {"name": "Basic", "cost": 25, "range": 80, "damage": 25, "cooldown": 45},
            1: {"name": "Fast", "cost": 40, "range": 60, "damage": 15, "cooldown": 20},
            2: {"name": "Sniper", "cost": 60, "range": 150, "damage": 50, "cooldown": 90},
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.gold = 50
        self.current_wave = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.explosions = []
        
        self.cursor_pos = [0, 0] # x,y index in a 2x2 grid
        self.selected_tower_type = 0
        self.wave_timer = self.WAVE_PREP_TIME - 1
        self.wave_complete = True
        
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        
        # Handle actions
        reward += self._handle_actions(action)
        
        # Update game logic
        reward += self._update_game_state()
        
        # Update score
        self.score += reward
        
        # Check termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.base_health <= 0:
                reward -= 100
            elif self.current_wave > self.MAX_WAVES:
                reward += 100
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement for cursor
        if movement == 1: self.cursor_pos[1] = 0 # Up
        elif movement == 2: self.cursor_pos[1] = 1 # Down
        elif movement == 3: self.cursor_pos[0] = 0 # Left
        elif movement == 4: self.cursor_pos[0] = 1 # Right

        # Shift to cycle tower type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: UI_cycle.wav
        
        # Space to place tower (on press)
        if space_held and not self.last_space_held:
            slot_idx = self.cursor_pos[1] * 2 + self.cursor_pos[0]
            slot_pos = self.tower_slots[slot_idx]
            spec = self.TOWER_SPECS[self.selected_tower_type]
            
            is_occupied = any(t['pos'] == slot_pos for t in self.towers)
            
            if not is_occupied and self.gold >= spec['cost']:
                self.gold -= spec['cost']
                self.towers.append({
                    "pos": slot_pos,
                    "type": self.selected_tower_type,
                    "cooldown": 0,
                    "spec": spec
                })
                # sfx: tower_place.wav
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return 0.0 # Actions themselves don't give rewards

    def _update_game_state(self):
        reward = 0.0
        
        # Wave management
        if not self.enemies and self.wave_complete is False:
            self.wave_complete = True
            self.wave_timer = 0
            if self.current_wave > 0:
                reward += 1.0 # Wave clear bonus
        
        if self.wave_complete and self.current_wave < self.MAX_WAVES:
            self.wave_timer += 1
            if self.wave_timer >= self.WAVE_PREP_TIME:
                self._start_next_wave()

        # Update towers
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = self._find_target(tower)
                if target:
                    self.projectiles.append({
                        "pos": pygame.Vector2(tower['pos']),
                        "target": target,
                        "speed": 10,
                        "damage": tower['spec']['damage']
                    })
                    tower['cooldown'] = tower['spec']['cooldown']
                    # sfx: tower_shoot.wav

        # Update projectiles
        for p in self.projectiles[:]:
            target_pos = p['target']['pos']
            p_pos = p['pos']
            direction = (target_pos - p_pos)
            if direction.length() < p['speed']:
                p['pos'] = target_pos
            else:
                p['pos'] += direction.normalize() * p['speed']
            
            # Collision
            hit = False
            for e in self.enemies:
                if p['pos'].distance_to(e['pos']) < 10:
                    e['health'] -= p['damage']
                    self.explosions.append({"pos": p['pos'], "radius": 0, "max_radius": 20, "life": 10})
                    self.projectiles.remove(p)
                    # sfx: explosion.wav
                    hit = True
                    break
            if hit: continue

            if not (0 <= p['pos'].x <= self.WIDTH and 0 <= p['pos'].y <= self.HEIGHT):
                self.projectiles.remove(p)
        
        # Update enemies
        for e in self.enemies[:]:
            if e['health'] <= 0:
                self.gold += e['value']
                reward += 0.1
                self.enemies.remove(e)
                # sfx: enemy_die.wav
                continue

            # Movement
            e['dist_on_path'] += e['speed']
            current_dist = 0
            for seg in self.path_segments:
                if current_dist + seg['length'] >= e['dist_on_path']:
                    progress = (e['dist_on_path'] - current_dist) / seg['length']
                    # FIX: lerp's second argument must be in [0, 1]. Negative dist_on_path
                    # for staggered spawns can cause a negative progress. Clamp it to 0.
                    e['pos'] = seg['start'].lerp(seg['end'], max(0.0, progress))
                    break
                current_dist += seg['length']
            else: # Reached end of path
                self.base_health = max(0, self.base_health - e['damage'])
                self.enemies.remove(e)
                # sfx: base_damage.wav

        # Update explosions
        for exp in self.explosions[:]:
            exp['life'] -= 1
            exp['radius'] += exp['max_radius'] / 10
            if exp['life'] <= 0:
                self.explosions.remove(exp)

        return reward

    def _find_target(self, tower):
        in_range_enemies = []
        tower_pos = pygame.Vector2(tower['pos'])
        for e in self.enemies:
            if tower_pos.distance_to(e['pos']) <= tower['spec']['range']:
                in_range_enemies.append(e)
        
        if not in_range_enemies:
            return None
        
        # Target enemy furthest along the path
        return max(in_range_enemies, key=lambda e: e['dist_on_path'])

    def _start_next_wave(self):
        self.current_wave += 1
        self.wave_complete = False
        if self.current_wave > self.MAX_WAVES:
            return

        # sfx: wave_start.wav
        num_enemies = 2 + self.current_wave * 2
        enemy_speed = 0.8 + (self.current_wave - 1) * 0.05
        enemy_health = 50 + (self.current_wave - 1) * 15

        for i in range(num_enemies):
            self.enemies.append({
                "pos": pygame.Vector2(self.path[0]),
                "dist_on_path": -i * 25, # Stagger spawn
                "speed": enemy_speed,
                "health": enemy_health,
                "max_health": enemy_health,
                "damage": 10,
                "value": 5
            })
    
    def _check_termination(self):
        if self.base_health <= 0:
            return True
        if self.current_wave > self.MAX_WAVES and not self.enemies:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 40)
        
        # Base
        pygame.gfxdraw.box(self.screen, pygame.Rect(80, 380, 40, 40), self.COLOR_BASE)
        pygame.draw.rect(self.screen, self.COLOR_BASE_STROKE, pygame.Rect(80, 380, 40, 40), 2)
        
        # Tower slots and cursor
        for i, pos in enumerate(self.tower_slots):
            is_occupied = any(t['pos'] == pos for t in self.towers)
            color = self.TOWER_COLORS[self.selected_tower_type] if is_occupied else self.COLOR_SLOT
            pygame.draw.rect(self.screen, color, (pos[0]-15, pos[1]-15, 30, 30), 1 if not is_occupied else 0, border_radius=3)
            
            cursor_idx = self.cursor_pos[1] * 2 + self.cursor_pos[0]
            if i == cursor_idx:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (pos[0]-18, pos[1]-18, 36, 36), 2, border_radius=5)

        # Towers
        for t in self.towers:
            pos_x, pos_y = int(t['pos'][0]), int(t['pos'][1])
            color = self.TOWER_COLORS[t['type']]
            pygame.gfxdraw.filled_polygon(self.screen, [(pos_x, pos_y-10), (pos_x-10, pos_y+8), (pos_x+10, pos_y+8)], color)
            pygame.gfxdraw.aapolygon(self.screen, [(pos_x, pos_y-10), (pos_x-10, pos_y+8), (pos_x+10, pos_y+8)], color)

        # Enemies
        for e in self.enemies:
            pos = (int(e['pos'].x), int(e['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_ENEMY_STROKE)
            # Health bar
            health_ratio = e['health'] / e['max_health']
            bar_width = int(16 * health_ratio)
            pygame.draw.rect(self.screen, (0, 255, 0), (pos[0] - 8, pos[1] - 15, bar_width, 3))

        # Projectiles
        for p in self.projectiles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, (255, 255, 255))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, (255, 255, 255))

        # Explosions
        for exp in self.explosions:
            alpha = int(255 * (exp['life'] / 10))
            color = (*self.COLOR_EXPLOSION, alpha)
            s = pygame.Surface((exp['max_radius']*2, exp['max_radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (exp['max_radius'], exp['max_radius']), int(exp['radius']))
            self.screen.blit(s, (exp['pos'][0] - exp['max_radius'], exp['pos'][1] - exp['max_radius']))

    def _render_ui(self):
        # Gold
        gold_text = self.font_ui.render(f"GOLD: {self.gold}", True, self.COLOR_TEXT)
        self.screen.blit(gold_text, (10, 10))
        
        # Base Health
        health_text = self.font_ui.render(f"BASE HP: {self.base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH - health_text.get_width() - 10, 10))
        
        # Wave info
        if self.wave_complete and self.current_wave < self.MAX_WAVES:
            time_left = (self.WAVE_PREP_TIME - self.wave_timer) // 30
            wave_info = f"WAVE {self.current_wave + 1} STARTING IN {time_left}"
        else:
            wave_info = f"WAVE {self.current_wave}/{self.MAX_WAVES}"
        wave_text = self.font_wave.render(wave_info, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH/2 - wave_text.get_width()/2, self.HEIGHT - 40))

        # Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_info = f"Tower: {spec['name']} | Cost: {spec['cost']}"
        tower_text = self.font_ui.render(tower_info, True, self.TOWER_COLORS[self.selected_tower_type])
        self.screen.blit(tower_text, (10, 35))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.current_wave,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test game logic assertions from brief
        self.reset()
        self.base_health = 110
        self.step(self.action_space.sample())
        assert self.base_health <= 100
        
        self.reset()
        self.gold = -10
        self.step(self.action_space.sample())
        assert self.gold >= 0

        self.reset()
        self._start_next_wave()
        assert self.enemies[0].speed == 0.8
        self._start_next_wave()
        assert self.enemies[0].speed == 0.8 + 0.05
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # The validate_implementation function is a useful check
    env = GameEnv()
    env.reset()
    env.validate_implementation()
    
    obs, info = env.reset()
    print("Initial state:", info)
    
    # Test a few random steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {info['steps']}: Action={action}, Reward={reward:.2f}, Info={info}, Terminated={terminated}")

    # To visualize the game, you would need a different setup
    # that blits the env.screen to a display window in a loop.
    # For example:
    
    # import sys
    # os.environ["SDL_VIDEODRIVER"] = "x11"
    # pygame.display.init()
    # env = GameEnv()
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # done = False
    # while not done:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
        
    #     # Simple human control mapping for testing
    #     keys = pygame.key.get_pressed()
    #     movement = 0
    #     if keys[pygame.K_UP]: movement = 1
    #     if keys[pygame.K_DOWN]: movement = 2
    #     if keys[pygame.K_LEFT]: movement = 3
    #     if keys[pygame.K_RIGHT]: movement = 4
    #     space = 1 if keys[pygame.K_SPACE] else 0
    #     shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
    #     action = [movement, space, shift]
        
    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     # Blit the environment's screen to the display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     if terminated:
    #         print("Game Over!", info)
    #         obs, info = env.reset()

    #     env.clock.tick(30) # Limit to 30 FPS
        
    # env.close()