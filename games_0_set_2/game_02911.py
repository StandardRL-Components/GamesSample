
# Generated: 2025-08-28T06:23:45.563190
# Source Brief: brief_02911.md
# Brief Index: 2911

        
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
        "Controls: Arrows to move selector. Space to place a tower. Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down tower defense game. Strategically place towers to defend your base from waves of enemies."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and rendering setup
        self.width, self.height = 640, 400
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.font_small = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)

        # Colors
        self.COLOR_BG = (25, 35, 55)
        self.COLOR_PATH = (45, 55, 75)
        self.COLOR_GRID = (60, 70, 90)
        self.COLOR_BASE = (60, 200, 120)
        self.COLOR_BASE_GLOW = (60, 200, 120, 50)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_ENEMY_GLOW = (220, 50, 50, 50)
        self.COLOR_SELECTOR = (255, 220, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR_BG = (100, 40, 40)
        self.COLOR_HEALTH_BAR = (40, 220, 40)

        # Game constants
        self.MAX_STEPS = 30 * 180  # 3 minutes at 30fps
        self.TOTAL_WAVES = 10
        self.WAVE_DELAY = 5 * 30 # 5 seconds

        # Initialize state variables
        self.path = []
        self.tower_spots = []
        self.tower_spot_grid_dims = (0, 0)
        self.tower_definitions = []

        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def _define_level(self):
        # Define the enemy path as a series of waypoints
        self.path = [
            (-50, 200), (100, 200), (100, 100), (400, 100),
            (400, 300), (200, 300), (200, 200), (self.width + 50, 200)
        ]
        
        # Define tower placement spots on a grid
        self.tower_spots = []
        self.tower_spot_grid_dims = (8, 4)
        grid_w, grid_h = self.tower_spot_grid_dims
        for y in range(grid_h):
            for x in range(grid_w):
                px = 100 + x * 60
                py = 40 + y * 60
                if y >= 2:
                    py += 100
                self.tower_spots.append((px, py))

        # Define tower types
        self.tower_definitions = [
            {
                "name": "Gatling", "cost": 100, "range": 80, "damage": 4, 
                "fire_rate": 5, "color": (0, 150, 255), "projectile_speed": 10
            },
            {
                "name": "Cannon", "cost": 250, "range": 120, "damage": 25, 
                "fire_rate": 30, "color": (255, 150, 0), "projectile_speed": 7
            },
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize level layout
        self._define_level()
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.money = 250
        self.base_health = 100
        self.max_base_health = 100
        self.game_over = False
        self.victory = False

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 0
        self.wave_timer = self.WAVE_DELAY // 2
        self.enemies_in_wave = 0
        self.enemies_spawned_in_wave = 0

        self.selector_index = 0
        self.selected_tower_type = 0
        self.last_action = [0, 0, 0]

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # Process player input (rising edge detection)
        reward += self._handle_input(action)
        
        if not self.game_over:
            # Update game logic
            self._update_waves()
            reward += self._update_towers()
            reward += self._update_projectiles()
            reward += self._update_enemies()
            self._update_particles()
        
        self.steps += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        if terminated:
            if self.victory:
                reward += 100 # Victory bonus
            else:
                reward -= 100 # Defeat penalty

        self.last_action = action
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_press, shift_press = action[0], action[1], action[2]
        last_movement, last_space, last_shift = self.last_action

        # Movement (on new direction press)
        if movement != 0 and movement != last_movement:
            grid_w, grid_h = self.tower_spot_grid_dims
            row, col = self.selector_index // grid_w, self.selector_index % grid_w
            if movement == 1: # Up
                row = max(0, row - 1)
            elif movement == 2: # Down
                row = min(grid_h - 1, row + 1)
            elif movement == 3: # Left
                col = max(0, col - 1)
            elif movement == 4: # Right
                col = min(grid_w - 1, col + 1)
            self.selector_index = row * grid_w + col
            # Sound: UI_move.wav

        # Cycle tower type (on key down)
        if shift_press and not last_shift:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_definitions)
            # Sound: UI_cycle.wav

        # Place tower (on key down)
        if space_press and not last_space:
            spot_pos = self.tower_spots[self.selector_index]
            tower_def = self.tower_definitions[self.selected_tower_type]
            
            is_occupied = any(t['pos'] == spot_pos for t in self.towers)
            
            if not is_occupied and self.money >= tower_def['cost']:
                self.money -= tower_def['cost']
                self.towers.append({
                    "pos": spot_pos,
                    "type_index": self.selected_tower_type,
                    "cooldown": 0,
                    "target": None
                })
                # Sound: place_tower.wav
        
        return 0

    def _update_waves(self):
        if self.current_wave > self.TOTAL_WAVES:
            return

        if len(self.enemies) == 0 and self.enemies_spawned_in_wave == self.enemies_in_wave:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                if self.current_wave > self.TOTAL_WAVES:
                    self.victory = True
                    return
                
                self.wave_timer = self.WAVE_DELAY
                self.enemies_in_wave = 5 + self.current_wave * 2
                self.enemies_spawned_in_wave = 0
                # Sound: wave_start.wav

        if self.enemies_spawned_in_wave < self.enemies_in_wave and self.wave_timer % (30 // self.current_wave + 1) == 0:
            self.enemies_spawned_in_wave += 1
            
            health_multiplier = 1 + (self.current_wave - 1) * 0.2
            speed_multiplier = 1 + (self.current_wave - 1) * 0.05
            
            self.enemies.append({
                "pos": list(self.path[0]),
                "path_index": 0,
                "health": 50 * health_multiplier,
                "max_health": 50 * health_multiplier,
                "speed": 1.0 * speed_multiplier,
            })

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            tower_def = self.tower_definitions[tower['type_index']]
            
            # Find a target
            target = None
            min_dist = float('inf')
            for enemy in self.enemies:
                dist = math.hypot(tower['pos'][0] - enemy['pos'][0], tower['pos'][1] - enemy['pos'][1])
                if dist <= tower_def['range'] and dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                self.projectiles.append({
                    "pos": list(tower['pos']),
                    "target": target,
                    "speed": tower_def['projectile_speed'],
                    "damage": tower_def['damage'],
                    "color": tower_def['color']
                })
                tower['cooldown'] = tower_def['fire_rate']
                # Sound: tower_fire.wav
        return reward

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target = proj['target']
            
            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            # Move towards target
            dx = target['pos'][0] - proj['pos'][0]
            dy = target['pos'][1] - proj['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < proj['speed']:
                # Hit
                target['health'] -= proj['damage']
                reward += 0.1 # Reward for hitting
                self._create_particles(proj['pos'], 5, self.COLOR_ENEMY, 1, 3)
                self.projectiles.remove(proj)
                # Sound: enemy_hit.wav
            else:
                proj['pos'][0] += (dx / dist) * proj['speed']
                proj['pos'][1] += (dy / dist) * proj['speed']
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            # Check for death
            if enemy['health'] <= 0:
                reward += 1 # Reward for kill
                self.money += 25 + self.current_wave * 2
                self.score += 1
                self._create_particles(enemy['pos'], 15, self.COLOR_ENEMY_GLOW, 2, 5)
                self.enemies.remove(enemy)
                # Sound: enemy_die.wav
                continue

            # Move along path
            path_index = enemy['path_index']
            if path_index >= len(self.path) - 1:
                # Reached base
                self.base_health -= 10
                reward -= 5 # Penalty for leak
                self._create_particles((self.width-25, 200), 20, self.COLOR_BASE, 3, 6)
                self.enemies.remove(enemy)
                # Sound: base_damage.wav
                continue

            target_pos = self.path[path_index + 1]
            dx = target_pos[0] - enemy['pos'][0]
            dy = target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < enemy['speed']:
                enemy['path_index'] += 1
                enemy['pos'] = list(target_pos)
            else:
                enemy['pos'][0] += (dx / dist) * enemy['speed']
                enemy['pos'][1] += (dy / dist) * enemy['speed']
        return reward

    def _create_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(min_speed, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": random.randint(10, 20),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.victory and len(self.enemies) == 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "money": self.money,
            "wave": self.current_wave,
            "base_health": self.base_health,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 30)

        # Draw tower spots
        for pos in self.tower_spots:
            pygame.draw.rect(self.screen, self.COLOR_GRID, (pos[0]-15, pos[1]-15, 30, 30), 1)

        # Draw base
        base_rect = pygame.Rect(self.width - 40, self.height/2 - 20, 40, 40)
        pygame.gfxdraw.box(self.screen, base_rect, self.COLOR_BASE_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)

        # Draw towers
        for tower in self.towers:
            tower_def = self.tower_definitions[tower['type_index']]
            color = tower_def['color']
            pos = tower['pos']
            points = [
                (pos[0], pos[1] - 12),
                (pos[0] - 10, pos[1] + 8),
                (pos[0] + 10, pos[1] + 8)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 9, self.COLOR_ENEMY_GLOW)
            # Body
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, self.COLOR_ENEMY)
            # Health bar
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            bar_w = 14
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (pos[0] - bar_w/2, pos[1] - 15, bar_w, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (pos[0] - bar_w/2, pos[1] - 15, bar_w * health_pct, 3))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, proj['color'])

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            life_pct = p['life'] / 20.0
            size = int(3 * life_pct)
            if size > 0:
                color = (*p['color'][:3], int(p['color'][3] * life_pct)) if len(p['color']) == 4 else p['color']
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)

        # Draw selector and tower range
        selector_pos = self.tower_spots[self.selector_index]
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        color = (
            self.COLOR_SELECTOR[0],
            self.COLOR_SELECTOR[1],
            int(100 + 155 * pulse)
        )
        pygame.draw.rect(self.screen, color, (selector_pos[0]-18, selector_pos[1]-18, 36, 36), 3, border_radius=4)
        
        tower_def = self.tower_definitions[self.selected_tower_type]
        pygame.gfxdraw.aacircle(self.screen, selector_pos[0], selector_pos[1], tower_def['range'], (255, 255, 255, 50))


    def _render_ui(self):
        # Top-left info
        wave_text = f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}"
        if self.current_wave <= self.TOTAL_WAVES and len(self.enemies) == 0 and self.enemies_spawned_in_wave == self.enemies_in_wave:
             wave_text = f"NEXT WAVE IN {self.wave_timer // 30 + 1}"
        self._draw_text(wave_text, (10, 10), self.COLOR_TEXT)
        self._draw_text(f"SCORE: {self.score}", (10, 30), self.COLOR_TEXT)

        # Bottom-left info (selected tower)
        tower_def = self.tower_definitions[self.selected_tower_type]
        tower_color = tower_def['color'] if self.money >= tower_def['cost'] else self.COLOR_ENEMY
        self._draw_text(f"TOWER: {tower_def['name']}", (10, self.height - 50), tower_color)
        self._draw_text(f"COST: {tower_def['cost']}", (10, self.height - 30), tower_color)

        # Top-right info
        self._draw_text(f"$: {self.money}", (self.width - 10, 10), self.COLOR_SELECTOR, align="right")
        self._draw_text(f"BASE HP: {self.base_health}", (self.width - 10, 30), self.COLOR_BASE, align="right")
        
        # Game Over / Victory Text
        if self.game_over:
            if self.victory:
                self._draw_text("VICTORY", (self.width/2, self.height/2 - 30), self.COLOR_BASE, self.font_large, align="center")
            else:
                self._draw_text("GAME OVER", (self.width/2, self.height/2 - 30), self.COLOR_ENEMY, self.font_large, align="center")

    def _draw_text(self, text, pos, color, font=None, align="left"):
        if font is None:
            font = self.font_small
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "right":
            text_rect.topright = pos
        elif align == "center":
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")