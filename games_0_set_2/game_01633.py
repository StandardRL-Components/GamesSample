
# Generated: 2025-08-28T02:12:10.053383
# Source Brief: brief_01633.md
# Brief Index: 1633

        
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
        "A top-down tower defense game. Place towers to defend your base from waves of geometric enemies."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        
        self._define_game_parameters()

        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_wave = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.reset()

        self.validate_implementation()

    def _define_game_parameters(self):
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2500
        self.MAX_WAVES = 10
        self.WAVE_PREP_TIME = 120 # steps between waves
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_BASE = (0, 150, 200)
        self.COLOR_BASE_GLOW = (0, 150, 200, 50)
        self.COLOR_PLACE_ZONE = (0, 255, 0, 30)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_INVALID = (255, 0, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BG = (100, 0, 0)
        self.COLOR_HEALTH_FG = (0, 200, 0)

        # Path
        self.PATH_WAYPOINTS = [
            (-50, 100), (150, 100), (150, 300), (450, 300), (450, 150), (self.WIDTH + 50, 150)
        ]
        
        # Placement
        self.PLACEMENT_GRID_DIMS = (6, 4)
        self.PLACEMENT_SPOTS = []
        for row in range(self.PLACEMENT_GRID_DIMS[1]):
            for col in range(self.PLACEMENT_GRID_DIMS[0]):
                x = 100 + col * 80
                y = 50 + row * 80
                self.PLACEMENT_SPOTS.append((x,y))

        # Towers
        self.TOWER_SPECS = {
            0: {"name": "Gatling", "cost": 75, "range": 80, "damage": 2, "fire_rate": 8, "color": (0, 200, 255)},
            1: {"name": "Cannon", "cost": 150, "range": 120, "damage": 15, "fire_rate": 60, "color": (255, 180, 0)},
        }

        # Enemies
        self.ENEMY_SPECS = {
            "grunt": {"base_health": 10, "base_speed": 1.0, "radius": 7, "value": 5, "color": (255, 50, 50)},
            "tank": {"base_health": 50, "base_speed": 0.6, "radius": 12, "value": 25, "color": (200, 0, 100)},
        }

        # Waves
        self.WAVE_DEFINITIONS = [
            [("grunt", 10, 30)], # Wave 1: 10 grunts, 30 frame interval
            [("grunt", 15, 25)],
            [("grunt", 20, 20), ("tank", 1, 0)],
            [("grunt", 15, 20), ("tank", 3, 60)],
            [("tank", 8, 45)],
            [("grunt", 30, 10), ("tank", 5, 60)],
            [("grunt", 20, 10), ("tank", 10, 30)],
            [("tank", 15, 25)],
            [("grunt", 50, 5)],
            [("grunt", 20, 10), ("tank", 20, 20)],
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = 100
        self.max_base_health = 100
        self.money = 100

        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_timer = self.WAVE_PREP_TIME

        self.enemies_to_spawn = []
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_index = 0
        self.selected_tower_type_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward_info = {"hits": 0, "kills": 0, "wave_cleared": False, "base_damage": 0}
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)
        
        self._update_waves()
        self._update_towers()
        self._update_projectiles(reward_info)
        self._update_enemies(reward_info)
        self._update_particles()
        
        self.steps += 1
        
        if not self.wave_in_progress and not self.enemies and self.wave_number <= self.MAX_WAVES:
            if self.wave_timer <= 0:
                self._start_next_wave()
                if self.wave_number > 1: # Don't reward for clearing wave 0
                    reward_info["wave_cleared"] = True
            else:
                self.wave_timer -= 1

        reward = self._calculate_reward(reward_info)
        self.score += reward
        terminated = self._check_termination()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held, shift_held):
        # Move cursor
        rows, cols = self.PLACEMENT_GRID_DIMS[1], self.PLACEMENT_GRID_DIMS[0]
        if movement == 1: # Up
            self.cursor_index = (self.cursor_index - cols) % (rows * cols)
        elif movement == 2: # Down
            self.cursor_index = (self.cursor_index + cols) % (rows * cols)
        elif movement == 3: # Left
            self.cursor_index = (self.cursor_index - 1) % (rows * cols)
        elif movement == 4: # Right
            self.cursor_index = (self.cursor_index + 1) % (rows * cols)
        
        # Cycle tower type on shift PRESS
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.TOWER_SPECS)
            # sfx: menu_click

        # Place tower on space PRESS
        if space_held and not self.prev_space_held:
            spec = self.TOWER_SPECS[self.selected_tower_type_idx]
            pos = self.PLACEMENT_SPOTS[self.cursor_index]
            is_occupied = any(t['pos'] == pos for t in self.towers)
            
            if self.money >= spec['cost'] and not is_occupied:
                self.money -= spec['cost']
                self.towers.append({
                    "pos": pos, "type": self.selected_tower_type_idx, 
                    "cooldown": 0, "spec": spec
                })
                self._create_particles(pos, spec['color'], 20, 2, 15)
                # sfx: place_tower

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            self.game_won = True
            return

        self.wave_in_progress = True
        self.enemies_to_spawn = []
        wave_def = self.WAVE_DEFINITIONS[self.wave_number - 1]
        
        spawn_delay = 0
        for enemy_type, count, interval in wave_def:
            for i in range(count):
                self.enemies_to_spawn.append({
                    "type": enemy_type,
                    "spawn_time": self.steps + spawn_delay,
                })
                spawn_delay += interval
        self.enemies_to_spawn.sort(key=lambda x: x['spawn_time'])
    
    def _update_waves(self):
        if self.enemies_to_spawn and self.enemies_to_spawn[0]['spawn_time'] <= self.steps:
            enemy_def = self.enemies_to_spawn.pop(0)
            spec = self.ENEMY_SPECS[enemy_def['type']]
            
            # Difficulty scaling
            speed_mod = 1 + (self.wave_number - 1) * 0.05
            health_mod = 1 + math.floor((self.wave_number - 1) / 2) * 1

            self.enemies.append({
                "pos": list(self.PATH_WAYPOINTS[0]),
                "health": spec['base_health'] * health_mod,
                "max_health": spec['base_health'] * health_mod,
                "speed": spec['base_speed'] * speed_mod,
                "path_index": 1,
                "spec": spec
            })
            # sfx: enemy_spawn

        if not self.enemies_to_spawn and self.wave_in_progress:
            self.wave_in_progress = False
            self.wave_timer = self.WAVE_PREP_TIME

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            target = None
            min_dist = tower['spec']['range']
            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1])
                if dist < min_dist:
                    min_dist = dist
                    target = enemy
            
            if target:
                tower['cooldown'] = tower['spec']['fire_rate']
                self.projectiles.append({
                    "pos": list(tower['pos']),
                    "target": target,
                    "damage": tower['spec']['damage'],
                    "speed": 8,
                    "color": tower['spec']['color']
                })
                # sfx: tower_shoot

    def _update_projectiles(self, reward_info):
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj['target']['pos']
            proj_pos = proj['pos']
            
            angle = math.atan2(target_pos[1] - proj_pos[1], target_pos[0] - proj_pos[0])
            proj_pos[0] += math.cos(angle) * proj['speed']
            proj_pos[1] += math.sin(angle) * proj['speed']

            if math.hypot(target_pos[0] - proj_pos[0], target_pos[1] - proj_pos[1]) < proj['target']['spec']['radius']:
                proj['target']['health'] -= proj['damage']
                reward_info["hits"] += 1
                self._create_particles(target_pos, proj['color'], 5, 1, 8)
                # sfx: enemy_hit
                
                if proj['target']['health'] <= 0:
                    reward_info["kills"] += 1
                    self.money += proj['target']['spec']['value']
                    self._create_particles(target_pos, proj['target']['spec']['color'], 30, 3, 20)
                    self.enemies.remove(proj['target'])
                    # sfx: enemy_die
                
                self.projectiles.remove(proj)

    def _update_enemies(self, reward_info):
        for enemy in self.enemies[:]:
            if enemy['path_index'] >= len(self.PATH_WAYPOINTS):
                self.base_health -= enemy['spec']['base_health'] // 2
                reward_info["base_damage"] += enemy['spec']['base_health'] // 2
                self.enemies.remove(enemy)
                self._create_particles((self.WIDTH-40, 150), self.COLOR_BASE, 50, 4, 25)
                # sfx: base_damage
                continue

            target_waypoint = self.PATH_WAYPOINTS[enemy['path_index']]
            dist_to_waypoint = math.hypot(target_waypoint[0] - enemy['pos'][0], target_waypoint[1] - enemy['pos'][1])

            if dist_to_waypoint < enemy['speed']:
                enemy['path_index'] += 1
            else:
                angle = math.atan2(target_waypoint[1] - enemy['pos'][1], target_waypoint[0] - enemy['pos'][0])
                enemy['pos'][0] += math.cos(angle) * enemy['speed']
                enemy['pos'][1] += math.sin(angle) * enemy['speed']

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _calculate_reward(self, reward_info):
        reward = 0.0
        reward += reward_info["hits"] * 0.1
        reward += reward_info["kills"] * 1.0
        if reward_info["wave_cleared"]:
            reward += 10.0
        
        # Living penalty
        reward -= 0.01

        if self._check_termination():
            if self.game_won:
                reward += 100.0
            elif self.base_health <= 0:
                reward -= 100.0
        
        return reward

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.game_won and not self.enemies:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
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
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.PATH_WAYPOINTS, 30)
        
        # Base
        base_rect = pygame.Rect(self.WIDTH - 50, 125, 50, 50)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        pygame.gfxdraw.aacircle(self.screen, int(base_rect.centerx), int(base_rect.centery), 40, self.COLOR_BASE_GLOW)
        pygame.gfxdraw.aacircle(self.screen, int(base_rect.centerx), int(base_rect.centery), 50, self.COLOR_BASE_GLOW)

        # Placement zones and cursor
        for i, pos in enumerate(self.PLACEMENT_SPOTS):
            is_occupied = any(t['pos'] == pos for t in self.towers)
            color = self.COLOR_PLACE_ZONE if not is_occupied else (255,0,0,30)
            pygame.gfxdraw.box(self.screen, (pos[0]-20, pos[1]-20, 40, 40), color)
            if i == self.cursor_index:
                spec = self.TOWER_SPECS[self.selected_tower_type_idx]
                can_afford = self.money >= spec['cost']
                cursor_color = self.COLOR_CURSOR if can_afford and not is_occupied else self.COLOR_CURSOR_INVALID
                pygame.draw.rect(self.screen, cursor_color, (pos[0]-22, pos[1]-22, 44, 44), 2, border_radius=3)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], spec['range'], (*cursor_color, 80))

        # Towers
        for tower in self.towers:
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            color = tower['spec']['color']
            size = 18
            poly = [(pos[0], pos[1]-size), (pos[0]+size, pos[1]), (pos[0], pos[1]+size), (pos[0]-size, pos[1])]
            pygame.gfxdraw.aapolygon(self.screen, poly, color)
            pygame.gfxdraw.filled_polygon(self.screen, poly, color)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            radius = enemy['spec']['radius']
            color = enemy['spec']['color']
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            # Health bar
            if enemy['health'] < enemy['max_health']:
                health_pct = enemy['health'] / enemy['max_health']
                bar_width = radius * 2
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (pos[0]-radius, pos[1]-radius-8, bar_width, 4))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (pos[0]-radius, pos[1]-radius-8, bar_width*health_pct, 4))

        # Projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.rect(self.screen, proj['color'], (pos[0]-2, pos[1]-2, 4, 4))

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = max(0, p['size'] * (p['life'] / p['max_life']))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0,0,size*2, size*2))
            self.screen.blit(temp_surf, (p['pos'][0] - size, p['pos'][1] - size))
            
    def _render_ui(self):
        # Top Bar
        bar_surf = pygame.Surface((self.WIDTH, 30), pygame.SRCALPHA)
        bar_surf.fill((0,0,0,100))
        self.screen.blit(bar_surf, (0,0))

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 7))

        # Money
        money_text = self.font_ui.render(f"$ {self.money}", True, (255, 223, 0))
        self.screen.blit(money_text, (160, 7))

        # Wave
        wave_str = f"WAVE {self.wave_number}/{self.MAX_WAVES}"
        if not self.wave_in_progress and self.wave_number < self.MAX_WAVES:
            wave_str += f" (STARTS IN {self.wave_timer//30 + 1}s)"
        wave_text = self.font_ui.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 7))

        # Base Health
        health_pct = max(0, self.base_health / self.max_base_health)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (self.WIDTH//2 - 100, 7, 200, 16))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (self.WIDTH//2 - 100, 7, 200*health_pct, 16))
        health_text = self.font_ui.render(f"{int(self.base_health)}/{self.max_base_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH//2 - health_text.get_width()//2, 7))

        # Selected Tower UI (Bottom Left)
        spec = self.TOWER_SPECS[self.selected_tower_type_idx]
        tower_ui_surf = pygame.Surface((150, 60), pygame.SRCALPHA)
        tower_ui_surf.fill((0,0,0,100))
        name_text = self.font_ui.render(spec['name'], True, spec['color'])
        cost_text = self.font_ui.render(f"Cost: {spec['cost']}", True, self.COLOR_TEXT)
        tower_ui_surf.blit(name_text, (10, 10))
        tower_ui_surf.blit(cost_text, (10, 30))
        self.screen.blit(tower_ui_surf, (10, self.HEIGHT - 70))
        
        # Game Over / Win
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,150))
            self.screen.blit(overlay, (0,0))
            msg = "YOU WON!" if self.game_won else "GAME OVER"
            color = (0, 255, 0) if self.game_won else (255, 0, 0)
            end_text = self.font_game_over.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH//2 - end_text.get_width()//2, self.HEIGHT//2 - end_text.get_height()//2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "money": self.money,
            "base_health": self.base_health,
        }

    def _create_particles(self, pos, color, count, speed_max, life_max):
        for _ in range(count):
            self.particles.append({
                "pos": list(pos),
                "vel": [(self.np_random.random()-0.5)*speed_max*2, (self.np_random.random()-0.5)*speed_max*2],
                "life": self.np_random.integers(life_max//2, life_max),
                "max_life": life_max,
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    running = True
    terminated = False
    
    # Game loop
    while running:
        action = [0, 0, 0] # no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print info for debugging
            # print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Health: {info['base_health']}")

        # Convert observation back to a Pygame Surface and draw it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS for human play

    env.close()