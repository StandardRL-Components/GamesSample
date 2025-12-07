import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:41:45.378041
# Source Brief: brief_02933.md
# Brief Index: 2933
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your core from incoming celestial threats by placing and upgrading powerful constellations on the grid."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a constellation and shift to upgrade an existing one."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 14, 9
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2

        self.MAX_STEPS = 1000
        self.CORE_START_HEALTH = 10
        self.STARTING_ENERGY = 100
        self.CLONE_COST = 150
        self.ACTION_COOLDOWN_STEPS = 5

        # --- Colors ---
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_GRID = (30, 20, 80)
        self.COLOR_CORE = (0, 255, 150)
        self.COLOR_CORE_GLOW = (0, 255, 150, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 50, 50, 60)
        self.COLOR_CONSTELLATION = (0, 200, 255)
        self.COLOR_CONSTELLATION_GLOW = (0, 200, 255, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_INVALID = (255, 0, 0)
        self.COLOR_ENERGY = (255, 220, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_UI_BG = (25, 20, 60, 180)

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
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Game Data Structures ---
        self.CONSTELLATION_CARDS = [
            {'name': 'Tri', 'shape': [(0, 0), (1, 0), (0, 1)], 'cost': 50},
            {'name': 'Line', 'shape': [(0, 0), (1, 0), (2, 0)], 'cost': 60},
            {'name': 'Square', 'shape': [(0, 0), (1, 0), (0, 1), (1, 1)], 'cost': 80},
            {'name': 'Cross', 'shape': [(0, 0), (-1, 0), (1, 0), (0, 1)], 'cost': 90}
        ]
        self.ENEMY_PATH_WAYPOINTS = [
            (-20, self.HEIGHT // 2),
            (self.GRID_X_OFFSET + 100, self.HEIGHT // 2),
            (self.GRID_X_OFFSET + 100, self.GRID_Y_OFFSET + 50),
            (self.WIDTH - self.GRID_X_OFFSET - 100, self.GRID_Y_OFFSET + 50),
            (self.WIDTH - self.GRID_X_OFFSET - 100, self.HEIGHT - self.GRID_Y_OFFSET - 50),
            (self.GRID_X_OFFSET + self.GRID_WIDTH, self.HEIGHT - self.GRID_Y_OFFSET - 50)
        ]
        self.CORE_POS = self.ENEMY_PATH_WAYPOINTS[-1]

        # --- State Variables ---
        self.steps = 0
        self.total_reward = 0.0
        self.game_over = False
        self.core_health = 0
        self.celestial_energy = 0
        self.cursor_pos = [0, 0]
        self.current_card = None
        self.constellations = []
        self.enemies = []
        self.particles = []
        self.background_stars = []
        self.action_cooldowns = {'place': 0, 'clone': 0}
        self.enemy_spawn_timer = 0.0
        self.enemy_spawn_rate = 0.0
        self.enemy_base_health = 0
        self.last_action_feedback = {'type': None, 'timer': 0}

        self._generate_background_stars()
        # self.reset() # reset is called by the environment runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.total_reward = 0.0
        self.game_over = False
        self.core_health = self.CORE_START_HEALTH
        self.celestial_energy = self.STARTING_ENERGY
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.constellations = []
        self.enemies = []
        self.particles = []
        self.action_cooldowns = {'place': 0, 'clone': 0}
        
        self.enemy_spawn_timer = 0.0
        self.enemy_spawn_rate = 1.0 / 10.0 # Initial: 1 every 10 steps
        self.enemy_base_health = 1

        self._draw_new_card()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        step_reward = 0.01  # Survival reward

        # --- Handle Actions ---
        action_reward = self._handle_actions(action)
        step_reward += action_reward

        # --- Update Game State ---
        self._update_difficulty()
        self._update_spawning()
        self._update_enemies()
        
        damage_reward = self._update_constellations_and_damage()
        step_reward += damage_reward
        
        self._update_particles()

        # --- Check Termination ---
        terminated = (self.core_health <= 0)
        truncated = (self.steps >= self.MAX_STEPS)
        if terminated or truncated:
            self.game_over = True
            if self.core_health <= 0:
                step_reward -= 100  # Penalty for losing
            elif self.steps >= self.MAX_STEPS:
                step_reward += 100  # Reward for winning

        self.total_reward += step_reward

        return (
            self._get_observation(),
            step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Private Helper Methods: Game Logic ---

    def _handle_actions(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # Update cooldowns
        for key in self.action_cooldowns:
            self.action_cooldowns[key] = max(0, self.action_cooldowns[key] - 1)
        if self.last_action_feedback['timer'] > 0:
            self.last_action_feedback['timer'] -= 1

        # Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # Place Constellation
        if space_pressed and self.action_cooldowns['place'] == 0:
            cost = self.current_card['cost']
            if self.celestial_energy >= cost and self._is_placement_valid(self.cursor_pos, self.current_card):
                # // sfx: place_constellation
                self.constellations.append({
                    'grid_pos': list(self.cursor_pos),
                    'card': self.current_card,
                    'level': 1,
                    'anim_timer': 1.0
                })
                self.celestial_energy -= cost
                reward += 1.0
                self._draw_new_card()
                self.action_cooldowns['place'] = self.ACTION_COOLDOWN_STEPS
            else:
                # // sfx: action_fail
                self.last_action_feedback = {'type': 'fail', 'timer': 10}


        # Clone Constellation
        if shift_pressed and self.action_cooldowns['clone'] == 0:
            target_constellation = None
            for const in self.constellations:
                # Check if cursor is on any part of the constellation shape
                for offset in const['card']['shape']:
                    gx, gy = const['grid_pos'][0] + offset[0], const['grid_pos'][1] + offset[1]
                    if [gx, gy] == self.cursor_pos:
                        target_constellation = const
                        break
                if target_constellation:
                    break
            
            if target_constellation and self.celestial_energy >= self.CLONE_COST:
                # // sfx: clone_success
                target_constellation['level'] += 1
                target_constellation['anim_timer'] = 1.0
                self.celestial_energy -= self.CLONE_COST
                reward += 2.0
                self.action_cooldowns['clone'] = self.ACTION_COOLDOWN_STEPS
                self._create_particles(self._grid_to_screen(target_constellation['grid_pos']), 30, self.COLOR_CONSTELLATION, 1.5)
            else:
                # // sfx: action_fail
                self.last_action_feedback = {'type': 'fail', 'timer': 10}

        return reward

    def _is_placement_valid(self, grid_pos, card):
        for offset in card['shape']:
            gx, gy = grid_pos[0] + offset[0], grid_pos[1] + offset[1]
            if not (0 <= gx < self.GRID_COLS and 0 <= gy < self.GRID_ROWS):
                return False # Part of shape is out of bounds
            
            # Check for overlap with existing constellations
            for const in self.constellations:
                for other_offset in const['card']['shape']:
                    other_gx, other_gy = const['grid_pos'][0] + other_offset[0], const['grid_pos'][1] + other_offset[1]
                    if gx == other_gx and gy == other_gy:
                        return False # Overlapping
        return True

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_spawn_rate = min(0.1, self.enemy_spawn_rate + 0.001)
        if self.steps > 0 and self.steps % 500 == 0:
            self.enemy_base_health += 1

    def _update_spawning(self):
        self.enemy_spawn_timer += self.enemy_spawn_rate
        if self.enemy_spawn_timer >= 1.0:
            self.enemy_spawn_timer -= 1.0
            # // sfx: enemy_spawn
            self.enemies.append({
                'pos': list(self.ENEMY_PATH_WAYPOINTS[0]),
                'health': self.enemy_base_health,
                'max_health': self.enemy_base_health,
                'path_index': 0,
                't': 0.0,
                'speed': self.np_random.uniform(0.015, 0.025)
            })

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            if enemy['path_index'] >= len(self.ENEMY_PATH_WAYPOINTS) - 1:
                # // sfx: core_hit
                self.core_health -= 1
                self._create_particles(enemy['pos'], 20, self.COLOR_ENEMY, 2.0)
                self.enemies.remove(enemy)
                continue

            p1 = self.ENEMY_PATH_WAYPOINTS[enemy['path_index']]
            p2 = self.ENEMY_PATH_WAYPOINTS[enemy['path_index'] + 1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            
            if dist > 0:
                enemy['t'] += enemy['speed'] * (self.CELL_SIZE / dist)
            
            if enemy['t'] >= 1.0:
                enemy['t'] = 0.0
                enemy['path_index'] += 1
            else:
                enemy['pos'][0] = p1[0] + (p2[0] - p1[0]) * enemy['t']
                enemy['pos'][1] = p1[1] + (p2[1] - p1[1]) * enemy['t']

    def _update_constellations_and_damage(self):
        reward = 0
        for const in self.constellations:
            if const['anim_timer'] > 0:
                const['anim_timer'] -= 0.05

            damage = const['level'] * 0.1 # Damage per step
            
            # Get all line segments for this constellation
            world_points = [self._grid_to_screen([const['grid_pos'][0] + p[0], const['grid_pos'][1] + p[1]]) for p in const['card']['shape']]
            lines = []
            if len(world_points) > 1:
                for i in range(len(world_points)):
                    for j in range(i + 1, len(world_points)):
                        lines.append((world_points[i], world_points[j]))

            for enemy in self.enemies[:]:
                for p1, p2 in lines:
                    dist_sq = self._dist_point_to_segment_sq(enemy['pos'], p1, p2)
                    damage_radius = 10 + const['level'] * 2
                    if dist_sq < damage_radius ** 2:
                        enemy['health'] -= damage
                        # // sfx: enemy_damage_tick
                        self._create_particles(enemy['pos'], 1, self.COLOR_CONSTELLATION, 0.5, life=5)
                        if enemy['health'] <= 0:
                            # // sfx: enemy_destroyed
                            reward += 0.1
                            self.celestial_energy += 10 + self.enemy_base_health
                            self._create_particles(enemy['pos'], 40, self.COLOR_ENERGY, 1.0)
                            if enemy in self.enemies:
                                self.enemies.remove(enemy)
                            break # Move to next enemy
                if enemy['health'] <= 0:
                    continue # Enemy already handled
        return reward
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    # --- Private Helper Methods: Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_grid()
        self._render_core()
        self._render_constellations()
        self._render_enemies()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.background_stars:
            pos, size, alpha = star
            color = (alpha, alpha, alpha)
            self.screen.set_at(pos, color)
    
    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT))

    def _render_core(self):
        health_ratio = max(0, self.core_health / self.CORE_START_HEALTH)
        radius = int(15 + 10 * health_ratio)
        color = (
            int(self.COLOR_CORE[0] * (1-health_ratio) + self.COLOR_ENEMY[0] * health_ratio),
            int(self.COLOR_CORE[1] * health_ratio),
            int(self.COLOR_CORE[2] * health_ratio)
        )
        glow_color = (color[0], color[1], color[2], 50)
        
        self._draw_glow_circle(self.screen, self.CORE_POS, radius * 2, glow_color)
        self._draw_glow_circle(self.screen, self.CORE_POS, radius, color)

    def _render_constellations(self):
        for const in self.constellations:
            level = const['level']
            points = [self._grid_to_screen([const['grid_pos'][0] + p[0], const['grid_pos'][1] + p[1]]) for p in const['card']['shape']]
            
            line_width = 1 + level // 2
            star_radius = 4 + level
            
            # Draw connecting lines
            if len(points) > 1:
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        p1 = points[i]
                        p2 = points[j]
                        pygame.draw.aaline(self.screen, self.COLOR_CONSTELLATION, p1, p2, line_width)
            
            # Draw stars
            for p in points:
                self._draw_glow_circle(self.screen, p, star_radius, self.COLOR_CONSTELLATION)
            
            # Animation pulse
            if const['anim_timer'] > 0:
                pulse_radius = (1.0 - const['anim_timer']) * self.CELL_SIZE * 1.5
                pulse_alpha = int(const['anim_timer'] * 100)
                center = self._grid_to_screen(const['grid_pos'])
                pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), int(pulse_radius), (*self.COLOR_CONSTELLATION, pulse_alpha))

    def _render_enemies(self):
        for enemy in self.enemies:
            radius = int(4 + enemy['max_health'])
            self._draw_glow_circle(self.screen, enemy['pos'], radius * 2, self.COLOR_ENEMY_GLOW)
            self._draw_glow_circle(self.screen, enemy['pos'], radius, self.COLOR_ENEMY)
            
            # Health bar
            health_ratio = max(0, enemy['health'] / enemy['max_health'])
            bar_w = radius * 2
            bar_h = 4
            bar_x = enemy['pos'][0] - bar_w / 2
            bar_y = enemy['pos'][1] - radius - bar_h - 2
            pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, bar_w * health_ratio, bar_h))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                self._draw_glow_circle(self.screen, p['pos'], size, color)

    def _render_cursor(self):
        is_valid = self._is_placement_valid(self.cursor_pos, self.current_card)
        can_afford = self.celestial_energy >= self.current_card['cost']
        color = self.COLOR_CURSOR if (is_valid and can_afford) else self.COLOR_CURSOR_INVALID

        # Draw ghost of constellation
        for offset in self.current_card['shape']:
            pos = self._grid_to_screen([self.cursor_pos[0] + offset[0], self.cursor_pos[1] + offset[1]])
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 5, (*color, 100))
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 5, (*color, 100))

        # Draw main cursor square
        center_pos = self._grid_to_screen(self.cursor_pos)
        rect = pygame.Rect(center_pos[0] - self.CELL_SIZE//2, center_pos[1] - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, (*color, 50), rect, 2)
        
        # Action feedback
        if self.last_action_feedback['timer'] > 0 and self.last_action_feedback['type'] == 'fail':
            alpha = int(200 * (self.last_action_feedback['timer'] / 10))
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((255, 0, 0, alpha))
            self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        # UI Panel for card
        panel_rect = pygame.Rect(self.WIDTH - 150, 10, 140, 120)
        self._draw_blurry_rect(self.screen, panel_rect, self.COLOR_UI_BG)
        
        # Current Card
        if self.current_card:
            card_title = self.font_medium.render(f"{self.current_card['name']}", True, self.COLOR_TEXT)
            self.screen.blit(card_title, (panel_rect.x + 10, panel_rect.y + 10))
            
            cost_text = self.font_small.render(f"Cost: {self.current_card['cost']}", True, self.COLOR_ENERGY)
            self.screen.blit(cost_text, (panel_rect.x + 10, panel_rect.y + 35))

        clone_text = self.font_small.render(f"Clone: {self.CLONE_COST}", True, self.COLOR_CONSTELLATION)
        self.screen.blit(clone_text, (panel_rect.x + 10, panel_rect.y + 55))

        # Core Health
        health_text = self.font_medium.render(f"Core: {self.core_health}/{self.CORE_START_HEALTH}", True, self.COLOR_CORE)
        self.screen.blit(health_text, (10, 10))

        # Energy
        energy_text = self.font_medium.render(f"Energy: {int(self.celestial_energy)}", True, self.COLOR_ENERGY)
        self.screen.blit(energy_text, (10, 35))

        # Timer
        time_text = self.font_large.render(f"{self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(centerx=self.WIDTH/2, y=10)
        self.screen.blit(time_text, time_rect)

        # Score
        score_text = self.font_medium.render(f"Score: {self.total_reward:.2f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 160, 10))
        self.screen.blit(score_text, score_rect)

    # --- Private Helper Methods: Utilities ---

    def _get_info(self):
        return {"score": self.total_reward, "steps": self.steps, "core_health": self.core_health}

    def _draw_new_card(self):
        self.current_card = self.np_random.choice(self.CONSTELLATION_CARDS)

    def _grid_to_screen(self, grid_pos):
        x = self.GRID_X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        y = self.GRID_Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        return [x, y]

    def _generate_background_stars(self):
        self.background_stars = []
        for _ in range(200):
            pos = (random.randint(0, self.WIDTH-1), random.randint(0, self.HEIGHT-1))
            size = random.uniform(0.5, 1.5)
            alpha = random.randint(50, 150)
            self.background_stars.append((pos, size, alpha))

    def _create_particles(self, pos, count, color, speed_mult=1.0, life=30):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, 3.0) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(life//2, life),
                'max_life': life,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    @staticmethod
    def _draw_glow_circle(surface, pos, radius, color):
        x, y = int(pos[0]), int(pos[1])
        r = int(radius)
        if r <= 0: return

        if len(color) == 4: # RGBA
            glow_color = (*color[:3], color[3] // 4)
            main_color = (*color[:3], color[3])
            
            # Glow effect
            pygame.gfxdraw.filled_circle(surface, x, y, r, glow_color)
            pygame.gfxdraw.aacircle(surface, x, y, r, glow_color)
            
            # Main circle
            inner_radius = int(r * 0.7)
            if inner_radius > 0:
                pygame.gfxdraw.filled_circle(surface, x, y, inner_radius, main_color)
                pygame.gfxdraw.aacircle(surface, x, y, inner_radius, main_color)
        else: # RGB
            pygame.gfxdraw.filled_circle(surface, x, y, r, color)
            pygame.gfxdraw.aacircle(surface, x, y, r, color)
            
    @staticmethod
    def _draw_blurry_rect(surface, rect, color):
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        s.fill(color)
        surface.blit(s, rect.topleft)

    @staticmethod
    def _dist_point_to_segment_sq(p, a, b):
        px, py = p
        ax, ay = a
        bx, by = b
        
        # Vector AB
        abx, aby = bx - ax, by - ay
        # Vector AP
        apx, apy = px - ax, py - ay
        
        ab_len_sq = abx*abx + aby*aby
        if ab_len_sq == 0:
            return apx*apx + apy*apy

        t = (apx * abx + apy * aby) / ab_len_sq
        t = max(0, min(1, t)) # Clamp to segment
        
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        
        dx, dy = px - closest_x, py - closest_y
        return dx*dx + dy*dy

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This part of the code is for human interaction and will not be run by the autograder.
    # It is included to allow for manual testing and visualization of the environment.
    
    # Un-set the dummy video driver to allow for display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Constellation Defense")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    running = True
    while running:
        # --- Action Mapping for Human Play ---
        movement = 0 # none
        space_pressed = 0
        shift_pressed = 0
        
        # This event loop is for human interaction, not for the agent.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
                if event.key == pygame.K_q:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_pressed = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_pressed = 1
            
        action = [movement, space_pressed, shift_pressed]

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        if done:
            font = pygame.font.SysFont("Consolas", 50, bold=True)
            text = font.render("GAME OVER", True, (255, 255, 255))
            text_rect = text.get_rect(center=(env.WIDTH / 2, env.HEIGHT / 2))
            screen.blit(text, text_rect)
            
            font_small = pygame.font.SysFont("Consolas", 20)
            text_small = font_small.render("Press 'R' to Reset", True, (200, 200, 200))
            text_small_rect = text_small.get_rect(center=(env.WIDTH / 2, env.HEIGHT / 2 + 40))
            screen.blit(text_small, text_small_rect)

        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()