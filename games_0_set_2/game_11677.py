import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:24:59.913350
# Source Brief: brief_01677.md
# Brief Index: 1677
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for "Fractal Farm," a simulation game where the player
    cultivates procedurally generated fractal farms. The goal is to maximize crop
    yield by manipulating time flow and managing a network of resource portals.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    game_description = (
        "Cultivate procedurally generated fractal farms and maximize crop yield by managing a "
        "network of resource portals and manipulating the flow of time."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to select a portal. Press space to cycle the selected portal's "
        "target. Hold shift to fast-forward time."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and World Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("monospace", 16)
            self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 30)


        # Colors
        self.COLOR_BG = (15, 18, 23)
        self.COLOR_GRID = (30, 35, 45)
        self.COLOR_PORTAL = (0, 150, 255)
        self.COLOR_PORTAL_GLOW = (0, 150, 255, 50)
        self.COLOR_PORTAL_SELECTED = (255, 255, 0)
        self.COLOR_PORTAL_LINK = (0, 150, 255, 100)
        self.COLOR_CROP_START = (0, 100, 20)
        self.COLOR_CROP_MID = (255, 200, 0)
        self.COLOR_CROP_END = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_TIME_FX = (255, 255, 255, 20)

        # Game Mechanics
        self.NORMAL_TIME_MULTIPLIER = 1.0
        self.FAST_TIME_MULTIPLIER = 4.0
        self.CROP_GROWTH_RATE = 0.002
        self.CROP_VALUE = 1.0
        self.PARTICLE_SPEED = 2.0

        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.time_flow_speed = 1.0
        self.farm_plots = []
        self.crops = []
        self.portals = []
        self.particles = []
        self.selected_portal_idx = 0
        self.prev_space_held = False
        self.unlocked_pattern_level = 0
        self.portal_capacity_level = 0
        self.last_pattern_unlock_score = 0
        self.last_portal_upgrade_score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_flow_speed = self.NORMAL_TIME_MULTIPLIER
        self.particles = []
        self.selected_portal_idx = 0
        self.prev_space_held = False
        self.last_pattern_unlock_score = 0
        self.last_portal_upgrade_score = 0
        self.unlocked_pattern_level = 0
        self.portal_capacity_level = 0

        self._generate_fractal_farm()
        self._initialize_crops()
        self._initialize_portals()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        self._handle_actions(action)

        harvested_value = self._update_crops()
        if harvested_value > 0:
            # SFX: Harvest
            self._process_harvest(harvested_value)
        
        score_increase = self._update_particles()
        if score_increase > 0:
            self.score += score_increase
            # Continuous reward for harvesting
            reward += score_increase * 0.1 # +0.1 per unit of crop
            # SFX: Resource_Deposit

        # Check for Progression Unlocks
        if self.score >= self.last_pattern_unlock_score + 5000:
            self.unlocked_pattern_level = min(self.unlocked_pattern_level + 1, 3) # Max 4 patterns
            self.last_pattern_unlock_score += 5000
            reward += 1.0 # Event-based reward
        
        if self.score >= self.last_portal_upgrade_score + 2500:
            self.portal_capacity_level += 1
            self.last_portal_upgrade_score += 2500
            reward += 0.5 # Event-based reward

        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS
        truncated = False
        if terminated:
            reward += 10.0 # Goal-oriented terminal reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self.time_flow_speed = self.FAST_TIME_MULTIPLIER if shift_held else self.NORMAL_TIME_MULTIPLIER
        
        # Debounced movement action for portal selection
        if movement != 0 and len(self.portals) > 1:
            current_pos = self.portals[self.selected_portal_idx]['pos']
            best_target_idx = -1
            min_dist_sq = float('inf')

            for i, portal in enumerate(self.portals):
                if i == self.selected_portal_idx: continue
                dx, dy = portal['pos'][0] - current_pos[0], portal['pos'][1] - current_pos[1]
                
                is_candidate = False
                if movement == 1 and dy < 0 and abs(dy) > abs(dx): is_candidate = True # Up
                elif movement == 2 and dy > 0 and abs(dy) > abs(dx): is_candidate = True # Down
                elif movement == 3 and dx < 0 and abs(dx) > abs(dy): is_candidate = True # Left
                elif movement == 4 and dx > 0 and abs(dx) > abs(dy): is_candidate = True # Right
                
                if is_candidate:
                    dist_sq = dx**2 + dy**2
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_target_idx = i
            
            if best_target_idx != -1:
                self.selected_portal_idx = best_target_idx
        
        if space_held and not self.prev_space_held and len(self.portals) > 1:
            # SFX: Portal_Configure
            portal = self.portals[self.selected_portal_idx]
            portal['target_idx'] = (portal['target_idx'] + 1) % len(self.portals)
            if portal['target_idx'] == self.selected_portal_idx:
                portal['target_idx'] = (portal['target_idx'] + 1) % len(self.portals)
        
        self.prev_space_held = space_held

    def _update_crops(self):
        total_harvested_value = 0
        for crop in self.crops:
            if crop['state'] == 'growing':
                crop['maturity'] = min(1.0, crop['maturity'] + self.CROP_GROWTH_RATE * self.time_flow_speed)
                if crop['maturity'] >= 1.0:
                    crop['state'] = 'harvest_ready'
                    # SFX: Crop_Ready
            
            if crop['state'] == 'harvest_ready':
                total_harvested_value += self.CROP_VALUE
                crop['maturity'] = 0.0
                crop['state'] = 'growing'
        return total_harvested_value

    def _process_harvest(self, value):
        if not self.portals: return
        # Distribute value among all portals for visual effect
        value_per_portal = value / len(self.portals)
        for portal in self.portals:
            target_portal = self.portals[portal['target_idx']]
            self._spawn_particle(
                portal['pos'], 
                target_portal['pos'], 
                value_per_portal,
                self.COLOR_CROP_END
            )

    def _update_particles(self):
        score_increase = 0
        for p in self.particles[:]:
            dx, dy = p['target_pos'][0] - p['pos'][0], p['target_pos'][1] - p['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < self.PARTICLE_SPEED:
                score_increase += p['value']
                self.particles.remove(p)
            else:
                p['pos'][0] += (dx / dist) * self.PARTICLE_SPEED
                p['pos'][1] += (dy / dist) * self.PARTICLE_SPEED
                p['life'] -= 1
                if p['life'] <= 0:
                    self.particles.remove(p)
        return score_increase
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_multiplier": self.time_flow_speed,
            "unlocked_patterns": self.unlocked_pattern_level,
        }

    def _render_background(self):
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        
        if self.time_flow_speed > self.NORMAL_TIME_MULTIPLIER:
            for _ in range(5):
                start_pos = (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT))
                end_pos = (start_pos[0] + self.np_random.integers(-20, 21), start_pos[1] + self.np_random.integers(-20, 21))
                pygame.draw.line(self.screen, self.COLOR_TIME_FX, start_pos, end_pos, 1)

    def _render_game(self):
        for p1 in self.farm_plots:
            pygame.draw.circle(self.screen, self.COLOR_GRID, p1, 2)

        for crop in self.crops:
            color = self._interpolate_color(self.COLOR_CROP_START, self.COLOR_CROP_MID, self.COLOR_CROP_END, crop['maturity'])
            size = int(2 + 4 * crop['maturity'])
            pygame.draw.circle(self.screen, color, crop['pos'], size)

        for i, portal in enumerate(self.portals):
            if portal['target_idx'] != i:
                pygame.draw.line(self.screen, self.COLOR_PORTAL_LINK, portal['pos'], self.portals[portal['target_idx']]['pos'], 1)

        for i, portal in enumerate(self.portals):
            pos = (int(portal['pos'][0]), int(portal['pos'][1]))
            color = self.COLOR_PORTAL_SELECTED if i == self.selected_portal_idx else self.COLOR_PORTAL
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, self.COLOR_PORTAL_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, self.COLOR_PORTAL_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, color)
            
            id_text = self.font_small.render(chr(65 + i), True, self.COLOR_TEXT)
            self.screen.blit(id_text, (pos[0] - id_text.get_width() // 2, pos[1] - id_text.get_height() // 2))

        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), 3)

    def _render_ui(self):
        score_text = self.font_large.render(f"Yield: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_small.render(f"Step: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        time_text = self.font_small.render(f"Time Flow: x{self.time_flow_speed:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 40))

        if self.portals:
            portal = self.portals[self.selected_portal_idx]
            target = self.portals[portal['target_idx']]
            portal_info = f"Selected: Portal {chr(65+self.selected_portal_idx)} -> Portal {chr(65+portal['target_idx'])}"
            portal_text = self.font_small.render(portal_info, True, self.COLOR_PORTAL_SELECTED)
            self.screen.blit(portal_text, (10, self.HEIGHT - 30))

    def _generate_fractal_farm(self):
        self.farm_plots = []
        center_x, center_y = self.WIDTH / 2, self.HEIGHT / 2
        
        patterns = [
            {'size': 100, 'depth': 3, 'branches': [(0,1), (0,-1), (1,0), (-1,0)], 'angle': 0}, # Cross
            {'size': 80, 'depth': 3, 'branches': [(1,1), (1,-1), (-1,1), (-1,-1)], 'angle': 0}, # X-shape
            {'size': 60, 'depth': 4, 'branches': [(0,-1)], 'angle': math.pi/6}, # Tree
            {'size': 120, 'depth': 2, 'branches': [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)], 'angle': 0} # Dense
        ]
        pattern = patterns[self.unlocked_pattern_level]
        start_y = self.HEIGHT - 50 if self.unlocked_pattern_level == 2 else center_y
        
        self._recursive_fractal(center_x, start_y, pattern['size'], pattern['depth'], pattern['branches'], pattern['angle'])

    def _recursive_fractal(self, x, y, size, depth, branches, angle_offset):
        if depth == 0 or size < 10:
            if 20 < x < self.WIDTH - 20 and 20 < y < self.HEIGHT - 20:
                 if (int(x), int(y)) not in self.farm_plots: self.farm_plots.append((int(x), int(y)))
            return

        for dx, dy in branches:
            angle = math.atan2(dy, dx)
            new_x = x + math.cos(angle + angle_offset) * size
            new_y = y + math.sin(angle + angle_offset) * size
            self._recursive_fractal(new_x, new_y, size * 0.6, depth - 1, branches, angle_offset)
        
        if 20 < x < self.WIDTH - 20 and 20 < y < self.HEIGHT - 20:
             if (int(x), int(y)) not in self.farm_plots: self.farm_plots.append((int(x), int(y)))
    
    def _initialize_crops(self):
        self.crops = []
        if not self.farm_plots: return
        for pos in self.farm_plots:
            self.crops.append({
                'pos': pos,
                'maturity': self.np_random.random() * 0.2,
                'state': 'growing'
            })

    def _initialize_portals(self):
        self.portals = []
        num_portals = 2 + self.unlocked_pattern_level
        
        positions = [
            (self.WIDTH * 0.2, self.HEIGHT * 0.2), (self.WIDTH * 0.8, self.HEIGHT * 0.8),
            (self.WIDTH * 0.8, self.HEIGHT * 0.2), (self.WIDTH * 0.2, self.HEIGHT * 0.8),
            (self.WIDTH * 0.5, self.HEIGHT * 0.5)
        ]
        
        num_to_spawn = min(num_portals, len(positions))
        for i in range(num_to_spawn):
            self.portals.append({
                'id': i, 'pos': positions[i], 'target_idx': (i + 1) % num_to_spawn
            })

    def _spawn_particle(self, start_pos, target_pos, value, color):
        self.particles.append({
            'pos': list(start_pos), 'target_pos': target_pos,
            'value': value, 'color': color, 'life': 300
        })

    def _interpolate_color(self, c1, c2, c3, t):
        t = max(0, min(1, t))
        if t < 0.5:
            return tuple(int(c1[i] + (c2[i] - c1[i]) * t * 2) for i in range(3))
        return tuple(int(c2[i] + (c3[i] - c2[i]) * (t - 0.5) * 2) for i in range(3))

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = GameEnv()
    # The following is a basic rendering loop for testing purposes
    # It is not part of the required Gymnasium API
    
    # Unset the dummy video driver to allow for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # Quit the headless instance
    pygame.init() # Re-init with display
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fractal Farm")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    running = True
    while running:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}")
            obs, info = env.reset()
        
        clock.tick(env.metadata["render_fps"])

    env.close()