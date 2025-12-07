import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:44:11.631806
# Source Brief: brief_01182.md
# Brief Index: 1182
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Expand your quantum colony across a hex grid, collecting resources while avoiding hostile sentinels. "
        "Use special abilities to protect and guide your growth to dominate the grid."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to influence colony growth direction. "
        "Press space to terraform new tiles and hold shift to activate the stealth field."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.WIN_SCORE = 1000
        
        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_GRID = (20, 40, 60)
        self.COLOR_COLONY = (0, 150, 255)
        self.COLOR_COLONY_GLOW = (0, 100, 200)
        self.COLOR_RESOURCE = (50, 255, 150)
        self.COLOR_RESOURCE_GLOW = (50, 200, 100)
        self.COLOR_SENTINEL = (255, 50, 100)
        self.COLOR_SENTINEL_GLOW = (200, 50, 80)
        self.COLOR_SCAN = (255, 100, 150)
        self.COLOR_TERRAFORM = (120, 50, 200)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        # Hex grid parameters
        self.hex_radius = 10
        self.hex_width = self.hex_radius * 2
        self.hex_height = math.sqrt(3) * self.hex_radius

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.colony_units = set()
        self.resource_nodes = set()
        self.terraformed_tiles = set()
        self.sentinels = []
        self.particles = []
        self.growth_bias = 0
        self.colony_expansion_cooldown = 0
        self.upgrades_unlocked = {}
        self.active_effects = {}
        self.reward_this_step = 0
        self.last_action_feedback = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0 # This represents colony size in the brief
        self.game_over = False
        
        # Reset colony
        self.colony_units = {(0, 0)}
        self.score = 1
        
        # Reset resources
        self.resource_nodes = set()
        for _ in range(50):
            q = self.np_random.integers(-18, 18)
            r = self.np_random.integers(-10, 10)
            if (q, r) != (0, 0):
                self.resource_nodes.add((q, r))

        # Reset other elements
        self.terraformed_tiles = set()
        self.particles = []
        self.last_action_feedback = {}
        
        # Reset sentinels
        self.sentinels = []
        for i in range(3):
            self.sentinels.append(self._create_sentinel())
            
        # Reset progression
        self.growth_bias = 0
        self.colony_expansion_cooldown = 0
        self.upgrades_unlocked = {'stealth_field': False}
        self.active_effects = {'stealth_field_timer': 0}
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        self.steps += 1
        
        # 1. Handle player action
        self._handle_action(action)

        # 2. Update game state
        self._update_sentinels()
        self._update_colony_growth()
        self._update_particles()
        self._update_effects()

        # 3. Check for termination
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.growth_bias = movement

        # Action priority: Shift > Space > Movement
        if shift_held and self.upgrades_unlocked.get('stealth_field') and self.active_effects['stealth_field_timer'] <= 0:
            # sound: upgrade_activate.wav
            self.active_effects['stealth_field_timer'] = 150 # 5 seconds at 30fps
            self.last_action_feedback = {'type': 'upgrade', 'pos': (self.WIDTH // 2, self.HEIGHT // 2), 'timer': 30}
        elif space_held:
            self._execute_terraform()
        
        if movement != 0:
            self.last_action_feedback = {'type': 'bias', 'dir': movement, 'timer': 15}

    def _execute_terraform(self):
        if not self.colony_units: return
        
        # Find largest cluster - for this game, we assume one cluster
        # Find a valid, non-terraformed neighbor to terraform
        all_neighbors = set()
        for unit in self.colony_units:
            for neighbor in self._get_hex_neighbors(unit[0], unit[1]):
                if neighbor not in self.colony_units and neighbor not in self.terraformed_tiles:
                    all_neighbors.add(neighbor)
        
        if all_neighbors:
            # sound: terraform.wav
            tile_to_terraform = list(all_neighbors)[self.np_random.integers(len(all_neighbors))]
            self.terraformed_tiles.add(tuple(tile_to_terraform))
            pos = self._hex_to_pixel(tile_to_terraform[0], tile_to_terraform[1])
            self._create_particles(pos, 20, self.COLOR_TERRAFORM, 1, 3, 20)
            self.last_action_feedback = {'type': 'terraform', 'pos': pos, 'timer': 30}


    def _update_sentinels(self):
        base_speed_increase = 0.05 * (self.steps // 200)
        max_speed = 2.5
        
        for s in self.sentinels:
            s['speed'] = min(max_speed, s['base_speed'] + base_speed_increase)
            
            # Move towards target
            dx = s['target'][0] - s['pos'][0]
            dy = s['target'][1] - s['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < s['speed']:
                s['target'] = (self.np_random.uniform(50, self.WIDTH - 50), self.np_random.uniform(50, self.HEIGHT - 50))
            else:
                s['pos'] = (s['pos'][0] + (dx / dist) * s['speed'], s['pos'][1] + (dy / dist) * s['speed'])
            
            # Scan logic
            s['scan_timer'] = (s['scan_timer'] + 1) % s['scan_period']
            if s['scan_timer'] == 0 and self.active_effects['stealth_field_timer'] <= 0:
                # sound: sentinel_scan.wav
                units_to_remove = set()
                scan_radius = s['scan_radius']
                for unit_hex in self.colony_units:
                    if unit_hex in self.terraformed_tiles:
                        continue
                    
                    unit_pos = self._hex_to_pixel(unit_hex[0], unit_hex[1])
                    if math.hypot(unit_pos[0] - s['pos'][0], unit_pos[1] - s['pos'][1]) < scan_radius:
                        units_to_remove.add(unit_hex)
                        self._create_particles(unit_pos, 10, self.COLOR_SENTINEL, 1, 3, 15)

                if units_to_remove:
                    # sound: unit_destroyed.wav
                    self.colony_units -= units_to_remove
                    self.reward_this_step -= 0.1 * len(units_to_remove)
                    self.score = len(self.colony_units)

    def _update_colony_growth(self):
        self.colony_expansion_cooldown -= 1
        if self.colony_expansion_cooldown > 0 or not self.colony_units:
            return
            
        self.colony_expansion_cooldown = 5 # Grow every 5 steps

        # Find all valid expansion candidates
        candidates = {}
        for unit in self.colony_units:
            for i, neighbor in enumerate(self._get_hex_neighbors(unit[0], unit[1])):
                if neighbor not in self.colony_units:
                    # Direction index for bias (1=E, 2=SE, 3=SW, 4=W, 5=NW, 6=NE)
                    # Map our movement to hex directions
                    dir_map = {1: 6, 2: 3, 3: 4, 4: 1} # Up->NE, Down->SW, Left->W, Right->E
                    
                    weight = 1.0
                    if self.growth_bias in dir_map and dir_map[self.growth_bias] == (i + 1):
                        weight = 5.0 # Strong bias
                    
                    if neighbor in candidates:
                        candidates[neighbor] += weight
                    else:
                        candidates[neighbor] = weight
        
        if not candidates: return

        # Choose a candidate based on weights
        total_weight = sum(candidates.values())
        rand_val = self.np_random.uniform(0, total_weight)
        cumulative_weight = 0
        chosen_hex = None
        for h, w in candidates.items():
            cumulative_weight += w
            if rand_val <= cumulative_weight:
                chosen_hex = h
                break
        
        if chosen_hex:
            # sound: expand.wav
            self.colony_units.add(chosen_hex)
            self.score = len(self.colony_units)
            pos = self._hex_to_pixel(chosen_hex[0], chosen_hex[1])
            self._create_particles(pos, 15, self.COLOR_COLONY, 1, 4, 20)

            # Check for resource collection
            if chosen_hex in self.resource_nodes:
                # sound: resource_collect.wav
                self.resource_nodes.remove(chosen_hex)
                self.reward_this_step += 1.0
                self._create_particles(pos, 30, self.COLOR_RESOURCE, 2, 5, 30)
            
            # Check for unlocks
            if not self.upgrades_unlocked['stealth_field'] and self.score >= 200:
                self.upgrades_unlocked['stealth_field'] = True
                self.reward_this_step += 5.0
                self.last_action_feedback = {'type': 'unlock', 'text': 'STEALTH FIELD UNLOCKED (SHIFT)', 'timer': 120}

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            p['size'] -= p['decay']
            p['life'] -= 1

    def _update_effects(self):
        if self.active_effects['stealth_field_timer'] > 0:
            self.active_effects['stealth_field_timer'] -= 1
        
        if self.last_action_feedback and self.last_action_feedback.get('timer', 0) > 0:
            self.last_action_feedback['timer'] -= 1
        else:
            self.last_action_feedback = {}

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.reward_this_step += 100.0
            return True
        if not self.colony_units and self.steps > 10: # Allow a moment at the start
            self.reward_this_step -= 100.0
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        for q in range(-25, 25):
            for r in range(-15, 15):
                pos = self._hex_to_pixel(q, r)
                if -self.hex_radius < pos[0] < self.WIDTH + self.hex_radius and \
                   -self.hex_radius < pos[1] < self.HEIGHT + self.hex_radius:
                    self._draw_hexagon(self.screen, self.COLOR_GRID, pos, self.hex_radius, 1)

        # Draw terraformed tiles
        for h in self.terraformed_tiles:
            pos = self._hex_to_pixel(h[0], h[1])
            pygame.gfxdraw.filled_polygon(self.screen, self._get_hex_points(pos, self.hex_radius), self.COLOR_TERRAFORM)
        
        # Draw resources
        for h in self.resource_nodes:
            pos = self._hex_to_pixel(h[0], h[1])
            pulse = (math.sin(self.steps * 0.1) + 1) / 2
            size = self.hex_radius * 0.5 + pulse * 2
            self._draw_glowing_rect(self.screen, self.COLOR_RESOURCE, self.COLOR_RESOURCE_GLOW, pos, size)

        # Draw colony
        for h in self.colony_units:
            pos = self._hex_to_pixel(h[0], h[1])
            self._draw_hexagon(self.screen, self.COLOR_COLONY, pos, self.hex_radius, 0, glow_color=self.COLOR_COLONY_GLOW)
        
        # Draw particles
        for p in self.particles:
            if p['size'] > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['size']))

        # Draw sentinels
        for s in self.sentinels:
            # Draw body
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 5
            self._draw_glowing_circle(self.screen, self.COLOR_SENTINEL, self.COLOR_SENTINEL_GLOW, s['pos'], 8 + pulse)
            
            # Draw scan radius
            if s['scan_timer'] < 30: # Flash the scan ring
                alpha = int(200 * (1 - s['scan_timer'] / 30))
                if alpha > 0:
                    self._draw_circle_alpha(self.screen, self.COLOR_SCAN + (alpha,), s['pos'], s['scan_radius'], 2)
        
        # Draw stealth field effect
        if self.active_effects['stealth_field_timer'] > 0:
            alpha = 50
            if self.active_effects['stealth_field_timer'] < 60: # Fade out
                alpha = int(50 * (self.active_effects['stealth_field_timer'] / 60))
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_COLONY[0], self.COLOR_COLONY[1], self.COLOR_COLONY[2], alpha))
            self.screen.blit(overlay, (0,0))


    def _render_ui(self):
        # Draw text with a shadow for readability
        def draw_text(text, font, color, pos, shadow_color, shadow_offset=(1, 1)):
            text_surf = font.render(text, True, shadow_color)
            self.screen.blit(text_surf, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score / Colony Size
        score_text = f"Colony Size: {self.score} / {self.WIN_SCORE}"
        draw_text(score_text, self.font_main, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW)
        
        # Steps
        steps_text = f"Cycle: {self.steps} / {self.MAX_STEPS}"
        draw_text(steps_text, self.font_small, self.COLOR_TEXT, (self.WIDTH - 150, 10), self.COLOR_TEXT_SHADOW)

        # Upgrade status
        if self.upgrades_unlocked.get('stealth_field'):
            color = self.COLOR_RESOURCE if self.active_effects['stealth_field_timer'] <= 0 else self.COLOR_GRID
            draw_text("Stealth Ready [SHIFT]", self.font_small, color, (10, 35), self.COLOR_TEXT_SHADOW)
        
        # Action feedback
        if self.last_action_feedback and self.last_action_feedback.get('timer', 0) > 0:
            fb = self.last_action_feedback
            if fb.get('type') == 'unlock':
                alpha = 255 if fb['timer'] > 30 else int(255 * (fb['timer']/30))
                unlock_font = pygame.font.Font(None, 40)
                text_surf = unlock_font.render(fb.get('text', ''), True, self.COLOR_RESOURCE)
                text_surf.set_alpha(alpha)
                pos = (self.WIDTH // 2 - text_surf.get_width() // 2, self.HEIGHT // 2 - text_surf.get_height() // 2)
                self.screen.blit(text_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "colony_size": len(self.colony_units),
        }

    # --- Utility and Drawing Helpers ---
    def _create_sentinel(self):
        return {
            'pos': (self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
            'target': (self.np_random.uniform(50, self.WIDTH-50), self.np_random.uniform(50, self.HEIGHT-50)),
            'base_speed': self.np_random.uniform(0.5, 1.0),
            'speed': 1.0,
            'scan_radius': self.np_random.uniform(80, 120),
            'scan_period': self.np_random.integers(150, 250),
            'scan_timer': self.np_random.integers(0, 100),
        }
        
    def _create_particles(self, pos, count, color, min_vel, max_vel, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_vel, max_vel)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(life // 2, life),
                'size': self.np_random.uniform(2, 5),
                'decay': 0.1,
                'color': color,
            })

    def _get_hex_points(self, center, radius):
        return [
            (center[0] + radius * math.cos(math.pi / 180 * (60 * i + 30)),
             center[1] + radius * math.sin(math.pi / 180 * (60 * i + 30)))
            for i in range(6)
        ]

    def _draw_hexagon(self, surface, color, center, radius, width=0, glow_color=None):
        points = self._get_hex_points(center, radius)
        int_points = [(int(p[0]), int(p[1])) for p in points]
        
        if glow_color:
            glow_points = self._get_hex_points(center, radius + 4)
            pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in glow_points], (*glow_color, 80))
            
        if width == 0:
            pygame.gfxdraw.filled_polygon(surface, int_points, color)
        else:
            pygame.gfxdraw.aapolygon(surface, int_points, color)

    def _draw_glowing_circle(self, surface, color, glow_color, center, radius):
        int_center = (int(center[0]), int(center[1]))
        pygame.gfxdraw.filled_circle(surface, int_center[0], int_center[1], int(radius + 5), (*glow_color, 80))
        pygame.gfxdraw.aacircle(surface, int_center[0], int_center[1], int(radius + 5), (*glow_color, 80))
        pygame.gfxdraw.filled_circle(surface, int_center[0], int_center[1], int(radius), color)
        pygame.gfxdraw.aacircle(surface, int_center[0], int_center[1], int(radius), color)

    def _draw_glowing_rect(self, surface, color, glow_color, center, size):
        rect = pygame.Rect(center[0] - size/2, center[1] - size/2, size, size)
        glow_rect = rect.inflate(8, 8)
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, (*glow_color, 80), (0, 0, *glow_rect.size), border_radius=2)
        surface.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(surface, color, rect, border_radius=2)

    def _draw_circle_alpha(self, surface, color_with_alpha, center, radius, width):
        # Pygame.gfxdraw doesn't handle alpha well for thick lines, so we use a surface
        target_rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, color_with_alpha, (radius, radius), radius, width)
        surface.blit(shape_surf, target_rect.topleft)

    def _hex_to_pixel(self, q, r):
        x = self.hex_radius * (3/2 * q) + self.WIDTH / 2
        y = self.hex_radius * (math.sqrt(3)/2 * q + math.sqrt(3) * r) + self.HEIGHT / 2
        return x, y

    def _get_hex_neighbors(self, q, r):
        # E, SE, SW, W, NW, NE
        return [
            (q + 1, r), (q, r + 1), (q - 1, r + 1),
            (q - 1, r), (q, r - 1), (q + 1, r - 1)
        ]

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the autograder.
    
    # To run this, you'll need to `pip install pygame`
    # and unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Quantum Colony")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    action = np.array([0, 0, 0]) # none, no space, no shift
    
    while not terminated and not truncated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        # Reset actions
        action.fill(0)
        
        # Movement bias
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Terraform
        if keys[pygame.K_SPACE]: action[1] = 1
        
        # Upgrade
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.metadata['render_fps'])

    env.close()