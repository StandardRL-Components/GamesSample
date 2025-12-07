import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:49:24.497780
# Source Brief: brief_03019.md
# Brief Index: 3019
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An isometric puzzle game where the agent must restore a Sumerian ziggurat.
    The agent controls a cursor to interact with levers, build structures,
    and unearth artifacts to gather resources for the restoration.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    game_description = (
        "Restore a Sumerian ziggurat in this isometric puzzle game. Interact with levers, build sites, "
        "and unearth artifacts to gather resources."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to interact with objects like "
        "levers, build sites, and the ziggurat."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 14
        self.GRID_HEIGHT = 12
        self.TILE_WIDTH_HALF = 26
        self.TILE_HEIGHT_HALF = 13
        self.ZIGGURAT_MAX_LEVELS = 4
        self.MAX_STEPS = 2000

        # --- Colors ---
        self.COLOR_BG = (25, 20, 35)
        self.COLOR_EARTH = (94, 75, 60)
        self.COLOR_STONE = (130, 120, 125)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_LEVER_INACTIVE = (100, 100, 100)
        self.COLOR_LEVER_ACTIVE = (255, 80, 80)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (240, 230, 220)
        self.COLOR_ZIGGURAT_BASE = (150, 120, 90)
        self.COLOR_ZIGGURAT_REPAIRED = (224, 195, 152)
        self.COLOR_BRIDGE = (139, 115, 85)
        self.COLOR_BUILD_SITE = (0, 0, 0, 60)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.font_ui = pygame.font.Font(None, 30)

        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.display.set_caption("Sumerian Restoration")
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = None
        self.last_space_held = None
        self.resources = None
        self.ziggurat_repaired_levels = None
        self.levers = None
        self.gears = None
        self.artifacts = None
        self.build_sites = None
        self.ziggurat_pos = (self.GRID_WIDTH // 2 - 1, 1)

    def _generate_puzzle(self):
        """Defines the puzzle layout and logic for a new game."""
        self.levers = [
            {'pos': (2, 5), 'state': False, 'id': 0, 'enabled': True},
            {'pos': (self.GRID_WIDTH - 3, 5), 'state': False, 'id': 1, 'enabled': False}
        ]
        self.gears = [
            {'pos': (2, 8), 'powered_by': [0], 'state': False, 'rotation': 0, 'id': 0},
            {'pos': (self.GRID_WIDTH - 3, 8), 'powered_by': [1], 'state': False, 'rotation': 0, 'id': 1}
        ]
        self.artifacts = [
            {'pos': (2, 10), 'unearthed': False, 'condition': {'type': 'gear', 'id': 0}, 'clay_reward': 10, 'is_key': False},
            {'pos': (self.GRID_WIDTH - 3, 10), 'unearthed': False, 'condition': {'type': 'gear', 'id': 1}, 'clay_reward': 2, 'is_key': True}
        ]
        self.build_sites = [
            {'pos': (self.GRID_WIDTH // 2 - 1, 5), 'built': False, 'cost': 8}
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        self.last_space_held = False
        self.resources = {'clay': 2}
        self.ziggurat_repaired_levels = 0

        self._generate_puzzle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01 # Small penalty for taking a step to encourage efficiency
        self.steps += 1

        self._handle_movement(movement)

        if space_held and not self.last_space_held:
            # Sfx: Click
            reward += self._handle_interaction()

        self.last_space_held = space_held

        reward += self._update_game_state()

        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not truncated: # Victory condition
            victory_bonus = 100.0
            reward += victory_bonus
            self.score += victory_bonus

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

    def _handle_interaction(self):
        # Lever interaction
        for lever in self.levers:
            if np.array_equal(self.cursor_pos, lever['pos']) and lever['enabled']:
                lever['state'] = not lever['state']
                # Sfx: Lever pull
                return 0.1

        # Build site interaction
        for site in self.build_sites:
            if np.array_equal(self.cursor_pos, site['pos']) and not site['built']:
                if self.resources['clay'] >= site['cost']:
                    self.resources['clay'] -= site['cost']
                    site['built'] = True
                    self.levers[1]['enabled'] = True # Enable the second lever
                    # Sfx: Building
                    return 2.0
                else:
                    # Sfx: Error/buzz
                    return -0.5

        # Ziggurat repair interaction
        if self.cursor_pos[0] in range(self.ziggurat_pos[0], self.ziggurat_pos[0] + 2) and self.cursor_pos[1] == self.ziggurat_pos[1]:
            key_artifact = next((art for art in self.artifacts if art['is_key']), None)
            if key_artifact and key_artifact['unearthed']:
                if self.ziggurat_repaired_levels < self.ZIGGURAT_MAX_LEVELS and self.resources['clay'] >= 2:
                    self.resources['clay'] -= 2
                    self.ziggurat_repaired_levels += 1
                    # Sfx: Repair success
                    return 5.0
                else:
                    return -0.5
        return 0.0

    def _update_game_state(self):
        reward = 0.0
        # Update gears
        for gear in self.gears:
            is_powered = all(self.levers[lever_id]['state'] for lever_id in gear['powered_by'])
            gear['state'] = is_powered

        # Update artifacts
        for artifact in self.artifacts:
            if not artifact['unearthed']:
                condition = artifact['condition']
                if condition['type'] == 'gear' and self.gears[condition['id']]['state']:
                    artifact['unearthed'] = True
                    self.resources['clay'] += artifact['clay_reward']
                    reward += 1.0
                    # Sfx: Artifact unearthed
        return reward

    def _check_termination(self):
        if self.ziggurat_repaired_levels >= self.ZIGGURAT_MAX_LEVELS:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "clay": self.resources['clay'],
            "ziggurat_progress": f"{self.ziggurat_repaired_levels}/{self.ZIGGURAT_MAX_LEVELS}"
        }

    def _cart_to_iso(self, x, y):
        iso_x = self.SCREEN_WIDTH / 2 + (x - y) * self.TILE_WIDTH_HALF
        iso_y = 100 + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_iso_rect(self, x, y, color, height_offset=0):
        iso_x, iso_y = self._cart_to_iso(x, y)
        iso_y -= height_offset
        points = [
            (iso_x, iso_y),
            (iso_x + self.TILE_WIDTH_HALF, iso_y + self.TILE_HEIGHT_HALF),
            (iso_x, iso_y + self.TILE_HEIGHT_HALF * 2),
            (iso_x - self.TILE_WIDTH_HALF, iso_y + self.TILE_HEIGHT_HALF),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, tuple(min(255, c * 1.2) for c in color if c < 255))
    
    def _render_game(self):
        # Draw ground tiles
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                self._draw_iso_rect(x, y, self.COLOR_EARTH)

        # Build render queue and sort by y-pos for correct layering
        render_queue = self.levers + self.gears + self.artifacts + self.build_sites
        render_queue.sort(key=lambda item: (item['pos'][1], item['pos'][0]))
        
        self._render_ziggurat()

        for item in render_queue:
            if 'state' in item and 'powered_by' in item: self._render_gear(item)
            elif 'state' in item: self._render_lever(item)
            elif 'unearthed' in item: self._render_artifact(item)
            elif 'built' in item:
                if item['built']: self._render_bridge(item)
                else: self._render_build_site(item)
        
        self._render_cursor()

    def _render_ziggurat(self):
        size = 2
        for level in range(self.ZIGGURAT_MAX_LEVELS, 0, -1):
            color = self.COLOR_ZIGGURAT_REPAIRED if self.ziggurat_repaired_levels >= level else self.COLOR_ZIGGURAT_BASE
            h_offset = (level -1) * self.TILE_HEIGHT_HALF * 2
            for i in range(size):
                for j in range(size):
                    self._draw_iso_rect(self.ziggurat_pos[0] + i, self.ziggurat_pos[1] + j, color, h_offset)

    def _render_lever(self, lever):
        iso_x, iso_y = self._cart_to_iso(*lever['pos'])
        base_rect = pygame.Rect(0, 0, 20, 10)
        base_rect.center = (iso_x, iso_y + self.TILE_HEIGHT_HALF)
        pygame.draw.rect(self.screen, self.COLOR_STONE, base_rect, border_radius=3)
        
        color = self.COLOR_LEVER_ACTIVE if lever['state'] else self.COLOR_LEVER_INACTIVE
        if not lever['enabled']: color = tuple(c // 2 for c in color)
        
        handle_x = iso_x + (6 if lever['state'] else -6)
        pygame.draw.line(self.screen, color, (iso_x, base_rect.top), (handle_x, base_rect.top - 20), 4)
        pygame.gfxdraw.filled_circle(self.screen, handle_x, base_rect.top - 20, 5, color)
        pygame.gfxdraw.aacircle(self.screen, handle_x, base_rect.top - 20, 5, (255,255,255))

    def _render_gear(self, gear):
        iso_x, iso_y = self._cart_to_iso(*gear['pos'])
        radius = 18
        color = self.COLOR_LEVER_ACTIVE if gear['state'] else self.COLOR_LEVER_INACTIVE
        
        if gear['state']:
            # Sfx: Gear turning
            gear['rotation'] = (gear['rotation'] + 4) % 360
        
        pygame.gfxdraw.filled_circle(self.screen, iso_x, iso_y + 10, radius, self.COLOR_STONE)
        for i in range(8):
            angle = math.radians(gear['rotation'] + i * 45)
            p1 = (iso_x + math.cos(angle) * (radius-2), iso_y + 10 + math.sin(angle) * (radius-2))
            p2 = (iso_x + math.cos(angle) * (radius+2), iso_y + 10 + math.sin(angle) * (radius+2))
            pygame.draw.line(self.screen, color, p1, p2, 5)
        pygame.gfxdraw.filled_circle(self.screen, iso_x, iso_y + 10, radius - 5, color)

    def _render_artifact(self, artifact):
        if not artifact['unearthed']: return
        iso_x, iso_y = self._cart_to_iso(*artifact['pos'])
        
        glow_radius = int(18 + 6 * math.sin(self.steps * 0.1))
        glow_color = (*self.COLOR_GOLD, 50)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (iso_x - glow_radius, iso_y - glow_radius + 10))
        
        color = self.COLOR_GOLD if artifact['is_key'] else (200, 200, 220)
        pygame.gfxdraw.filled_circle(self.screen, iso_x, iso_y + 10, 8, color)
        pygame.gfxdraw.aacircle(self.screen, iso_x, iso_y + 10, 8, (255, 255, 255))

    def _render_build_site(self, site):
        iso_x, iso_y = self._cart_to_iso(*site['pos'])
        s = pygame.Surface((self.TILE_WIDTH_HALF * 2, self.TILE_HEIGHT_HALF * 2), pygame.SRCALPHA)
        s.fill(self.COLOR_BUILD_SITE)
        self.screen.blit(s, (iso_x - self.TILE_WIDTH_HALF, iso_y))

    def _render_bridge(self, site):
        self._draw_iso_rect(site['pos'][0], site['pos'][1], self.COLOR_BRIDGE, height_offset=2)

    def _render_cursor(self):
        iso_x, iso_y = self._cart_to_iso(*self.cursor_pos)
        pulse = 2 * math.sin(self.steps * 0.2)
        w, h = self.TILE_WIDTH_HALF + pulse, self.TILE_HEIGHT_HALF + pulse/2
        points = [
            (iso_x, iso_y - h + self.TILE_HEIGHT_HALF),
            (iso_x + w, iso_y + self.TILE_HEIGHT_HALF),
            (iso_x, iso_y + h + self.TILE_HEIGHT_HALF),
            (iso_x - w, iso_y + self.TILE_HEIGHT_HALF),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CURSOR)

    def _render_ui(self):
        clay_text = self.font_ui.render(f"Clay: {self.resources['clay']}", True, self.COLOR_UI_TEXT)
        self.screen.blit(clay_text, (20, 20))
        
        prog_text = self.font_ui.render(f"Ziggurat: {self.ziggurat_repaired_levels}/{self.ZIGGURAT_MAX_LEVELS}", True, self.COLOR_UI_TEXT)
        text_rect = prog_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(prog_text, text_rect)

        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(bottomleft=(20, self.SCREEN_HEIGHT - 10))
        self.screen.blit(score_text, score_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Example of how to run the environment for human play
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # Mapping keyboard keys to actions for human control
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not done:
        movement = 0
        space_held = 0
        shift_held = 0 # Unused in this example

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # only one movement at a time
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")
            done = True

    env.close()