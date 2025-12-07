import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:51:17.397400
# Source Brief: brief_02317.md
# Brief Index: 2317
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
        "A grid-based puzzle game where you use tools to grow or shrink cells to match their target sizes. "
        "Solve the puzzle to activate the portal and advance to the next level."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'space' to use the selected tool or "
        "enter the portal. Press 'shift' to switch between the grow and shrink tools."
    )
    auto_advance = False

    # --- Level Definitions ---
    # Each level defines grid size, cell properties, and portal position.
    # 'radius' is the starting radius in pixels.
    # 'target_radius' is the radius required to solve the puzzle for that cell.
    LEVEL_DEFINITIONS = [
        # Level 1: Introduction to shrinking
        {'grid_size': (10, 8), 'cells': [{'pos': (5, 4), 'radius': 20, 'target_radius': 0}], 'portal_pos': (2, 2)},
        # Level 2: Two cells to shrink
        {'grid_size': (10, 8), 'cells': [{'pos': (3, 2), 'radius': 15, 'target_radius': 0}, {'pos': (7, 6), 'radius': 20, 'target_radius': 0}], 'portal_pos': (5, 4)},
        # Level 3: Introduction to growing
        {'grid_size': (12, 9), 'cells': [{'pos': (2, 2), 'radius': 10, 'target_radius': 25}, {'pos': (9, 6), 'radius': 20, 'target_radius': 0}], 'portal_pos': (6, 4)},
        # Level 4: A mix of tasks
        {'grid_size': (12, 9), 'cells': [{'pos': (2, 6), 'radius': 0, 'target_radius': 20}, {'pos': (6, 4), 'radius': 25, 'target_radius': 0}, {'pos': (10, 2), 'radius': 15, 'target_radius': 15}], 'portal_pos': (1, 1)},
        # Level 5: Larger grid, more cells
        {'grid_size': (14, 10), 'cells': [{'pos': (3, 3), 'radius': 20, 'target_radius': 0}, {'pos': (10, 7), 'radius': 20, 'target_radius': 0}, {'pos': (7, 5), 'radius': 0, 'target_radius': 20}], 'portal_pos': (1, 8)},
        # Level 6: A symmetric pattern
        {'grid_size': (14, 10), 'cells': [{'pos': (3, 2), 'radius': 0, 'target_radius': 15}, {'pos': (10, 2), 'radius': 0, 'target_radius': 15}, {'pos': (3, 7), 'radius': 15, 'target_radius': 0}, {'pos': (10, 7), 'radius': 15, 'target_radius': 0}], 'portal_pos': (7, 0)},
        # Level 7: Chained puzzle
        {'grid_size': (16, 12), 'cells': [{'pos': (4, 9), 'radius': 25, 'target_radius': 0}, {'pos': (8, 6), 'radius': 25, 'target_radius': 0}, {'pos': (12, 3), 'radius': 25, 'target_radius': 0}], 'portal_pos': (1, 1)},
        # Level 8: Don't touch
        {'grid_size': (16, 12), 'cells': [{'pos': (4, 3), 'radius': 20, 'target_radius': 20}, {'pos': (12, 9), 'radius': 20, 'target_radius': 20}, {'pos': (8, 6), 'radius': 25, 'target_radius': 0}], 'portal_pos': (8, 1)},
        # Level 9: Grow to max
        {'grid_size': (18, 13), 'cells': [{'pos': (4, 4), 'radius': 0, 'target_radius': 22}, {'pos': (13, 8), 'radius': 0, 'target_radius': 22}, {'pos': (8, 2), 'radius': 20, 'target_radius': 0}], 'portal_pos': (1, 11)},
        # Level 10: Final challenge
        {'grid_size': (20, 15), 'cells': [{'pos': (3, 11), 'radius': 20, 'target_radius': 0}, {'pos': (16, 3), 'radius': 20, 'target_radius': 0}, {'pos': (10, 7), 'radius': 0, 'target_radius': 25}], 'portal_pos': (10, 1)},
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.NUM_LEVELS = len(self.LEVEL_DEFINITIONS)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 28, bold=True)

        # --- Colors ---
        self.COLOR_BG = (15, 18, 26)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_CELL = (0, 120, 255)
        self.COLOR_CELL_TARGET_MET = (0, 200, 150)
        self.COLOR_PORTAL = (255, 200, 0)
        self.COLOR_PORTAL_ACTIVE = (255, 255, 150)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TOOL_GROW = (40, 220, 110)
        self.COLOR_TOOL_SHRINK = (230, 50, 80)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_UI_BG = (25, 30, 45)

        # --- Game State ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 0
        self.cursor_grid_pos = [0, 0]
        self.cursor_pixel_pos = [0.0, 0.0]
        self.inventory = []
        self.selected_tool_idx = 0
        self.previous_space_held = False
        self.previous_shift_held = False
        self.particles = []
        self.cells = []
        self.portal = {}
        self.level_data = {}
        self.grid_offset = (0, 0)
        self.cell_pixel_size = 0
        
        # A random number generator for reproducibility
        self.np_random = None
        
    def _setup_level(self, level_num):
        self.level_data = self.LEVEL_DEFINITIONS[level_num - 1]
        grid_w, grid_h = self.level_data['grid_size']

        # Calculate grid rendering properties
        padding = 40
        ui_bar_height = 60
        drawable_w = self.SCREEN_WIDTH - 2 * padding
        drawable_h = self.SCREEN_HEIGHT - 2 * padding - ui_bar_height
        
        cell_w = drawable_w / grid_w
        cell_h = drawable_h / grid_h
        self.cell_pixel_size = min(cell_w, cell_h)

        grid_pixel_w = self.cell_pixel_size * grid_w
        grid_pixel_h = self.cell_pixel_size * grid_h
        self.grid_offset = (
            (self.SCREEN_WIDTH - grid_pixel_w) / 2,
            (self.SCREEN_HEIGHT - ui_bar_height - grid_pixel_h) / 2
        )

        self.cells = [cell.copy() for cell in self.level_data['cells']]
        self.portal = {'pos': self.level_data['portal_pos'], 'active': False}
        
        self.cursor_grid_pos = [grid_w // 2, grid_h // 2]
        self.cursor_pixel_pos = list(self._grid_to_pixel(self.cursor_grid_pos))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 1
        self.inventory = ['grow', 'shrink']
        self.selected_tool_idx = 0
        self.previous_space_held = False
        self.previous_shift_held = False
        self.particles.clear()
        
        self._setup_level(self.current_level)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for each step to encourage efficiency

        # --- Handle Input ---
        if shift_held and not self.previous_shift_held:
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.inventory)
            # SFX: UI_SWITCH

        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self.cursor_grid_pos[0] += dx
            self.cursor_grid_pos[1] += dy
            grid_w, grid_h = self.level_data['grid_size']
            self.cursor_grid_pos[0] = max(0, min(grid_w - 1, self.cursor_grid_pos[0]))
            self.cursor_grid_pos[1] = max(0, min(grid_h - 1, self.cursor_grid_pos[1]))

        if space_held and not self.previous_space_held:
            action_taken = False
            # 1. Check for portal entry
            if self.portal['active'] and list(self.cursor_grid_pos) == list(self.portal['pos']):
                reward += self._advance_level()
                action_taken = True
                # SFX: LEVEL_COMPLETE
            
            # 2. Check for tool use on cell
            if not action_taken:
                for cell in self.cells:
                    if list(self.cursor_grid_pos) == list(cell['pos']):
                        tool = self.inventory[self.selected_tool_idx]
                        old_radius = cell['radius']
                        old_dist_to_target = abs(old_radius - cell['target_radius'])
                        
                        if tool == 'grow':
                            cell['radius'] = min(cell['radius'] + 5, self.cell_pixel_size / 2 - 2)
                            self._create_particles(self._grid_to_pixel(cell['pos']), self.COLOR_TOOL_GROW, 20)
                            # SFX: GROW
                        elif tool == 'shrink':
                            cell['radius'] = max(0, cell['radius'] - 5)
                            self._create_particles(self._grid_to_pixel(cell['pos']), self.COLOR_TOOL_SHRINK, 20)
                            # SFX: SHRINK

                        new_dist_to_target = abs(cell['radius'] - cell['target_radius'])
                        if new_dist_to_target < old_dist_to_target:
                            reward += 1.0  # Reward for making progress
                        break

        # --- Update Game State ---
        self.steps += 1

        # Check for portal activation
        if not self.portal['active']:
            all_targets_met = all(c['radius'] == c['target_radius'] for c in self.cells)
            if all_targets_met:
                self.portal['active'] = True
                reward += 10.0
                self._create_particles(self._grid_to_pixel(self.portal['pos']), self.COLOR_PORTAL, 50, life=60)
                # SFX: PORTAL_ACTIVATE
        
        self.score += reward
        self.previous_space_held = space_held
        self.previous_shift_held = shift_held
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        
        if truncated and not self.game_over:
            self.score -= 100 # Timeout penalty
            reward -= 100

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _advance_level(self):
        level_reward = 50.0
        self.score += level_reward
        self.current_level += 1

        if self.current_level > self.NUM_LEVELS:
            self.game_over = True
            win_bonus = 100.0
            self.score += win_bonus
            return level_reward + win_bonus
        
        self._setup_level(self.current_level)
        self.particles.clear()
        return level_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        grid_w, grid_h = self.level_data['grid_size']
        for i in range(grid_w + 1):
            x = self.grid_offset[0] + i * self.cell_pixel_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_offset[1]), (x, self.grid_offset[1] + grid_h * self.cell_pixel_size))
        for i in range(grid_h + 1):
            y = self.grid_offset[1] + i * self.cell_pixel_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset[0], y), (self.grid_offset[0] + grid_w * self.cell_pixel_size, y))

        # Draw portal
        portal_px, portal_py = self._grid_to_pixel(self.portal['pos'])
        if self.portal['active']:
            # Animated glow
            t = self.steps * 0.1
            for i in range(5):
                angle = t + i * (2 * math.pi / 5)
                px = int(portal_px + math.cos(angle) * 10)
                py = int(portal_py + math.sin(angle) * 10)
                pygame.gfxdraw.filled_circle(self.screen, px, py, 3, self.COLOR_PORTAL_ACTIVE)
            pygame.gfxdraw.aacircle(self.screen, int(portal_px), int(portal_py), 12, self.COLOR_PORTAL_ACTIVE)
        else:
            pygame.gfxdraw.aacircle(self.screen, int(portal_px), int(portal_py), 8, self.COLOR_PORTAL)
            pygame.gfxdraw.filled_circle(self.screen, int(portal_px), int(portal_py), 6, self.COLOR_PORTAL)

        # Draw cells
        for cell in self.cells:
            px, py = self._grid_to_pixel(cell['pos'])
            radius = int(cell['radius'])
            if radius > 0:
                color = self.COLOR_CELL_TARGET_MET if radius == cell['target_radius'] else self.COLOR_CELL
                pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, color)
        
        self._update_and_draw_particles()

        # Draw cursor with smooth interpolation
        target_pixel_pos = self._grid_to_pixel(self.cursor_grid_pos)
        lerp_factor = 0.4
        self.cursor_pixel_pos[0] += (target_pixel_pos[0] - self.cursor_pixel_pos[0]) * lerp_factor
        self.cursor_pixel_pos[1] += (target_pixel_pos[1] - self.cursor_pixel_pos[1]) * lerp_factor
        
        cursor_rect = pygame.Rect(0, 0, self.cell_pixel_size, self.cell_pixel_size)
        cursor_rect.center = self.cursor_pixel_pos
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=3)

    def _render_ui(self):
        ui_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 60, self.SCREEN_WIDTH, 60)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, self.SCREEN_HEIGHT - 60), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 60))

        # Level and Score
        level_text = self.font_main.render(f"Level: {self.current_level}/{self.NUM_LEVELS}", True, self.COLOR_UI_TEXT)
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        steps_text = self.font_main.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (15, 10))
        self.screen.blit(score_text, (150, 10))
        self.screen.blit(steps_text, (320, 10))
        
        # Tool Inventory
        tool_text = self.font_main.render("Tool:", True, self.COLOR_UI_TEXT)
        self.screen.blit(tool_text, (20, self.SCREEN_HEIGHT - 42))

        for i, tool in enumerate(self.inventory):
            is_selected = i == self.selected_tool_idx
            base_x = 90 + i * 100
            base_y = self.SCREEN_HEIGHT - 30
            
            box_color = self.COLOR_CURSOR if is_selected else self.COLOR_GRID
            pygame.draw.rect(self.screen, box_color, (base_x - 25, base_y - 20, 80, 40), 2 if is_selected else 1, border_radius=5)
            
            if tool == 'grow':
                color = self.COLOR_TOOL_GROW
                pygame.draw.line(self.screen, color, (base_x, base_y - 10), (base_x, base_y + 10), 3)
                pygame.draw.line(self.screen, color, (base_x - 10, base_y), (base_x + 10, base_y), 3)
                text = self.font_main.render("Grow", True, color if is_selected else self.COLOR_UI_TEXT)
                self.screen.blit(text, (base_x + 20, base_y - 10))
            elif tool == 'shrink':
                color = self.COLOR_TOOL_SHRINK
                pygame.draw.line(self.screen, color, (base_x - 10, base_y), (base_x + 10, base_y), 3)
                text = self.font_main.render("Shrink", True, color if is_selected else self.COLOR_UI_TEXT)
                self.screen.blit(text, (base_x + 20, base_y - 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0,0))
            msg = "YOU WIN!" if self.current_level > self.NUM_LEVELS else "GAME OVER"
            win_text = self.font_title.render(msg, True, self.COLOR_UI_TEXT)
            win_rect = win_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(win_text, win_rect)

    def _grid_to_pixel(self, grid_pos):
        x = self.grid_offset[0] + (grid_pos[0] + 0.5) * self.cell_pixel_size
        y = self.grid_offset[1] + (grid_pos[1] + 0.5) * self.cell_pixel_size
        return x, y
        
    def _create_particles(self, pos, color, count, life=30, speed=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5), 
                   math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5)]
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'color': color,
                'life': life, 'max_life': life
            })

    def _update_and_draw_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.remove(p)
                continue

            radius = int((p['life'] / p['max_life']) * 4)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
            "game_over": self.game_over,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is not part of the required environment but is useful for development
    
    # Un-comment the following line to run with display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Cell Puzzle Environment")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    
    # To track key presses for single-press actions
    last_shift_pressed = False
    last_space_pressed = False

    while not terminated and not truncated:
        movement = 0 # no-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # Capture single press for space and shift
        current_space_pressed = keys[pygame.K_SPACE]
        if current_space_pressed and not last_space_pressed:
            space_action = 1
        last_space_pressed = current_space_pressed

        current_shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        if current_shift_pressed and not last_shift_pressed:
            shift_action = 1
        last_shift_pressed = current_shift_pressed

        # Note: The environment expects held state, but for manual play,
        # single press is more intuitive for actions like 'switch tool'.
        # We will send the held state as the environment expects.
        action = [movement, 1 if keys[pygame.K_SPACE] else 0, 1 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0 and reward != -0.01:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Level: {info['level']}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Run at a slower rate for manual play

    env.close()