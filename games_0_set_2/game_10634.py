import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:47:19.446956
# Source Brief: brief_00634.md
# Brief Index: 634
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A stealth puzzle game where the player manipulates light beams using mirrors
    and prisms to reach a goal while avoiding detection by patrolling guards.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "A stealth puzzle game where you use mirrors and prisms to guide a light beam to a goal, "
        "while avoiding detection by patrolling guards."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to place or rotate an item, "
        "and press shift to cycle through available items."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.render_mode = render_mode
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game Constants ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_GOAL = (0, 255, 128)
        self.COLOR_GOAL_GLOW = (0, 255, 128, 50)
        self.COLOR_GUARD = (255, 50, 50)
        self.COLOR_GUARD_GLOW = (255, 50, 50, 70)
        self.COLOR_MIRROR = (50, 150, 255)
        self.COLOR_PRISM = (255, 220, 50)
        self.COLOR_BEAM = (0, 255, 128)
        self.COLOR_BEAM_GLOW = (0, 255, 128, 60)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_HIGHLIGHT = (255, 220, 50)

        self.MAX_STEPS = 1000
        self.BEAM_MAX_LENGTH = 50 # Max reflection/refraction steps

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.level = 0
        self.game_over = False
        
        self.grid_w, self.grid_h = 0, 0
        self.tile_size = 0
        self.grid_offset_x, self.grid_offset_y = 0, 0

        self.player_pos = [0, 0] # Actually the cursor
        self.light_source_pos = [0, 0]
        self.goal_pos = [0, 0]
        
        self.guards = []
        self.placed_items = {} # (x, y) -> {'type': 'mirror'/'prism', 'rotation': 0}
        self.beam_segments = []
        
        self.inventory = ['mirror', 'prism']
        self.selected_inventory_idx = 0
        
        self.last_space_press = False
        self.last_shift_press = False

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.last_space_press = False
        self.last_shift_press = False

        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        """Initializes game state for the current level."""
        self.grid_w = 20 + min(self.level, 10)
        self.grid_h = 12 + min(self.level, 8)
        self.tile_size = min((self.width - 40) // self.grid_w, (self.height - 80) // self.grid_h)
        self.grid_offset_x = (self.width - self.grid_w * self.tile_size) // 2
        self.grid_offset_y = (self.height - self.grid_h * self.tile_size) // 2 + 40

        self.light_source_pos = [1, self.grid_h // 2]
        self.player_pos = [2, self.grid_h // 2]
        self.goal_pos = [self.grid_w - 2, self.grid_h // 2]

        self.placed_items.clear()
        self.guards.clear()
        
        num_guards = 1 + (self.level - 1) // 2
        for i in range(num_guards):
            path_type = self.np_random.choice(['rect', 'line_v', 'line_h'])
            path = []
            start_x = self.np_random.integers(3, self.grid_w - 3)
            start_y = self.np_random.integers(1, self.grid_h - 1)
            
            if path_type == 'rect':
                w, h = self.np_random.integers(3, 6), self.np_random.integers(3, 6)
                path.extend([[x, start_y] for x in range(start_x, min(self.grid_w - 1, start_x + w))])
                path.extend([[min(self.grid_w - 1, start_x + w - 1), y] for y in range(start_y, min(self.grid_h - 1, start_y + h))])
                path.extend([[x, min(self.grid_h - 1, start_y + h - 1)] for x in range(min(self.grid_w - 1, start_x + w - 1), start_x - 1, -1)])
                path.extend([[start_x, y] for y in range(min(self.grid_h - 1, start_y + h - 1), start_y - 1, -1)])
            elif path_type == 'line_v':
                length = self.np_random.integers(4, self.grid_h - 2)
                path.extend([[start_x, y] for y in range(start_y, min(self.grid_h - 1, start_y + length))])
                path.extend([[start_x, y] for y in range(min(self.grid_h - 1, start_y + length - 1), start_y - 1, -1)])
            else: # line_h
                length = self.np_random.integers(4, self.grid_w - 4)
                path.extend([[x, start_y] for x in range(start_x, min(self.grid_w - 1, start_x + length))])
                path.extend([[x, start_y] for x in range(min(self.grid_w - 1, start_x + length - 1), start_x - 1, -1)])
            
            if path:
                self.guards.append({'pos': path[0], 'path': path, 'path_idx': 0})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Use presses (rising edge) to avoid repeated actions
        space_press = space_held and not self.last_space_press
        shift_press = shift_held and not self.last_shift_press
        self.last_space_press = space_held
        self.last_shift_press = shift_held

        reward = -0.1 # Small penalty for each step to encourage efficiency
        self.steps += 1
        
        # 1. Handle player action
        action_taken = self._handle_player_action(movement, space_press, shift_press)
        if action_taken['placed_item']:
            reward += 1.0 # Small reward for placing an item
        
        # 2. Update guards
        self._update_guards()

        # 3. Calculate beam path and check for goal/detection
        goal_reached, detected = self._calculate_beam_paths()
        
        # 4. Check termination conditions
        terminated = False
        truncated = False
        if detected:
            reward = -100.0
            self.score += reward
            terminated = True
            self.game_over = True
        elif goal_reached:
            reward = 100.0 + max(0, 100 - self.steps) # Bonus for speed
            self.score += reward
            self.level += 1
            # Episode terminates on win, but game continues to next level in the next reset
            terminated = True 
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_action(self, movement, space_press, shift_press):
        action_info = {'placed_item': False}
        # --- Movement ---
        if movement == 1: self.player_pos[1] -= 1 # Up
        elif movement == 2: self.player_pos[1] += 1 # Down
        elif movement == 3: self.player_pos[0] -= 1 # Left
        elif movement == 4: self.player_pos[0] += 1 # Right
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.grid_w - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.grid_h - 1)

        # --- Shift: Cycle selected item ---
        if shift_press:
            self.selected_inventory_idx = (self.selected_inventory_idx + 1) % len(self.inventory)
        
        # --- Space: Place/Rotate item ---
        if space_press:
            pos_tuple = tuple(self.player_pos)
            if pos_tuple in self.placed_items:
                self.placed_items[pos_tuple]['rotation'] = (self.placed_items[pos_tuple]['rotation'] + 45) % 360
            elif pos_tuple != tuple(self.light_source_pos) and pos_tuple != tuple(self.goal_pos):
                item_type = self.inventory[self.selected_inventory_idx]
                self.placed_items[pos_tuple] = {'type': item_type, 'rotation': 45 if item_type == 'mirror' else 0}
                action_info['placed_item'] = True
        
        return action_info

    def _update_guards(self):
        for guard in self.guards:
            guard['path_idx'] = (guard['path_idx'] + 1) % len(guard['path'])
            guard['pos'] = guard['path'][guard['path_idx']]

    def _calculate_beam_paths(self):
        segments = []
        q = [(self.light_source_pos, (1, 0), 0)] # pos, dir, depth
        visited = set()
        detected = False
        goal_reached = False

        while q:
            pos, direction, depth = q.pop(0)

            if depth > self.BEAM_MAX_LENGTH: continue
            
            key = (tuple(pos), direction)
            if key in visited: continue
            visited.add(key)
            
            current_pos = list(pos)
            for _ in range(self.BEAM_MAX_LENGTH):
                next_pos = [current_pos[0] + direction[0], current_pos[1] + direction[1]]
                
                # Check boundaries
                if not (0 <= next_pos[0] < self.grid_w and 0 <= next_pos[1] < self.grid_h):
                    segments.append((current_pos, next_pos))
                    break
                
                # Check for goal
                if tuple(next_pos) == tuple(self.goal_pos):
                    segments.append((current_pos, next_pos))
                    goal_reached = True
                    break

                # Check for detection
                for guard in self.guards:
                    if tuple(next_pos) == tuple(guard['pos']):
                        segments.append((current_pos, next_pos))
                        detected = True
                        break
                if detected or goal_reached: break
                
                # Check for items
                item = self.placed_items.get(tuple(next_pos))
                if item:
                    segments.append((current_pos, next_pos))
                    new_dirs = self._get_new_directions(direction, item)
                    for new_dir in new_dirs:
                        q.append((next_pos, new_dir, depth + 1))
                    break
                
                current_pos = next_pos
            else: # No break
                segments.append((pos, current_pos))
            
            if detected or goal_reached:
                break

        self.beam_segments = segments
        return goal_reached, detected

    def _get_new_directions(self, direction, item):
        rot = item['rotation']
        if item['type'] == 'mirror':
            # Mirror reflects at 90 degrees. A 45-degree mirror reflects horizontal to vertical.
            # Mirror at 45 or 225 deg ('/'):
            if rot in [45, 225]:
                return [(-direction[1], -direction[0])]
            # Mirror at 135 or 315 deg ('\'):
            elif rot in [135, 315]:
                return [(direction[1], direction[0])]
            return [] # In-line with mirror
        
        elif item['type'] == 'prism':
            # Prism splits beam.
            dx, dy = direction
            if rot % 180 == 0: # Horizontal split for vertical beam
                return [(dy, dx), (-dy, -dx)] if dy != 0 else []
            elif rot % 180 == 90: # Vertical split for horizontal beam
                return [(dy, dx), (-dy, -dx)] if dx != 0 else []
            return [(dy, dx), (-dy, -dx)] # Default split
        
        return []

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_goal()
        self._render_items()
        self._render_beam()
        self._render_guards()
        self._render_player_cursor()
        self._render_light_source()
        
    def _render_ui(self):
        # --- Top Bar ---
        bar_rect = pygame.Rect(0, 0, self.width, 40)
        pygame.draw.rect(self.screen, (30, 30, 40), bar_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 40), (self.width, 40), 1)

        level_text = self.font_large.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (20, 8))

        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.width - score_text.get_width() - 20, 8))

        # --- Bottom Bar (Inventory) ---
        selected_item = self.inventory[self.selected_inventory_idx]
        inv_text = self.font_small.render(f"SELECTED: {selected_item.upper()}", True, self.COLOR_UI_HIGHLIGHT)
        self.screen.blit(inv_text, (self.grid_offset_x, self.height - 25))
        
        help_text = self.font_small.render("ARROWS: Move | SPACE: Place/Rotate | SHIFT: Cycle Item", True, self.COLOR_UI_TEXT)
        self.screen.blit(help_text, (self.width - help_text.get_width() - self.grid_offset_x, self.height - 25))

    def _grid_to_pixel(self, grid_pos):
        x = self.grid_offset_x + grid_pos[0] * self.tile_size + self.tile_size // 2
        y = self.grid_offset_y + grid_pos[1] * self.tile_size + self.tile_size // 2
        return int(x), int(y)

    def _render_glow_circle(self, pos, color, radius):
        surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surface, color, (radius, radius), radius)
        self.screen.blit(surface, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_grid(self):
        for x in range(self.grid_w + 1):
            start = (self.grid_offset_x + x * self.tile_size, self.grid_offset_y)
            end = (self.grid_offset_x + x * self.tile_size, self.grid_offset_y + self.grid_h * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for y in range(self.grid_h + 1):
            start = (self.grid_offset_x, self.grid_offset_y + y * self.tile_size)
            end = (self.grid_offset_x + self.grid_w * self.tile_size, self.grid_offset_y + y * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _render_goal(self):
        px, py = self._grid_to_pixel(self.goal_pos)
        r = self.tile_size // 2
        self._render_glow_circle((px, py), self.COLOR_GOAL_GLOW, r * 2)
        pygame.gfxdraw.box(self.screen, pygame.Rect(px - r, py - r, self.tile_size, self.tile_size), self.COLOR_GOAL)

    def _render_player_cursor(self):
        px, py = self._grid_to_pixel(self.player_pos)
        r = self.tile_size // 3
        pygame.gfxdraw.rectangle(self.screen, pygame.Rect(px - r, py - r, r*2, r*2), (*self.COLOR_PLAYER, 150))

    def _render_light_source(self):
        px, py = self._grid_to_pixel(self.light_source_pos)
        r = int(self.tile_size * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, px, py, r, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, r, self.COLOR_PLAYER)

    def _render_guards(self):
        for guard in self.guards:
            px, py = self._grid_to_pixel(guard['pos'])
            r = int(self.tile_size * 0.35)
            self._render_glow_circle((px, py), self.COLOR_GUARD_GLOW, int(r * 1.5))
            pygame.gfxdraw.filled_circle(self.screen, px, py, r, self.COLOR_GUARD)
            pygame.gfxdraw.aacircle(self.screen, px, py, r, self.COLOR_GUARD)

    def _render_items(self):
        for pos, item in self.placed_items.items():
            px, py = self._grid_to_pixel(pos)
            color = self.COLOR_MIRROR if item['type'] == 'mirror' else self.COLOR_PRISM
            
            if item['type'] == 'mirror':
                r = int(self.tile_size * 0.45)
                angle = math.radians(item['rotation'])
                x1 = px - r * math.cos(angle)
                y1 = py - r * math.sin(angle)
                x2 = px + r * math.cos(angle)
                y2 = py + r * math.sin(angle)
                pygame.draw.aaline(self.screen, color, (x1, y1), (x2, y2), 2)
            
            elif item['type'] == 'prism':
                r = int(self.tile_size * 0.4)
                points = []
                for i in range(3):
                    angle = math.radians(item['rotation'] + 120 * i)
                    points.append((px + r * math.cos(angle), py + r * math.sin(angle)))
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_beam(self):
        for start_pos, end_pos in self.beam_segments:
            p_start = self._grid_to_pixel(start_pos)
            p_end = self._grid_to_pixel(end_pos)
            
            # Glow effect
            dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
            if dx == 0 and dy == 0: continue
            
            perp_dx, perp_dy = -dy, dx
            norm = math.hypot(perp_dx, perp_dy)
            if norm > 0:
                perp_dx /= norm
                perp_dy /= norm
            
            w = 8
            glow_poly = [
                (p_start[0] - perp_dx*w, p_start[1] - perp_dy*w),
                (p_start[0] + perp_dx*w, p_start[1] + perp_dy*w),
                (p_end[0] + perp_dx*w, p_end[1] + perp_dy*w),
                (p_end[0] - perp_dx*w, p_end[1] - perp_dy*w),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, glow_poly, self.COLOR_BEAM_GLOW)
            
            # Core beam
            pygame.draw.aaline(self.screen, self.COLOR_BEAM, p_start, p_end, 2)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_pos": self.player_pos,
            "guards": len(self.guards),
            "items_placed": len(self.placed_items)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    
    # Use a real display for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Light & Shadow")
    
    obs, info = env.reset()
    done = False
    clock = pygame.time.Clock()
    
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = 0
    shift_held = 0

    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not done:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
        
        # --- Action Polling ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != -0.1:
            print(f"Step: {info['steps']}, Level: {info['level']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        if terminated or truncated:
            print("--- Episode Finished ---")
            print(f"Final Score: {info['score']:.2f}, Steps: {info['steps']}, Level: {info['level']}")
            if env.game_over:
                print("GAME OVER. Resetting...")
            obs, info = env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control the speed of manual play

    env.close()