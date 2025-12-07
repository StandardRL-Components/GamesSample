import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:06:50.503357
# Source Brief: brief_02587.md
# Brief Index: 2587
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: A cyberpunk circuit-building puzzle game.

    The player controls a cursor on a grid to place circuit components from a limited
    inventory. The goal is to connect a power source to all target nodes on the grid.
    Successful connections create visually satisfying chain reactions and deactivate targets.
    The game rewards efficient solutions and penalizes wasted components.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Cursor Movement (0:None, 1:Up, 2:Down, 3:Left, 4:Right)
    - action[1]: Place Component (1:Pressed)
    - action[2]: Cycle Inventory (1:Pressed)

    Observation Space: A 640x400 RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "A cyberpunk circuit-building puzzle game. Connect a power source to all target nodes by placing circuit components on the grid."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place the selected component. Press shift to cycle through available components."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    GRID_LEFT, GRID_TOP = 40, 40
    CELL_SIZE = 32
    MAX_STEPS = 1000

    # --- Colors (Cyberpunk/Tron Theme) ---
    COLOR_BG = (10, 15, 30)
    COLOR_GRID = (20, 40, 70)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SOURCE = (0, 150, 255)
    COLOR_TARGET_ACTIVE = (255, 50, 50)
    COLOR_TARGET_INACTIVE = (80, 80, 80)
    COLOR_COMPONENT_IDLE = (100, 120, 150)
    COLOR_COMPONENT_POWERED = (100, 255, 255)
    COLOR_UI_TEXT = (200, 220, 255)
    COLOR_UI_BG = (20, 30, 50, 180) # RGBA for transparency

    # --- Component Definitions ---
    # ID: (Name, Connection Deltas)
    COMPONENTS = {
        1: ("Vertical", ((0, -1), (0, 1))),
        2: ("Horizontal", ((-1, 0), (1, 0))),
        3: ("Corner ┓", ((0, -1), (1, 0))),
        4: ("Corner ┏", ((0, -1), (-1, 0))),
        5: ("Corner ┛", ((0, 1), (1, 0))),
        6: ("Corner ┗", ((0, 1), (-1, 0))),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.source_pos = None
        self.targets = None
        self.inventory = None
        self.available_component_types = None
        self.selected_component_type = None

        self.cursor_pos = None
        self.visual_cursor_pos = None
        self.cursor_interpolation_speed = 0.4

        self.powered_cells = None
        self.particles = None

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        self.prev_space_held = False
        self.prev_shift_held = False

        # --- Final Validation ---
        # self.validate_implementation() # Uncomment to run validation on init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []

        self._generate_level()

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        screen_coords = self._grid_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        self.visual_cursor_pos = [float(screen_coords[0]), float(screen_coords[1])]

        self.prev_space_held = False
        self.prev_shift_held = False

        self._trace_circuit()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward_this_step = 0

        # --- Action Handling ---
        # 1. Cursor Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # 2. Cycle Inventory (on press)
        if shift_held and not self.prev_shift_held:
            # # sound: inventory_cycle.wav
            current_idx = self.available_component_types.index(self.selected_component_type)
            next_idx = (current_idx + 1) % len(self.available_component_types)
            self.selected_component_type = self.available_component_types[next_idx]

        # 3. Place Component (on press)
        if space_held and not self.prev_space_held:
            cx, cy = self.cursor_pos
            if self.inventory.get(self.selected_component_type, 0) > 0 and self.grid[cy][cx] == 0:
                # # sound: place_component.wav
                self.grid[cy][cx] = self.selected_component_type
                self.inventory[self.selected_component_type] -= 1
                reward_this_step += 1.0  # Reward for valid placement

                # Recalculate circuit and check for newly deactivated targets
                old_powered_targets = sum(1 for t in self.targets if not t['active'])
                self._trace_circuit()
                new_powered_targets = sum(1 for t in self.targets if not t['active'])
                
                if new_powered_targets > old_powered_targets:
                    # # sound: target_deactivated.wav
                    reward_this_step += 10.0 * (new_powered_targets - old_powered_targets)
                    for target in self.targets:
                        if not target['active'] and (target['pos'][0], target['pos'][1]) in self.powered_cells:
                             self._create_explosion(self._grid_to_screen(target['pos'][0], target['pos'][1]), self.COLOR_COMPONENT_POWERED)

            else:
                # # sound: error.wav
                reward_this_step -= 0.1 # Penalty for invalid placement

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Game Logic Update ---
        self.steps += 1
        self._update_particles()
        
        target_cursor_px = self._grid_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        self.visual_cursor_pos[0] += (target_cursor_px[0] - self.visual_cursor_pos[0]) * self.cursor_interpolation_speed
        self.visual_cursor_pos[1] += (target_cursor_px[1] - self.visual_cursor_pos[1]) * self.cursor_interpolation_speed

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if all(not t['active'] for t in self.targets):
                # # sound: level_win.wav
                self.win_message = "SYSTEM SECURE"
                reward_this_step += 50.0
            else:
                # # sound: level_lose.wav
                self.win_message = "CONNECTION FAILED"
                reward_this_step -= 50.0

        self.score += reward_this_step
        return self._get_observation(), reward_this_step, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "components_left": sum(self.inventory.values())}

    # --- Helper Methods ---

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.source_pos = (0, self.GRID_ROWS // 2)
        self.targets = [{'pos': (self.GRID_COLS - 1, self.GRID_ROWS // 2 - 2), 'active': True},
                        {'pos': (self.GRID_COLS - 1, self.GRID_ROWS // 2 + 2), 'active': True}]
        
        self.inventory = {1: 5, 2: 8, 3: 3, 4: 3, 5: 3, 6: 3}
        self.available_component_types = sorted(self.inventory.keys())
        self.selected_component_type = self.available_component_types[0]

    def _check_termination(self):
        win = all(not t['active'] for t in self.targets)
        lose = sum(self.inventory.values()) == 0 and not win
        timeout = self.steps >= self.MAX_STEPS
        return win or lose or timeout

    def _trace_circuit(self):
        self.powered_cells = set()
        q = [self.source_pos]
        visited = {self.source_pos}

        while q:
            x, y = q.pop(0)
            self.powered_cells.add((x, y))

            comp_id = self.grid[y][x]
            if comp_id == 0 and (x, y) != self.source_pos:
                continue

            # Connections from source are omnidirectional
            connections = self.COMPONENTS.values() if (x,y) == self.source_pos else [self.COMPONENTS[comp_id]]
            
            for comp_name, deltas in connections:
                for dx, dy in deltas:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and (nx, ny) not in visited:
                        neighbor_id = self.grid[ny][nx]
                        if neighbor_id > 0:
                            # Check if neighbor connects back
                            neighbor_deltas = self.COMPONENTS[neighbor_id][1]
                            for ndx, ndy in neighbor_deltas:
                                if nx + ndx == x and ny + ndy == y:
                                    visited.add((nx, ny))
                                    q.append((nx, ny))
                                    break
        
        # Update targets
        for target in self.targets:
            if (target['pos'][0], target['pos'][1]) in self.powered_cells:
                target['active'] = False

    def _grid_to_screen(self, x, y):
        return (self.GRID_LEFT + x * self.CELL_SIZE + self.CELL_SIZE // 2,
                self.GRID_TOP + y * self.CELL_SIZE + self.CELL_SIZE // 2)

    def _create_explosion(self, pos, color):
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(20, 40)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['vel'][1] += 0.05 # gravity

    # --- Rendering Methods ---

    def _render_game(self):
        # 1. Background Grid
        for x in range(self.GRID_COLS + 1):
            px = self.GRID_LEFT + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_TOP), (px, self.GRID_TOP + self.GRID_ROWS * self.CELL_SIZE))
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_TOP + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_LEFT, py), (self.GRID_LEFT + self.GRID_COLS * self.CELL_SIZE, py))

        # 2. Components
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                comp_id = self.grid[y][x]
                if comp_id > 0:
                    center_x, center_y = self._grid_to_screen(x, y)
                    color = self.COLOR_COMPONENT_POWERED if (x, y) in self.powered_cells else self.COLOR_COMPONENT_IDLE
                    deltas = self.COMPONENTS[comp_id][1]
                    for dx, dy in deltas:
                        end_x = center_x + dx * self.CELL_SIZE // 2
                        end_y = center_y + dy * self.CELL_SIZE // 2
                        pygame.draw.line(self.screen, color, (center_x, center_y), (end_x, end_y), 3)

        # 3. Source and Targets
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 4
        self._draw_glow_circle(self.screen, self._grid_to_screen(*self.source_pos), self.COLOR_SOURCE, self.CELL_SIZE // 3, int(4 + pulse))
        
        for target in self.targets:
            color = self.COLOR_TARGET_ACTIVE if target['active'] else self.COLOR_TARGET_INACTIVE
            is_powered = (target['pos'][0], target['pos'][1]) in self.powered_cells
            glow = int(4 + pulse) if is_powered else 4
            self._draw_glow_circle(self.screen, self._grid_to_screen(*target['pos']), color, self.CELL_SIZE // 4, glow)

        # 4. Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 40.0))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, 2, 2, 2, color)
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

        # 5. Cursor
        cursor_rect = pygame.Rect(0, 0, self.CELL_SIZE, self.CELL_SIZE)
        cursor_rect.center = (int(self.visual_cursor_pos[0]), int(self.visual_cursor_pos[1]))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=4)


    def _render_ui(self):
        # UI Background Panel
        panel_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 60, self.SCREEN_WIDTH, 60)
        s = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (panel_rect.left, panel_rect.top))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        # Inventory Display
        inv_start_x = 20
        for i, comp_type in enumerate(self.available_component_types):
            count = self.inventory.get(comp_type, 0)
            color = self.COLOR_CURSOR if comp_type == self.selected_component_type else self.COLOR_UI_TEXT
            
            # Draw selection box
            if comp_type == self.selected_component_type:
                sel_rect = pygame.Rect(inv_start_x - 5, self.SCREEN_HEIGHT - 55, 50, 50)
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, sel_rect, 1, border_radius=4)

            # Draw component symbol
            name, deltas = self.COMPONENTS[comp_type]
            center_x, center_y = inv_start_x + 20, self.SCREEN_HEIGHT - 30
            for dx, dy in deltas:
                end_x = center_x + dx * 10
                end_y = center_y + dy * 10
                pygame.draw.line(self.screen, color, (center_x, center_y), (end_x, end_y), 2)
            
            # Draw count
            count_text = self.font_small.render(f"x{count}", True, color)
            self.screen.blit(count_text, (inv_start_x + 35, self.SCREEN_HEIGHT - 25))
            
            inv_start_x += 70

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_CURSOR)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _draw_glow_circle(self, surface, pos, color, radius, glow_size):
        pos = (int(pos[0]), int(pos[1]))
        for i in range(glow_size, 0, -1):
            alpha = int(100 * (1 - (i / glow_size)))
            glow_color = color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius + i, glow_color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cyber Circuit")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # Action mapping for human player
        movement = 0 # None
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'R'
                    obs, info = env.reset()
                    total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}")
    env.close()