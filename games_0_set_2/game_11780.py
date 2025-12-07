import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    GameEnv: An underwater structural engineering puzzle.

    Players place color-coded pressure plates onto a grid to build a stable
    structure against crushing water pressure that increases with depth.

    **Core Gameplay Loop:**
    1.  **Select Plate:** Use the 'Shift' key to cycle through available plate types.
    2.  **Position Cursor:** Use arrow keys to move the cursor on the building grid.
    3.  **Place Plate:** Press 'Space' to place the selected plate, consuming resources.
    4.  **Achieve Stability:** The structure's stability is constantly calculated.
        Pressure from the water and plates above is distributed downwards.
    5.  **Descend:** To complete a level, build a stable structure with at least
        one plate in the top row.
    6.  **Survive:** Reach Depth Level 10 to win. If the total pressure load
        exceeds the structure's total resistance, it collapses, ending the game.

    **Visuals:**
    -   A deep-sea aesthetic with ambient particles (bubbles).
    -   Plates are rendered as rounded rectangles.
    -   A color overlay on each plate visualizes the pressure it's enduring,
        shifting from blue (low pressure) to red (high pressure).
    -   A clean UI displays depth, score, resources, and the selected plate.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Build a stable structure underwater to withstand increasing pressure. "
        "Place plates on a grid to reach the target depth without collapsing."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'space' to place a plate "
        "and 'shift' to cycle through plate types."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 10, 8
    CELL_SIZE = 32
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 50, 80)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (20, 35, 60, 180)
    COLOR_LOW_PRESSURE = (0, 150, 255)
    COLOR_HIGH_PRESSURE = (255, 50, 50)
    
    PLATE_SPECS = {
        1: {'name': 'Basic', 'color': (60, 120, 220), 'resistance': 15, 'cost': 10},
        2: {'name': 'Reinforced', 'color': (60, 220, 120), 'resistance': 30, 'cost': 25},
        3: {'name': 'Plated', 'color': (220, 120, 60), 'resistance': 60, 'cost': 50},
    }
    
    MAX_LEVEL = 10
    MAX_STEPS = 1000
    COLLAPSE_THRESHOLD = 1.0 # Collapse if load > resistance * threshold

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # State variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.resources = 0
        self.grid = None
        self.pressure_grid = None
        self.cursor_pos = [0, 0]
        self.available_plate_types = []
        self.selected_plate_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.bubbles = []
        self.collapse_info = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.collapse_info = None
        self.info_cost = 0
        self.info_level_up = False
        self.info_collapse = False
        self.info_win = False


        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS - 1]
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._setup_level()
        self._create_bubbles(100)
        
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.pressure_grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=float)
        
        self.resources = 150 + self.level * 20
        
        if self.level < 3:
            self.available_plate_types = [1]
        elif self.level < 6:
            self.available_plate_types = [1, 2]
        else:
            self.available_plate_types = [1, 2, 3]
        
        self.selected_plate_idx = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Survival reward per step
        self.steps += 1
        
        # Reset transient info flags
        self.info_cost = 0
        self.info_level_up = False
        self.info_collapse = False
        self.info_win = False

        self._handle_input(action)

        if self.collapse_info:
            self.collapse_info['timer'] -= 1
            if self.collapse_info['timer'] <= 0:
                self.game_over = True
        
        terminated = self._check_termination()
        
        # Calculate event-based rewards
        current_info = self._get_info()
        if current_info.get("level_up"):
            reward += 50.0
            self.score += 500
        if current_info.get("collapse"):
            reward -= 50.0
            self.score -= 500
        if current_info.get("win"):
            reward += 100.0
            self.score += 1000
        if current_info.get("cost"):
            reward -= current_info["cost"] * 0.1
            self.score -= current_info["cost"]
        
        self.score += 1 # Base score for surviving a step

        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        if not self.collapse_info:
            # --- Movement ---
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            if movement == 2: self.cursor_pos[1] += 1  # Down
            if movement == 3: self.cursor_pos[0] -= 1  # Left
            if movement == 4: self.cursor_pos[0] += 1  # Right
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

            # --- Cycle Plate ---
            if shift_pressed:
                self.selected_plate_idx = (self.selected_plate_idx + 1) % len(self.available_plate_types)

            # --- Place Plate ---
            if space_pressed:
                self._place_plate()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _place_plate(self):
        x, y = self.cursor_pos
        
        # Can only place on empty cells, and either on the floor or on another plate
        can_place = self.grid[y, x] == 0 and \
                    (y == self.GRID_ROWS - 1 or (y < self.GRID_ROWS - 1 and self.grid[y + 1, x] != 0))

        plate_type_id = self.available_plate_types[self.selected_plate_idx]
        cost = self.PLATE_SPECS[plate_type_id]['cost']

        if can_place and self.resources >= cost:
            self.resources -= cost
            self.grid[y, x] = plate_type_id
            self.info_cost = cost # For reward calculation
            self._create_particles(self.GRID_X + x * self.CELL_SIZE + self.CELL_SIZE//2, 
                                   self.GRID_Y + y * self.CELL_SIZE + self.CELL_SIZE//2)
            
            self._update_stability_and_check_progress()
        else:
            pass

    def _update_stability_and_check_progress(self):
        # --- Calculate Pressure ---
        base_pressure = self.level
        self.pressure_grid.fill(0)
        total_load = 0
        total_resistance = 0

        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                plate_id = self.grid[y, x]
                if plate_id > 0:
                    pressure_from_above = self.pressure_grid[y - 1, x] if y > 0 else 0
                    current_pressure = base_pressure + pressure_from_above
                    self.pressure_grid[y, x] = current_pressure
                    
                    total_load += current_pressure
                    total_resistance += self.PLATE_SPECS[plate_id]['resistance']

        # --- Check for Collapse ---
        is_stable = total_load <= total_resistance * self.COLLAPSE_THRESHOLD
        if not is_stable:
            self.info_collapse = True
            self.collapse_info = {'timer': 60, 'plates': []}
            for y in range(self.GRID_ROWS):
                for x in range(self.GRID_COLS):
                    if self.grid[y,x] > 0:
                        self.collapse_info['plates'].append({
                            'pos': [self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE],
                            'vel': [random.uniform(-1, 1), random.uniform(-2, 0)],
                            'rot': 0,
                            'rot_vel': random.uniform(-5, 5),
                            'type': self.grid[y,x]
                        })
            return

        # --- Check for Level Completion ---
        # Condition: structure is stable and at least one block is in the top row
        top_row_built = np.any(self.grid[0, :] > 0)
        if is_stable and top_row_built:
            self.level += 1
            self.info_level_up = True
            if self.level > self.MAX_LEVEL:
                self.info_win = True
                self.game_over = True
            else:
                self._setup_level()

    def _check_termination(self):
        if self.collapse_info and self.collapse_info['timer'] <= 0:
            return True
        if self.info_win:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_render_bubbles()
        self._render_grid()
        if self.collapse_info:
            self._render_collapse_animation()
        else:
            self._render_structure()
            self._render_cursor()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        info = {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "resources": self.resources,
        }
        if self.info_cost > 0:
            info["cost"] = self.info_cost
        if self.info_level_up:
            info["level_up"] = True
        if self.info_collapse:
            info["collapse"] = True
        if self.info_win:
            info["win"] = True
        return info

    # --- Rendering Methods ---
    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT))

    def _render_structure(self):
        max_possible_pressure = self.level * self.GRID_ROWS
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                plate_id = self.grid[y, x]
                if plate_id > 0:
                    plate_spec = self.PLATE_SPECS[plate_id]
                    rect = pygame.Rect(self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE,
                                       self.CELL_SIZE, self.CELL_SIZE)
                    
                    # Draw plate with pressure visualization
                    pressure = self.pressure_grid[y, x]
                    pressure_ratio = min(1.0, pressure / max(1, max_possible_pressure))
                    
                    base_color = pygame.Color(plate_spec['color'])
                    pressure_color = base_color.lerp(self.COLOR_HIGH_PRESSURE, pressure_ratio)
                    
                    # Draw rounded rectangle
                    pygame.draw.rect(self.screen, pressure_color, rect.inflate(-4, -4), border_radius=4)
                    
                    # Add a subtle highlight/edge
                    highlight_color = tuple(min(255, c + 40) for c in pressure_color[:3])
                    pygame.draw.rect(self.screen, highlight_color, rect.inflate(-4, -4), 1, border_radius=4)

    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE,
                           self.CELL_SIZE, self.CELL_SIZE)
        
        # Pulsing alpha effect
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        cursor_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        cursor_surface.fill((*self.COLOR_CURSOR, alpha))
        self.screen.blit(cursor_surface, rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)

    def _render_ui(self):
        # Panel background
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 65), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, self.SCREEN_HEIGHT - 65))
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (0, self.SCREEN_HEIGHT - 65), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT-65), 1)

        # Info Text
        texts = [
            f"DEPTH: {self.level}/{self.MAX_LEVEL}",
            f"SCORE: {int(self.score)}",
            f"RESOURCES: {self.resources}",
        ]
        for i, text in enumerate(texts):
            rendered_text = self.font_main.render(text, True, self.COLOR_UI_TEXT)
            self.screen.blit(rendered_text, (20 + i * 200, self.SCREEN_HEIGHT - 48))

        # Selected Plate Info
        if not self.collapse_info and len(self.available_plate_types) > 0:
            plate_id = self.available_plate_types[self.selected_plate_idx]
            spec = self.PLATE_SPECS[plate_id]
            
            # Plate preview
            preview_rect = pygame.Rect(self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 55, 40, 40)
            pygame.draw.rect(self.screen, spec['color'], preview_rect, border_radius=5)
            pygame.draw.rect(self.screen, tuple(min(255, c+40) for c in spec['color']), preview_rect, 1, border_radius=5)

            # Plate text
            name_text = self.font_small.render(spec['name'], True, self.COLOR_UI_TEXT)
            cost_text = self.font_small.render(f"Cost: {spec['cost']}", True, self.COLOR_UI_TEXT)
            self.screen.blit(name_text, (self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT - 52))
            self.screen.blit(cost_text, (self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT - 32))
            
    def _render_collapse_animation(self):
        for plate in self.collapse_info['plates']:
            plate['vel'][1] += 0.1 # Gravity
            plate['pos'][0] += plate['vel'][0]
            plate['pos'][1] += plate['vel'][1]
            plate['rot'] += plate['rot_vel']
            
            spec = self.PLATE_SPECS[plate['type']]
            color = spec['color']
            
            size = self.CELL_SIZE - 4
            plate_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            plate_rect = pygame.Rect(0, 0, size, size)
            pygame.draw.rect(plate_surf, color, plate_rect, border_radius=4)
            
            alpha = max(0, 255 * (self.collapse_info['timer'] / 60))
            plate_surf.set_alpha(alpha)
            
            rotated_surf = pygame.transform.rotate(plate_surf, plate['rot'])
            new_rect = rotated_surf.get_rect(center=(plate['pos'][0] + size//2, plate['pos'][1] + size//2))
            self.screen.blit(rotated_surf, new_rect.topleft)

    def _create_particles(self, x, y, count=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(15, 30),
                'color': random.choice([(200,200,255), (255,255,255), (150,200,255)])
            })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = 255 * (p['life'] / 30)
                size = int(p['life'] / 10) + 1
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), size, (*p['color'], int(alpha))
                )

    def _create_bubbles(self, count=50):
        self.bubbles = []
        for _ in range(count):
            self.bubbles.append({
                'pos': [random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)],
                'speed': random.uniform(0.2, 1.0),
                'size': random.randint(1, 4),
                'alpha': random.randint(20, 80)
            })

    def _update_and_render_bubbles(self):
        for b in self.bubbles:
            b['pos'][1] -= b['speed']
            if b['pos'][1] < -b['size']:
                b['pos'][0] = random.uniform(0, self.SCREEN_WIDTH)
                b['pos'][1] = self.SCREEN_HEIGHT + b['size']
            
            pygame.gfxdraw.filled_circle(
                self.screen, int(b['pos'][0]), int(b['pos'][1]), b['size'], (*self.COLOR_UI_TEXT, b['alpha'])
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(b['pos'][0]), int(b['pos'][1]), b['size'], (*self.COLOR_UI_TEXT, b['alpha'])
            )
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is not used by the evaluation environment.
    
    # Un-dummy the video driver to see the game
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Underwater Pressure Puzzle")
    clock = pygame.time.Clock()

    while running:
        # --- Action mapping from keyboard to MultiDiscrete ---
        movement = 0  # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()
    env.close()