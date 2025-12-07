
# Generated: 2025-08-27T23:34:44.405598
# Source Brief: brief_03508.md
# Brief Index: 3508

        
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
        "Controls: ←→ to select an interactive element. Press space to activate the selected element."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a procedurally generated isometric room by solving interconnected visual puzzles within a time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 30 * 60 * 2 # 2 minutes for a shorter, more intense game
    TIME_LIMIT_SECONDS = 120

    # Colors
    COLOR_BG = (25, 28, 44)
    COLOR_WALL = (45, 50, 77)
    COLOR_FLOOR = (35, 40, 60)
    COLOR_GRID = (55, 62, 92)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TIMER_WARN = (255, 80, 80)
    COLOR_INTERACTIVE = (60, 140, 255)
    COLOR_SELECTED = (255, 200, 80)
    COLOR_SOLVED = (80, 255, 120)
    COLOR_INCORRECT = (255, 60, 60)
    
    # Puzzle constants
    NUM_PILLARS = 3
    NUM_PANELS = 3
    NUM_SLIDERS = 3
    PILLAR_COLORS = [(255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50)]
    PANEL_SYMBOLS = ['circle', 'square', 'triangle', 'diamond']
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.rng = None
        self.cursor_pos = 0
        self.last_space_state = False
        self.interactive_elements = []
        self.particles = []
        self.flash_effect = (0, None) # duration, color
        
        # Puzzle states
        self.pillar_states = []
        self.panel_states = []
        self.slider_states = []
        
        self.pillar_solution = []
        self.panel_solution = []
        self.slider_solution = []
        
        self.puzzles_solved = [False, False, False] # Pillars, Panels, Sliders
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def _world_to_iso(self, x, y, z=0):
        iso_x = (x - y) * 1.5
        iso_y = (x + y) * 0.75 - z
        return iso_x + self.SCREEN_WIDTH // 2, iso_y + self.SCREEN_HEIGHT // 4

    def _draw_iso_cube(self, surface, x, y, z, size, height, color, edge_color=None):
        if edge_color is None:
            edge_color = tuple(max(0, c-40) for c in color)

        points = [
            self._world_to_iso(x, y, z),
            self._world_to_iso(x + size, y, z),
            self._world_to_iso(x + size, y + size, z),
            self._world_to_iso(x, y + size, z),
            self._world_to_iso(x, y, z + height),
            self._world_to_iso(x + size, y, z + height),
            self._world_to_iso(x + size, y + size, z + height),
            self._world_to_iso(x, y + size, z + height),
        ]
        
        # Top face
        pygame.draw.polygon(surface, color, [points[4], points[5], points[6], points[7]])
        pygame.draw.aalines(surface, edge_color, True, [points[4], points[5], points[6], points[7]])
        # Right face
        pygame.draw.polygon(surface, tuple(max(0, c-20) for c in color), [points[1], points[2], points[6], points[5]])
        pygame.draw.aalines(surface, edge_color, True, [points[1], points[2], points[6], points[5]])
        # Left face
        pygame.draw.polygon(surface, tuple(max(0, c-40) for c in color), [points[2], points[3], points[7], points[6]])
        pygame.draw.aalines(surface, edge_color, True, [points[2], points[3], points[7], points[6]])

    def _draw_symbol(self, surface, symbol, center_pos, size, color):
        x, y = center_pos
        if symbol == 'circle':
            pygame.gfxdraw.aacircle(surface, int(x), int(y), int(size/2), color)
            pygame.gfxdraw.filled_circle(surface, int(x), int(y), int(size/2), color)
        elif symbol == 'square':
            rect = pygame.Rect(x - size/2, y - size/2, size, size)
            pygame.draw.rect(surface, color, rect)
        elif symbol == 'triangle':
            points = [(x, y - size/2), (x - size/2, y + size/2), (x + size/2, y + size/2)]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        elif symbol == 'diamond':
            points = [(x, y - size/2), (x - size/2, y), (x, y + size/2), (x + size/2, y)]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
            
    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                'life': self.rng.integers(15, 30),
                'color': color
            })
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.cursor_pos = 0
        self.last_space_state = False
        self.particles.clear()
        self.flash_effect = (0, None)

        # --- Procedural Puzzle Generation ---
        # 1. Generate solutions
        self.pillar_solution = self.rng.choice(len(self.PILLAR_COLORS), self.NUM_PILLARS, replace=True).tolist()
        self.panel_solution = self.rng.choice(len(self.PANEL_SYMBOLS), self.NUM_PANELS, replace=True).tolist()
        self.slider_solution = self.rng.integers(0, 101, size=self.NUM_SLIDERS).tolist()

        # 2. Generate initial (non-solved) states
        self.pillar_states = [(s + 1) % len(self.PILLAR_COLORS) for s in self.pillar_solution]
        self.panel_states = [(s + 1) % len(self.PANEL_SYMBOLS) for s in self.panel_solution]
        self.slider_states = [0] * self.NUM_SLIDERS
        
        # 3. Define interactive elements
        self.interactive_elements = []
        for i in range(self.NUM_PILLARS):
            self.interactive_elements.append({'type': 'pillar', 'id': i, 'pos': (-60 + i * 40, 80)})
        for i in range(self.NUM_PANELS):
            self.interactive_elements.append({'type': 'panel', 'id': i, 'pos': (-40, -10 + i * 40)})
        for i in range(self.NUM_SLIDERS):
            self.interactive_elements.append({'type': 'slider', 'id': i, 'pos': (80, -60 + i * 40)})

        self.puzzles_solved = [False, False, False]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        space_pressed = space_held and not self.last_space_state
        self.last_space_state = space_held
        
        # --- Update Game Logic ---
        self.steps += 1
        self.time_left -= 1
        
        # Update cursor
        if movement in [3, 1]: # Left or Up
            self.cursor_pos = (self.cursor_pos - 1) % len(self.interactive_elements)
        elif movement in [4, 2]: # Right or Down
            self.cursor_pos = (self.cursor_pos + 1) % len(self.interactive_elements)
            
        # Handle interaction
        if space_pressed:
            element = self.interactive_elements[self.cursor_pos]
            el_type = element['type']
            el_id = element['id']
            
            action_valid = True
            if el_type == 'pillar':
                self.pillar_states[el_id] = (self.pillar_states[el_id] + 1) % len(self.PILLAR_COLORS)
            elif el_type == 'panel':
                if self.puzzles_solved[0]: # Panels are only active after pillars are solved
                    self.panel_states[el_id] = (self.panel_states[el_id] + 1) % len(self.PANEL_SYMBOLS)
                else: action_valid = False
            elif el_type == 'slider':
                if self.puzzles_solved[1]: # Sliders are only active after panels are solved
                    self.slider_states[el_id] = (self.slider_states[el_id] + 10) % 110
                else: action_valid = False
            
            if action_valid:
                # # Sound: bleep.wav
                self.flash_effect = (5, self.COLOR_INTERACTIVE)
                world_pos = self.interactive_elements[self.cursor_pos]['pos']
                screen_pos = self._world_to_iso(world_pos[0], world_pos[1], 30)
                self._create_particles(screen_pos, self.COLOR_SELECTED, 10)
            else:
                # # Sound: error.wav
                self.flash_effect = (5, self.COLOR_INCORRECT)


        # --- Check puzzle solutions and calculate rewards ---
        # Pillars
        if not self.puzzles_solved[0]:
            if self.pillar_states == self.pillar_solution:
                self.puzzles_solved[0] = True
                reward += 25
                # # Sound: puzzle_solve_1.wav
                self.flash_effect = (15, self.COLOR_SOLVED)
                for el in self.interactive_elements:
                    if el['type'] == 'pillar':
                        screen_pos = self._world_to_iso(el['pos'][0], el['pos'][1], 30)
                        self._create_particles(screen_pos, self.COLOR_SOLVED, 30)

        # Panels
        if self.puzzles_solved[0] and not self.puzzles_solved[1]:
            if self.panel_states == self.panel_solution:
                self.puzzles_solved[1] = True
                reward += 25
                # # Sound: puzzle_solve_2.wav
                self.flash_effect = (15, self.COLOR_SOLVED)
                for el in self.interactive_elements:
                    if el['type'] == 'panel':
                        screen_pos = self._world_to_iso(el['pos'][0], el['pos'][1], 30)
                        self._create_particles(screen_pos, self.COLOR_SOLVED, 30)

        # Sliders
        if self.puzzles_solved[1] and not self.puzzles_solved[2]:
            current_dists = sum(abs(c - s) for c, s in zip(self.slider_states, self.slider_solution))
            if current_dists == 0:
                self.puzzles_solved[2] = True
                reward += 25
                # # Sound: puzzle_solve_3.wav
                self.flash_effect = (15, self.COLOR_SOLVED)
                for el in self.interactive_elements:
                    if el['type'] == 'slider':
                        screen_pos = self._world_to_iso(el['pos'][0], el['pos'][1], 30)
                        self._create_particles(screen_pos, self.COLOR_SOLVED, 30)
            elif space_pressed and self.interactive_elements[self.cursor_pos]['type'] == 'slider':
                # Reward shaping for sliders
                prev_val = (self.slider_states[el_id] - 10) % 110
                prev_dist = abs(prev_val - self.slider_solution[el_id])
                new_dist = abs(self.slider_states[el_id] - self.slider_solution[el_id])
                if new_dist < prev_dist: reward += 0.5
                else: reward -= 0.5

        # --- Update particles and effects ---
        if self.flash_effect[0] > 0:
            self.flash_effect = (self.flash_effect[0] - 1, self.flash_effect[1])
            
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # --- Check termination conditions ---
        win_condition = all(self.puzzles_solved)
        lose_condition = self.time_left <= 0 or self.steps >= self.MAX_STEPS
        
        terminated = win_condition or lose_condition
        if terminated:
            self.game_over = True
            if win_condition:
                reward += 100
                # # Sound: win.wav
            else: # Lose condition
                reward -= 100
                # # Sound: lose.wav
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _render_game(self):
        # --- Background and Room ---
        # Floor
        floor_points = [self._world_to_iso(-100, -100), self._world_to_iso(100, -100), 
                        self._world_to_iso(100, 100), self._world_to_iso(-100, 100)]
        pygame.gfxdraw.filled_polygon(self.screen, floor_points, self.COLOR_FLOOR)
        
        # Grid lines
        for i in range(-100, 101, 20):
            pygame.draw.aaline(self.screen, self.COLOR_GRID, self._world_to_iso(i, -100), self._world_to_iso(i, 100))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, self._world_to_iso(-100, i), self._world_to_iso(100, i))

        # Walls
        self._draw_iso_cube(self.screen, -105, -100, 0, 5, 200, self.COLOR_WALL)
        self._draw_iso_cube(self.screen, -100, 100, 0, 205, 200, self.COLOR_WALL)

        # Door
        door_color = self.COLOR_SOLVED if all(self.puzzles_solved) else self.COLOR_WALL
        self._draw_iso_cube(self.screen, -105, 0, 0, 5, 60, door_color)
        if all(self.puzzles_solved):
            glow_surf = pygame.Surface((100, 200), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_SOLVED, 30), (0,0,100,200), border_radius=10)
            door_pos = self._world_to_iso(-100, 30, 100)
            self.screen.blit(glow_surf, (door_pos[0]-50, door_pos[1]-100), special_flags=pygame.BLEND_RGBA_ADD)


        # --- Render Puzzles ---
        # Pillars
        for i in range(self.NUM_PILLARS):
            x, y = self.interactive_elements[i]['pos']
            color = self.PILLAR_COLORS[self.pillar_states[i]]
            self._draw_iso_cube(self.screen, x, y, 0, 20, 50, color)
            if self.puzzles_solved[0]:
                self._draw_iso_cube(self.screen, x, y, 50, 20, 5, self.COLOR_SOLVED)
        
        # Panels
        for i in range(self.NUM_PANELS):
            el_index = i + self.NUM_PILLARS
            x, y = self.interactive_elements[el_index]['pos']
            is_active = self.puzzles_solved[0]
            color = self.COLOR_WALL if not is_active else (40, 45, 70)
            self._draw_iso_cube(self.screen, x, y, 20, 40, 5, color)
            if is_active:
                symbol_idx = self.panel_states[i]
                symbol_color = self.COLOR_SOLVED if self.puzzles_solved[1] else self.COLOR_TEXT
                symbol_pos = self._world_to_iso(x + 20, y + 20, 27)
                self._draw_symbol(self.screen, self.PANEL_SYMBOLS[symbol_idx], symbol_pos, 20, symbol_color)
                
        # Sliders
        for i in range(self.NUM_SLIDERS):
            el_index = i + self.NUM_PILLARS + self.NUM_PANELS
            x, y = self.interactive_elements[el_index]['pos']
            is_active = self.puzzles_solved[1]
            track_color = self.COLOR_SOLVED if self.puzzles_solved[2] else self.COLOR_WALL
            self._draw_iso_cube(self.screen, x, y, 20, 5, 100, track_color)
            if is_active:
                slider_z = 20 + self.slider_states[i] * 0.8
                self._draw_iso_cube(self.screen, x-5, y-5, slider_z, 15, 5, self.COLOR_INTERACTIVE)

        # --- Render Hints ---
        # Panel solution hint (on wall)
        if self.puzzles_solved[0]:
            self._draw_iso_cube(self.screen, -100, -80, 80, 60, 5, (20,20,20))
            for i, s_idx in enumerate(self.panel_solution):
                pos = self._world_to_iso(-85 + i*20, -65, 87)
                self._draw_symbol(self.screen, self.PANEL_SYMBOLS[s_idx], pos, 10, self.COLOR_SOLVED)
        
        # Slider solution hint (on wall)
        if self.puzzles_solved[1]:
            self._draw_iso_cube(self.screen, 20, 100, 80, 5, 60, (20,20,20))
            for i, val in enumerate(self.slider_solution):
                z = 82 + val * 0.4
                self._draw_iso_cube(self.screen, 20, 100 + i*20, z, 5, 2, self.COLOR_SOLVED)

        # --- Render Cursor ---
        selected_el = self.interactive_elements[self.cursor_pos]
        x, y = selected_el['pos']
        w, h = (25, 60) if selected_el['type'] == 'pillar' else (45, 10) if selected_el['type'] == 'panel' else (15, 110)
        z = 0 if selected_el['type'] == 'pillar' else 20
        if selected_el['type'] == 'slider': z = 20 + self.slider_states[selected_el['id']] * 0.8
        
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 5
        center_pos = self._world_to_iso(x + w/4, y + w/4, z + h/2)
        pygame.gfxdraw.aacircle(self.screen, int(center_pos[0]), int(center_pos[1]), int(15 + pulse), (*self.COLOR_SELECTED, 180))
        
        # --- Render Particles & Effects ---
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, p['life'] / 6))
        
        if self.flash_effect[0] > 0:
            flash_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            alpha = int(self.flash_effect[0] * 20)
            flash_surf.fill((*self.flash_effect[1], alpha))
            self.screen.blit(flash_surf, (0,0))

    def _render_ui(self):
        # --- Timer ---
        time_sec = self.time_left // self.FPS
        time_str = f"{time_sec // 60:02d}:{time_sec % 60:02d}"
        time_color = self.COLOR_TEXT if time_sec > 10 or (self.steps % self.FPS < self.FPS/2) else self.COLOR_TIMER_WARN
        time_surf = self.font_medium.render(time_str, True, time_color)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
        # --- Score ---
        score_str = f"Score: {self.score}"
        score_surf = self.font_small.render(score_str, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # --- Game Over Message ---
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "ROOM ESCAPED" if all(self.puzzles_solved) else "TIME UP"
            msg_surf = self.font_large.render(msg, True, self.COLOR_SOLVED if all(self.puzzles_solved) else self.COLOR_TIMER_WARN)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left / self.FPS,
            "puzzles_solved": self.puzzles_solved,
        }

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

# Example usage to run and display the game
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Escape Room")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Game loop for human control
    while not done:
        # Action defaults
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render the observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

        if done:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Pause for 3 seconds before quitting
            
    env.close()