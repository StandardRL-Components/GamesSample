import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:29:35.926106
# Source Brief: brief_02273.md
# Brief Index: 2273
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
        "Restore power to the neon city by completing circuits. Pick up and place letters to form words along the grid pathways."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to pick up or place a letter. "
        "Press shift to teleport a held letter, which costs energy."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Visual Style ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_GRID = (30, 45, 75)
        self.COLOR_CIRCUIT_INCOMPLETE = (180, 40, 40)
        self.COLOR_CIRCUIT_COMPLETE = (40, 220, 120)
        self.COLOR_LETTER_NORMAL = (100, 150, 255)
        self.COLOR_LETTER_MAGNETIZED = (255, 255, 100)
        self.COLOR_CURSOR = (220, 80, 220)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_ENERGY_BAR = (50, 200, 255)
        self.COLOR_ENERGY_BG = (50, 50, 80)
        self.COLOR_WIN = (100, 255, 150)
        self.COLOR_LOSE = (255, 100, 100)

        # --- Game Configuration ---
        self.GRID_COLS = 20
        self.GRID_ROWS = 12
        self.CELL_SIZE = 30
        self.GRID_OFFSET_X = (self.screen_width - self.GRID_COLS * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.screen_height - self.GRID_ROWS * self.CELL_SIZE) // 2
        self.MAX_STEPS = 1000
        self.TELEPORT_COST = 5
        self.WORD_LIST = ["POWER", "CIRCUIT", "ENERGY", "NEON", "GRID", "SYSTEM", "ONLINE", "NODE", "LINK", "CODE", "VOLT", "BYTE"]

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.max_energy = 0
        self.energy = 0
        self.circuits = []
        self.letters = []
        self.grid = []
        self.cursor_pos = [0, 0]
        self.magnetized_letter = None
        self.particles = []
        self.prev_space_state = 0
        self.prev_shift_state = 0
        self.win_message = ""
        
        # --- Initialization ---
        # The reset method will be called to initialize the state, so no need to call _generate_level here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        self.max_energy = 50
        self.energy = self.max_energy
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.magnetized_letter = None
        self.particles = []
        
        self.prev_space_state = 0
        self.prev_shift_state = 0
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_state
        shift_pressed = shift_held and not self.prev_shift_state

        # 1. Handle cursor movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_x = self.cursor_pos[0] + dx
            new_y = self.cursor_pos[1] + dy
            if 0 <= new_x < self.GRID_COLS and 0 <= new_y < self.GRID_ROWS:
                self.cursor_pos = [new_x, new_y]

        # 2. Handle interactions
        if space_pressed: # Magnetize / Place
            reward += self._handle_placement(is_teleport=False)
        elif shift_pressed: # Teleport
            if self.magnetized_letter and self.energy >= self.TELEPORT_COST:
                self.energy -= self.TELEPORT_COST
                reward += self._handle_placement(is_teleport=True)
                reward -= 0.1 # Small cost for teleport action
                # sfx: teleport_zap
                self._create_teleport_effect(self._grid_to_pixel(self.cursor_pos))
            elif self.magnetized_letter and self.energy < self.TELEPORT_COST:
                # sfx: action_fail
                pass

        # 3. Update game state
        reward += self._update_circuits()

        # 4. Update particles and other animations
        self._update_particles()
        if self.magnetized_letter:
            self.magnetized_letter['pixel_pos'][0] += (self._grid_to_pixel(self.cursor_pos)[0] - self.magnetized_letter['pixel_pos'][0]) * 0.4
            self.magnetized_letter['pixel_pos'][1] += (self._grid_to_pixel(self.cursor_pos)[1] - self.magnetized_letter['pixel_pos'][1]) * 0.4

        # 5. Check for termination
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated
        
        self.prev_space_state = space_held
        self.prev_shift_state = shift_held

        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated:
            self.win_message = "SYSTEM TIMEOUT"
            self.game_over = True
            reward -= 10.0

        return self._get_observation(), reward, self.game_over, truncated, self._get_info()

    def _handle_placement(self, is_teleport):
        reward = 0
        cx, cy = self.cursor_pos
        
        if self.magnetized_letter is None:
            # Try to pick up a letter
            if self.grid[cy][cx] is not None:
                self.magnetized_letter = self.grid[cy][cx]
                self.grid[cy][cx] = None
                self.magnetized_letter['magnetized'] = True
                self.magnetized_letter['pixel_pos'] = list(self._grid_to_pixel(self.cursor_pos))
                # sfx: magnetize_on
        else:
            # Try to place a letter
            if self.grid[cy][cx] is None:
                placed_letter = self.magnetized_letter
                placed_letter['pos'] = [cx, cy]
                placed_letter['magnetized'] = False
                self.grid[cy][cx] = placed_letter
                self.magnetized_letter = None
                
                # Check if placement was correct
                is_correct = False
                for circuit in self.circuits:
                    if tuple(self.cursor_pos) in circuit['slot_map']:
                        slot_index = circuit['slot_map'][tuple(self.cursor_pos)]
                        if circuit['word'][slot_index] == placed_letter['char']:
                            is_correct = True
                            break
                
                if is_correct:
                    reward += 1.0 # Reward for correct placement
                    self.score += 10
                    # sfx: place_correct
                else:
                    reward -= 1.0 # Penalty for incorrect placement
                    self.score -= 5
                    # sfx: place_wrong
        return reward

    def _update_circuits(self):
        reward = 0
        for circuit in self.circuits:
            if not circuit['completed']:
                is_now_complete = True
                for i, pos in enumerate(circuit['slots']):
                    letter_on_grid = self.grid[pos[1]][pos[0]]
                    if letter_on_grid is None or letter_on_grid['char'] != circuit['word'][i]:
                        is_now_complete = False
                        break
                
                if is_now_complete:
                    circuit['completed'] = True
                    reward += 10.0 # Big reward for completing a circuit
                    self.score += 100
                    # sfx: circuit_complete
                    self._create_circuit_complete_effect(circuit)
        return reward

    def _check_termination(self):
        if self.energy <= 0:
            self.win_message = "ENERGY DEPLETED"
            return True, -50.0
        
        if all(c['completed'] for c in self.circuits):
            self.win_message = "CITY POWER RESTORED!"
            return True, 100.0
        
        return False, 0.0

    def _generate_level(self):
        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.circuits = []
        self.letters = []
        
        num_circuits = min(3 + (self.level - 1), 5)
        available_words = self.WORD_LIST[:]
        self.np_random.shuffle(available_words)
        
        used_slots = set()
        
        for i in range(num_circuits):
            if not available_words: break
            
            word = available_words.pop(0)
            path = None
            for _ in range(50): # Try 50 times to find a non-overlapping path
                path = self._generate_path(len(word), used_slots)
                if path:
                    break
            
            if path:
                for pos in path:
                    used_slots.add(pos)
                
                slot_map = {pos: i for i, pos in enumerate(path)}
                self.circuits.append({'word': word, 'slots': path, 'slot_map': slot_map, 'completed': False})
                
                for char in word:
                    self.letters.append({'char': char, 'pos': [0, 0], 'pixel_pos': [0,0], 'magnetized': False})

        # Scatter letters
        empty_cells = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (c, r) not in used_slots:
                    empty_cells.append([c, r])
        
        self.np_random.shuffle(empty_cells)
        
        for letter in self.letters:
            if not empty_cells: break # Should not happen with proper grid size
            pos = empty_cells.pop()
            letter['pos'] = pos
            self.grid[pos[1]][pos[0]] = letter

    def _generate_path(self, length, used_slots):
        for _ in range(20): # Try 20 different start points
            path = []
            start_x = self.np_random.integers(0, self.GRID_COLS)
            start_y = self.np_random.integers(0, self.GRID_ROWS)
            pos = (start_x, start_y)
            
            if pos in used_slots: continue
            
            path.append(pos)
            current_pos = list(pos)
            
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for _ in range(length - 1):
                self.np_random.shuffle(moves)
                moved = False
                for dx, dy in moves:
                    next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                    if (0 <= next_pos[0] < self.GRID_COLS and
                        0 <= next_pos[1] < self.GRID_ROWS and
                        next_pos not in used_slots and
                        next_pos not in path):
                        path.append(next_pos)
                        current_pos = list(next_pos)
                        moved = True
                        break
                if not moved:
                    path = None # Failed to find a path
                    break
            if path: return path
        return None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_circuits()
        self._render_letters()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        power_level = 0
        if self.circuits:
            completed_circuits = sum(1 for c in self.circuits if c['completed'])
            power_level = (completed_circuits / len(self.circuits)) * 100
        
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "city_power": power_level,
        }

    # --- Rendering Methods ---

    def _render_background_effects(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_COLS * self.CELL_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_ROWS * self.CELL_SIZE))

    def _render_circuits(self):
        for circuit in self.circuits:
            color = self.COLOR_CIRCUIT_COMPLETE if circuit['completed'] else self.COLOR_CIRCUIT_INCOMPLETE
            glow_color = (*color, 50)

            # Draw connections
            for i in range(len(circuit['slots']) - 1):
                start_pixel = self._grid_to_pixel(circuit['slots'][i])
                end_pixel = self._grid_to_pixel(circuit['slots'][i+1])
                pygame.draw.line(self.screen, color, start_pixel, end_pixel, 2)

            # Draw slots
            for pos in circuit['slots']:
                pixel_pos = self._grid_to_pixel(pos)
                rect = pygame.Rect(pixel_pos[0] - self.CELL_SIZE//2 + 2, pixel_pos[1] - self.CELL_SIZE//2 + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 2, border_radius=3)
                # Glow effect for slots
                pygame.draw.rect(self.screen, glow_color, rect.inflate(4,4), 2, border_radius=5)

    def _render_letters(self):
        for letter in self.letters:
            if letter['magnetized']:
                continue # Rendered with cursor
            
            pixel_pos = self._grid_to_pixel(letter['pos'])
            self._draw_letter_component(pixel_pos, letter['char'], self.COLOR_LETTER_NORMAL)

    def _render_cursor(self):
        cursor_pixel_pos = self._grid_to_pixel(self.cursor_pos)
        
        # Draw cursor
        size = self.CELL_SIZE // 2
        points = [
            (cursor_pixel_pos[0] - size, cursor_pixel_pos[1]),
            (cursor_pixel_pos[0], cursor_pixel_pos[1] - size),
            (cursor_pixel_pos[0] + size, cursor_pixel_pos[1]),
            (cursor_pixel_pos[0], cursor_pixel_pos[1] + size),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CURSOR)
        pygame.gfxdraw.filled_polygon(self.screen, points, (*self.COLOR_CURSOR, 50))

        # Draw magnetized letter
        if self.magnetized_letter:
            self._draw_letter_component(self.magnetized_letter['pixel_pos'], self.magnetized_letter['char'], self.COLOR_LETTER_MAGNETIZED, is_magnetized=True)

    def _draw_letter_component(self, pixel_pos, char, color, is_magnetized=False):
        rect_size = self.CELL_SIZE - 8
        rect = pygame.Rect(pixel_pos[0] - rect_size//2, pixel_pos[1] - rect_size//2, rect_size, rect_size)
        
        if is_magnetized:
            # Pulsing glow effect
            glow_size = rect_size + 8 + 4 * math.sin(self.steps * 0.2)
            glow_rect = pygame.Rect(pixel_pos[0] - glow_size//2, pixel_pos[1] - glow_size//2, glow_size, glow_size)
            pygame.draw.rect(self.screen, (*color, 30), glow_rect, border_radius=8)
            pygame.draw.rect(self.screen, (*color, 60), glow_rect.inflate(-6, -6), border_radius=6)

        pygame.draw.rect(self.screen, color, rect, 2, border_radius=5)
        text_surf = self.font_medium.render(char, True, color)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

    def _render_particles(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

    def _render_ui(self):
        # Energy Bar
        bar_width, bar_height = 150, 20
        energy_ratio = max(0, self.energy / self.max_energy)
        fill_width = int(bar_width * energy_ratio)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BG, (10, 10, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, (10, 10, fill_width, bar_height), border_radius=4)
        
        # City Power
        info = self._get_info()
        power_text = f"CITY POWER: {info['city_power']:.0f}%"
        text_surf = self.font_small.render(power_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.screen_width - text_surf.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {self.score}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, self.screen_height - text_surf.get_height() - 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        color = self.COLOR_WIN if "RESTORED" in self.win_message else self.COLOR_LOSE
        text_surf = self.font_large.render(self.win_message, True, color)
        text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        self.screen.blit(text_surf, text_rect)

    # --- Helper & Effect Methods ---

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return (x, y)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['radius'] = max(0, p['radius'] * 0.95)
            if p['lifespan'] > 0 and p['max_lifespan'] > 0:
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                p['color'] = (*p['base_color'], alpha)

    def _create_teleport_effect(self, pos):
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            self.particles.append({
                'pos': list(pos), 'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.uniform(3, 7), 'lifespan': 20, 'max_lifespan': 20,
                'base_color': self.COLOR_LETTER_MAGNETIZED, 'color': (*self.COLOR_LETTER_MAGNETIZED, 255)
            })

    def _create_circuit_complete_effect(self, circuit):
        for i in range(len(circuit['slots'])):
            for _ in range(5):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0.5, 1.5)
                pixel_pos = self._grid_to_pixel(circuit['slots'][i])
                self.particles.append({
                    'pos': list(pixel_pos), 'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                    'radius': random.uniform(2, 4), 'lifespan': 30, 'max_lifespan': 30,
                    'base_color': self.COLOR_CIRCUIT_COMPLETE, 'color': (*self.COLOR_CIRCUIT_COMPLETE, 255)
                })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    real_screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Neon Circuit")
    
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Energy: {info['energy']}, Power: {info['city_power']:.0f}%")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print("Game Over!")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(30) # Limit to 30 FPS

    env.close()