
# Generated: 2025-08-28T04:40:09.510991
# Source Brief: brief_02387.md
# Brief Index: 2387

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. "
        "Press Space to interact with a puzzle element. "
        "Hold Shift to request a hint after several failed attempts."
    )

    game_description = (
        "Escape a cursed manor by solving a series of five visual puzzles before you run out of turns. "
        "Each interaction costs one turn. Think carefully!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # --- Colors and Fonts ---
        self.COLOR_BG = (26, 26, 46)  # Dark purple-blue
        self.COLOR_FLOOR = (45, 45, 65)
        self.COLOR_WALL = (35, 35, 55)
        self.COLOR_INTERACTIVE = (233, 69, 96)  # Bright red
        self.COLOR_CURSOR = (255, 220, 0) # Gold
        self.COLOR_SOLVED = (0, 255, 190)  # Bright cyan
        self.COLOR_TEXT = (240, 240, 255)
        self.COLOR_HINT = (0, 150, 255) # Bright blue

        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.time_remaining = 0
        self.current_room = 0
        self.puzzles_solved = []
        self.cursor_pos = [0, 0]
        self.puzzles = []
        self.failed_attempts = 0
        self.hint_active = False
        self.hint_cooldown = 0
        self.last_interaction_feedback = 0 # for visual effect
        self.last_solve_time = 0

        # --- Puzzle Definitions ---
        self._puzzle_defs = self._create_puzzle_definitions()

        # Initialize state
        self.reset()
        
        self.validate_implementation()

    def _create_puzzle_definitions(self):
        return [
            { # Puzzle 0: Lights Out (3x3)
                'size': (3, 3),
                'init': self._init_lights_out,
                'interact': self._interact_lights_out,
                'distance': lambda state, sol: np.sum(sol - state),
                'draw': self._draw_lights_out,
                'hint': self._get_best_move_hint
            },
            { # Puzzle 1: Pattern Match (3x3)
                'size': (3, 3),
                'init': self._init_pattern_match,
                'interact': self._interact_pattern_match,
                'distance': lambda state, sol: np.sum(state != sol),
                'draw': self._draw_pattern_match,
                'hint': self._get_best_move_hint
            },
            { # Puzzle 2: Rotation Puzzle (4x4)
                'size': (4, 4),
                'init': self._init_rotation_puzzle,
                'interact': self._interact_rotation_puzzle,
                'distance': lambda state, sol: np.sum(state != sol),
                'draw': self._draw_rotation_puzzle,
                'hint': self._get_best_move_hint
            },
            { # Puzzle 3: Sliding Puzzle (3x3)
                'size': (3, 3),
                'init': self._init_sliding_puzzle,
                'interact': self._interact_sliding_puzzle,
                'distance': self._distance_sliding_puzzle,
                'draw': self._draw_sliding_puzzle,
                'hint': self._get_best_move_hint
            },
            { # Puzzle 4: Memory Grid (4x4)
                'size': (4, 4),
                'init': self._init_memory_puzzle,
                'interact': self._interact_memory_puzzle,
                'distance': lambda state, sol: np.sum(np.abs(state - sol)),
                'draw': self._draw_memory_puzzle,
                'hint': self._get_best_move_hint
            }
        ]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.time_remaining = 90
        self.current_room = 0
        self.puzzles_solved = [False] * 5
        self.failed_attempts = 0
        self.hint_active = False
        self.hint_cooldown = 0
        self.last_interaction_feedback = 0
        self.last_solve_time = 0

        self.puzzles = []
        for i in range(5):
            p_def = self._puzzle_defs[i]
            state, solution = p_def['init'](p_def['size'])
            self.puzzles.append({'state': state, 'solution': solution, 'type': i})
        
        self.cursor_pos = [0, 0]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        turn_taken = False
        
        # --- Handle Cursor Movement ---
        puzzle_size = self._puzzle_defs[self.current_room]['size']
        if movement != 0:
            # sfx: cursor_move.wav
            if movement == 1: self.cursor_pos[1] -= 1
            elif movement == 2: self.cursor_pos[1] += 1
            elif movement == 3: self.cursor_pos[0] -= 1
            elif movement == 4: self.cursor_pos[0] += 1
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, puzzle_size[0] - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, puzzle_size[1] - 1)

        # --- Handle Interactions (These consume a turn) ---
        if space_held or shift_held:
            turn_taken = True
            self.time_remaining -= 1
            self.hint_active = False
            self.hint_cooldown = max(0, self.hint_cooldown - 1)

        if space_held:
            # sfx: interact.wav
            puzzle = self.puzzles[self.current_room]
            p_def = self._puzzle_defs[puzzle['type']]
            
            old_dist = p_def['distance'](puzzle['state'], puzzle['solution'])
            
            p_def['interact'](puzzle, tuple(self.cursor_pos))
            
            new_dist = p_def['distance'](puzzle['state'], puzzle['solution'])

            # Reward shaping
            if new_dist < old_dist:
                reward += 0.1
                self.last_interaction_feedback = 1
            elif new_dist > old_dist:
                reward -= 0.1
                self.last_interaction_feedback = -1
            else:
                self.last_interaction_feedback = 0

            # Check for puzzle solved
            if new_dist == 0:
                # sfx: puzzle_solved.wav
                reward += 10
                self.score += 10
                self.puzzles_solved[self.current_room] = True
                self.failed_attempts = 0
                self.last_solve_time = self.steps
                if self.current_room < 4:
                    self.current_room += 1
                    self.cursor_pos = [0, 0]
                else: # All puzzles solved
                    self.game_won = True
            else:
                self.failed_attempts += 1
        
        elif shift_held:
            if self.failed_attempts >= 5 and self.hint_cooldown == 0:
                # sfx: hint_activate.wav
                self.hint_active = True
                self.hint_cooldown = 3 # Can't spam hints
                self.failed_attempts = 0 # Reset attempts after hint
                reward -= 0.5 # Small penalty for using a hint
            else:
                # sfx: hint_fail.wav
                pass # No effect if conditions not met

        self.steps += 1
        terminated = False
        
        if self.game_won:
            # sfx: game_win.wav
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0:
            # sfx: game_lose.wav
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= 1000:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_room()
        self._render_current_puzzle()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_remaining": self.time_remaining}

    # =========== PUZZLE IMPLEMENTATIONS ===========

    def _init_lights_out(self, size):
        state = np.ones(size, dtype=int)
        solution = np.ones(size, dtype=int)
        for _ in range(self.np_random.integers(5, 10)):
            x, y = self.np_random.integers(0, size[0]), self.np_random.integers(0, size[1])
            self._flip_tile(state, x, y)
        if np.all(state == solution): # ensure it's not already solved
            self._flip_tile(state, 0, 0)
        return state, solution

    def _interact_lights_out(self, puzzle, pos):
        self._flip_tile(puzzle['state'], pos[0], pos[1])

    def _flip_tile(self, state, x, y):
        size = state.shape
        for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size[0] and 0 <= ny < size[1]:
                state[nx, ny] = 1 - state[nx, ny]
    
    def _draw_lights_out(self, surface, puzzle, center, tile_size):
        state = puzzle['state']
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                iso_x, iso_y = self._iso_transform(x, y, center, tile_size)
                color = self.COLOR_SOLVED if state[x, y] == 1 else self.COLOR_WALL
                self._draw_iso_rect(surface, (iso_x, iso_y), tile_size, color)

    def _init_pattern_match(self, size):
        solution = self.np_random.integers(0, 4, size=size)
        state = np.copy(solution)
        for _ in range(self.np_random.integers(4, 7)):
            x, y = self.np_random.integers(0, size[0]), self.np_random.integers(0, size[1])
            state[x, y] = (state[x, y] + self.np_random.integers(1, 4)) % 4
        return state, solution

    def _interact_pattern_match(self, puzzle, pos):
        puzzle['state'][pos] = (puzzle['state'][pos] + 1) % 4

    def _draw_pattern_match(self, surface, puzzle, center, tile_size):
        state, solution = puzzle['state'], puzzle['solution']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                iso_x, iso_y = self._iso_transform(x, y, center, tile_size)
                self._draw_iso_rect(surface, (iso_x, iso_y), tile_size, colors[state[x,y]])
        # Draw solution hint
        sol_center = (center[0] + 200, center[1])
        self._draw_text("Target", self.font_small, sol_center[0], sol_center[1] - 50, surface)
        for x in range(solution.shape[0]):
            for y in range(solution.shape[1]):
                iso_x, iso_y = self._iso_transform(x, y, sol_center, tile_size//2)
                self._draw_iso_rect(surface, (iso_x, iso_y), tile_size//2, colors[solution[x,y]])

    def _init_rotation_puzzle(self, size):
        solution = np.zeros(size, dtype=int)
        state = self.np_random.integers(0, 4, size=size)
        return state, solution

    def _interact_rotation_puzzle(self, puzzle, pos):
        puzzle['state'][pos] = (puzzle['state'][pos] + 1) % 4

    def _draw_rotation_puzzle(self, surface, puzzle, center, tile_size):
        state = puzzle['state']
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                iso_x, iso_y = self._iso_transform(x, y, center, tile_size)
                self._draw_iso_rect(surface, (iso_x, iso_y), tile_size, self.COLOR_WALL)
                angle = state[x, y] * math.pi / 2
                p1 = (iso_x, iso_y)
                p2 = (iso_x + math.cos(angle) * tile_size * 0.4, iso_y - math.sin(angle) * tile_size * 0.4)
                pygame.draw.line(surface, self.COLOR_INTERACTIVE, p1, p2, 3)
                pygame.gfxdraw.filled_circle(surface, int(p2[0]), int(p2[1]), 4, self.COLOR_INTERACTIVE)

    def _init_sliding_puzzle(self, size):
        solution_state = np.arange(1, size[0]*size[1]+1).reshape(size)
        solution_state[-1, -1] = 0
        state = np.copy(solution_state)
        for _ in range(100):
            blank_pos = np.argwhere(state == 0)[0]
            moves = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = blank_pos[0] + dx, blank_pos[1] + dy
                if 0 <= nx < size[0] and 0 <= ny < size[1]:
                    moves.append((nx, ny))
            swap_pos = random.choice(moves)
            state[blank_pos[0], blank_pos[1]], state[swap_pos[0], swap_pos[1]] = \
                state[swap_pos[0], swap_pos[1]], state[blank_pos[0], blank_pos[1]]
        return state, solution_state

    def _interact_sliding_puzzle(self, puzzle, pos):
        state = puzzle['state']
        blank_pos = np.argwhere(state == 0)[0]
        if abs(pos[0] - blank_pos[0]) + abs(pos[1] - blank_pos[1]) == 1:
            state[blank_pos[0], blank_pos[1]], state[pos[0], pos[1]] = \
                state[pos[0], pos[1]], state[blank_pos[0], blank_pos[1]]

    def _distance_sliding_puzzle(self, state, solution):
        dist = 0
        for i in range(1, state.shape[0] * state.shape[1]):
            pos_state = np.argwhere(state == i)[0]
            pos_sol = np.argwhere(solution == i)[0]
            dist += abs(pos_state[0] - pos_sol[0]) + abs(pos_state[1] - pos_sol[1])
        return dist
        
    def _draw_sliding_puzzle(self, surface, puzzle, center, tile_size):
        state = puzzle['state']
        font = pygame.font.Font(None, tile_size)
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                if state[x, y] == 0: continue
                iso_x, iso_y = self._iso_transform(x, y, center, tile_size)
                self._draw_iso_rect(surface, (iso_x, iso_y), tile_size, self.COLOR_INTERACTIVE)
                text = font.render(str(state[x, y]), True, self.COLOR_BG)
                text_rect = text.get_rect(center=(iso_x, iso_y))
                surface.blit(text, text_rect)

    def _init_memory_puzzle(self, size):
        solution = np.zeros(size, dtype=int)
        num_to_light = self.np_random.integers(4, 7)
        for _ in range(num_to_light):
            while True:
                x, y = self.np_random.integers(0, size[0]), self.np_random.integers(0, size[1])
                if solution[x, y] == 0:
                    solution[x, y] = 1
                    break
        state = np.zeros(size, dtype=int) # Player starts with a blank slate
        return state, solution

    def _interact_memory_puzzle(self, puzzle, pos):
        puzzle['state'][pos] = 1 - puzzle['state'][pos]

    def _draw_memory_puzzle(self, surface, puzzle, center, tile_size):
        state, solution = puzzle['state'], puzzle['solution']
        is_showing_solution = (self.steps - self.last_solve_time) < 90 and self.current_room == 4

        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                iso_x, iso_y = self._iso_transform(x, y, center, tile_size)
                is_on = state[x, y] == 1
                is_solution_tile = solution[x, y] == 1
                
                if is_showing_solution:
                    color = self.COLOR_SOLVED if is_solution_tile else self.COLOR_WALL
                else:
                    color = self.COLOR_INTERACTIVE if is_on else self.COLOR_WALL
                
                self._draw_iso_rect(surface, (iso_x, iso_y), tile_size, color)
        if is_showing_solution:
            self._draw_text("MEMORIZE", self.font_main, center[0], center[1]-120, surface)

    def _get_best_move_hint(self, puzzle, p_def):
        best_move = None
        current_dist = p_def['distance'](puzzle['state'], puzzle['solution'])
        min_dist = current_dist

        temp_puzzle = {'state': np.copy(puzzle['state']), 'solution': puzzle['solution']}
        size = puzzle['state'].shape

        for x in range(size[0]):
            for y in range(size[1]):
                p_def['interact'](temp_puzzle, (x, y))
                new_dist = p_def['distance'](temp_puzzle['state'], temp_puzzle['solution'])
                if new_dist < min_dist:
                    min_dist = new_dist
                    best_move = (x, y)
                temp_puzzle['state'] = np.copy(puzzle['state']) # Reset for next try
        
        return best_move

    # =========== RENDERING HELPERS ===========

    def _iso_transform(self, x, y, center, tile_size):
        iso_x = center[0] + (x - y) * (tile_size * 0.866)
        iso_y = center[1] + (x + y) * (tile_size * 0.5) - 50
        return iso_x, iso_y

    def _draw_iso_rect(self, surface, pos, size, color):
        points = [
            (pos[0], pos[1] - size * 0.5),
            (pos[0] + size * 0.866, pos[1]),
            (pos[0], pos[1] + size * 0.5),
            (pos[0] - size * 0.866, pos[1])
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_room(self):
        # Draw floor
        floor_points = [
            (self.SCREEN_WIDTH / 2, 50),
            (self.SCREEN_WIDTH - 50, self.SCREEN_HEIGHT / 2),
            (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50),
            (50, self.SCREEN_HEIGHT / 2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, floor_points, self.COLOR_FLOOR)
        pygame.gfxdraw.aapolygon(self.screen, floor_points, self.COLOR_WALL)
        
        # Draw door
        door_color = self.COLOR_SOLVED if self.puzzles_solved[self.current_room] else self.COLOR_WALL
        door_rect = pygame.Rect(self.SCREEN_WIDTH/2 - 40, 60, 80, 100)
        pygame.draw.rect(self.screen, self.COLOR_BG, door_rect)
        pygame.draw.rect(self.screen, door_color, door_rect, 3)
        pygame.gfxdraw.filled_circle(self.screen, int(door_rect.right - 15), int(door_rect.centery), 5, door_color)


    def _render_current_puzzle(self):
        if self.game_over: return

        puzzle = self.puzzles[self.current_room]
        p_def = self._puzzle_defs[puzzle['type']]
        size = p_def['size']
        tile_size = 30 if max(size) > 3 else 40
        center = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30)

        # Draw puzzle elements
        if self.puzzles_solved[self.current_room]:
            # Flash effect on solve
            if (self.steps - self.last_solve_time) < 30:
                alpha = 255 * (1 - (self.steps - self.last_solve_time)/30)
                s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                s.fill((self.COLOR_SOLVED[0], self.COLOR_SOLVED[1], self.COLOR_SOLVED[2], alpha))
                self.screen.blit(s, (0,0))
            self._draw_text("SOLVED", self.font_main, center[0], center[1], self.screen)
        else:
            p_def['draw'](self.screen, puzzle, center, tile_size)
            
            # Draw cursor
            cursor_iso_x, cursor_iso_y = self._iso_transform(self.cursor_pos[0], self.cursor_pos[1], center, tile_size)
            pulse = abs(math.sin(self.steps * 0.2))
            cursor_size = tile_size * (1.1 + 0.1 * pulse)
            points = [
                (cursor_iso_x, cursor_iso_y - cursor_size * 0.5),
                (cursor_iso_x + cursor_size * 0.866, cursor_iso_y),
                (cursor_iso_x, cursor_iso_y + cursor_size * 0.5),
                (cursor_iso_x - cursor_size * 0.866, cursor_iso_y)
            ]
            pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, points, 2)

            # Draw hint
            if self.hint_active:
                hint_pos = p_def['hint'](puzzle, p_def)
                if hint_pos:
                    hint_iso_x, hint_iso_y = self._iso_transform(hint_pos[0], hint_pos[1], center, tile_size)
                    pygame.gfxdraw.filled_circle(self.screen, int(hint_iso_x), int(hint_iso_y), int(10 + 5*pulse), self.COLOR_HINT)
                    pygame.gfxdraw.aacircle(self.screen, int(hint_iso_x), int(hint_iso_y), int(10 + 5*pulse), self.COLOR_HINT)

    def _render_ui(self):
        # Timer
        time_color = self.COLOR_INTERACTIVE if self.time_remaining <= 15 else self.COLOR_TEXT
        self._draw_text(f"Turns: {self.time_remaining}", self.font_main, 100, 30, self.screen, color=time_color)
        
        # Score
        self._draw_text(f"Score: {self.score}", self.font_main, self.SCREEN_WIDTH - 100, 30, self.screen)
        
        # Puzzle indicators
        for i in range(5):
            x = self.SCREEN_WIDTH / 2 - 60 + i * 30
            y = self.SCREEN_HEIGHT - 25
            color = self.COLOR_SOLVED if self.puzzles_solved[i] else self.COLOR_WALL
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 10, color)
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 10, color)

        # Game Over/Win text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU ESCAPED!" if self.game_won else "TIME'S UP"
            color = self.COLOR_SOLVED if self.game_won else self.COLOR_INTERACTIVE
            self._draw_text(msg, self.font_large, self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2, self.screen, color=color)

    def _draw_text(self, text, font, x, y, surface, color=None, center=True):
        if color is None: color = self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        surface.blit(text_surface, text_rect)
        
    def validate_implementation(self):
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a dummy screen for display if not running headless
    pygame.display.set_caption("Cursed Manor")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print(f"GAME: Cursed Manor")
    print(f"DESCRIPTION: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")

    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # In a turn-based game, we only step when an action is taken
        # This simple human player loop sends an action every frame
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Time: {info['time_remaining']}")

        # Display the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Limit frame rate for human playability
        
    print("\nGAME OVER")
    print(f"Final Score: {info['score']}")
    
    # Keep the window open for a few seconds to show the final screen
    pygame.time.wait(3000)
    env.close()