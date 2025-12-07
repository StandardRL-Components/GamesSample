
# Generated: 2025-08-28T04:48:55.295602
# Source Brief: brief_02421.md
# Brief Index: 2421

        
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


class Crystal:
    """A simple class to hold crystal state."""
    def __init__(self, crystal_id, grid_pos):
        self.id = crystal_id
        self.grid_pos = grid_pos  # (row, col)
        self.orientation = 0  # 0-7
        self.is_lit = False
        self.is_source = False
        self.screen_pos = (0, 0) # To be calculated

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to select a crystal. Press Space to rotate it. "
        "The goal is to light up all crystals."
    )

    game_description = (
        "A mind-bending puzzle game. Rotate crystalline mirrors to redirect a beam of light "
        "and illuminate the entire cavern before you run out of moves."
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
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 50)

        # --- Game Constants ---
        self.GRID_ROWS = 7
        self.GRID_COLS = 11
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = 24
        self.MAX_MOVES = 50
        self.NUM_CRYSTALS = 10

        # --- Colors ---
        self.COLOR_BG = (15, 10, 30)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_UNLIT = (50, 80, 160)
        self.COLOR_UNLIT_BORDER = (100, 130, 210)
        self.COLOR_LIT = (180, 255, 255)
        self.COLOR_LIT_BORDER = (255, 255, 255)
        self.COLOR_SOURCE = (255, 255, 100)
        self.COLOR_SOURCE_BORDER = (255, 255, 255)
        self.COLOR_BEAM = (255, 255, 220)
        self.COLOR_SELECT = (100, 255, 100)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.remaining_moves = 0
        self.crystals = []
        self.crystals_by_row = {}
        self.row_keys = []
        self.selected_row_idx = 0
        self.selected_col_idx = 0
        self.source_crystal_id = -1
        self.beams = []
        self.total_crystals = 0
        self.lit_crystal_count = 0

        # Map directions (0-7, N, NE, E, ...) to grid offsets (dr, dc)
        self.DIR_VECTORS = {
            0: (-1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1),
            4: (1, 0), 5: (1, -1), 6: (0, -1), 7: (-1, -1)
        }

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.remaining_moves = self.MAX_MOVES
        
        self._generate_level()
        self._update_lights()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_press, _ = action
        reward = 0
        
        lit_before = self.lit_crystal_count

        action_taken = self._handle_input(movement, space_press == 1)

        if action_taken:
            self._update_lights()
            lit_after = self.lit_crystal_count
            
            # Reward for change in lit crystals
            reward += (lit_after - lit_before) * 5.0

            terminated = self._check_termination()
            if terminated:
                if self.lit_crystal_count == self.total_crystals:
                    reward += 100 # Win bonus
                    self.score += 100
                    self.win_message = "PUZZLE SOLVED!"
                else:
                    reward = -50 # Loss penalty
                    self.win_message = "OUT OF MOVES"
            self.score += reward
        else:
            # No move-consuming action was taken
            terminated = False

        self.steps += 1
        if self.steps >= 500: # Max episode length
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_press):
        """Process actions and update state. Returns True if a move was consumed."""
        # Movement action (change selection)
        if movement != 0:
            if not self.row_keys: return False
            
            current_row_key = self.row_keys[self.selected_row_idx]
            num_in_row = len(self.crystals_by_row[current_row_key])

            if movement == 1: # Up
                self.selected_row_idx = (self.selected_row_idx - 1 + len(self.row_keys)) % len(self.row_keys)
                self.selected_col_idx = 0
            elif movement == 2: # Down
                self.selected_row_idx = (self.selected_row_idx + 1) % len(self.row_keys)
                self.selected_col_idx = 0
            elif movement == 3: # Left
                self.selected_col_idx = (self.selected_col_idx - 1 + num_in_row) % num_in_row
            elif movement == 4: # Right
                self.selected_col_idx = (self.selected_col_idx + 1) % num_in_row
            # Sound placeholder: pygame.mixer.Sound.play(self.select_sound)
            return False # Selection does not consume a move

        # Rotation action
        if space_press:
            selected_crystal = self._get_selected_crystal()
            if selected_crystal and not selected_crystal.is_source:
                selected_crystal.orientation = (selected_crystal.orientation + 1) % 8
                self.remaining_moves -= 1
                # Sound placeholder: pygame.mixer.Sound.play(self.rotate_sound)
                return True # Rotation consumes a move
        
        return False

    def _generate_level(self):
        self.crystals = []
        self.crystals_by_row = {}
        
        # Generate potential positions in a diamond shape
        possible_positions = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (self.GRID_ROWS // 2 - r) * (1 if self.GRID_ROWS % 2 == 0 else -1) <= c - self.GRID_COLS // 2 <= (self.GRID_ROWS // 2 + r):
                     if abs(r - self.GRID_ROWS//2) + abs(c - self.GRID_COLS//2) < (self.GRID_ROWS + self.GRID_COLS)//4 + 1:
                        possible_positions.append((r,c))
        
        # Ensure we have enough positions
        if len(possible_positions) < self.NUM_CRYSTALS:
            raise ValueError("Not enough possible positions for the number of crystals")

        crystal_pos = self.np_random.choice(len(possible_positions), self.NUM_CRYSTALS, replace=False)
        crystal_grid_positions = [possible_positions[i] for i in crystal_pos]

        for i, pos in enumerate(crystal_grid_positions):
            crystal = Crystal(i, pos)
            crystal.orientation = self.np_random.integers(0, 8)
            self.crystals.append(crystal)

        # Select a source crystal (not on the edge)
        non_edge_crystals = [c for c in self.crystals if 0 < c.grid_pos[0] < self.GRID_ROWS -1 and 0 < c.grid_pos[1] < self.GRID_COLS -1]
        if not non_edge_crystals: non_edge_crystals = self.crystals # Fallback
        source_crystal = self.np_random.choice(non_edge_crystals)
        
        source_crystal.is_source = True
        source_crystal.is_lit = True
        self.source_crystal_id = source_crystal.id

        # Organize for selection
        for c in self.crystals:
            r, col = c.grid_pos
            c.screen_pos = self._iso_to_screen(r, col)
            if r not in self.crystals_by_row:
                self.crystals_by_row[r] = []
            self.crystals_by_row[r].append(c)
        
        for r in self.crystals_by_row:
            self.crystals_by_row[r].sort(key=lambda c: c.grid_pos[1])
        
        self.row_keys = sorted(self.crystals_by_row.keys())
        self.selected_row_idx = 0
        self.selected_col_idx = 0
        self.total_crystals = len(self.crystals)

    def _get_reflections(self, crystal, in_dir):
        """Calculates outgoing directions for a given crystal and incoming light direction."""
        o = crystal.orientation
        # Type 0, 4: '\' mirror
        if o in [0, 4]:
            if in_dir == 0: return [6] # N -> W
            if in_dir == 6: return [0]
            if in_dir == 4: return [2] # S -> E
            if in_dir == 2: return [4]
        # Type 1, 5: '|' pass-through
        elif o in [1, 5]:
            if in_dir == 0: return [4] # N -> S
            if in_dir == 4: return [0]
        # Type 2, 6: '/' mirror
        elif o in [2, 6]:
            if in_dir == 0: return [2] # N -> E
            if in_dir == 2: return [0]
            if in_dir == 4: return [6] # S -> W
            if in_dir == 6: return [4]
        # Type 3, 7: '-' pass-through
        elif o in [3, 7]:
            if in_dir == 2: return [6] # E -> W
            if in_dir == 6: return [2]
        return [] # Absorbed

    def _update_lights(self):
        """The core light propagation algorithm."""
        if not self.crystals: return

        for c in self.crystals:
            c.is_lit = False
        
        source_crystal = self.crystals[self.source_crystal_id]
        source_crystal.is_lit = True
        
        lit_crystal_ids = {source_crystal.id}
        
        # (start_crystal_id, direction)
        queue = [(source_crystal.id, d) for d in range(8)]
        
        # (start_crystal_id, end_crystal_id, direction)
        visited_beams = set()
        
        self.beams = []
        
        crystal_map = {c.grid_pos: c for c in self.crystals}

        while queue:
            start_id, direction = queue.pop(0)
            
            start_crystal = self.crystals[start_id]
            r, c = start_crystal.grid_pos
            dr, dc = self.DIR_VECTORS[direction]

            path_ended = False
            for i in range(1, max(self.GRID_ROWS, self.GRID_COLS)):
                nr, nc = r + i * dr, c + i * dc

                # Check for beam termination
                if not (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS):
                    self.beams.append((start_crystal.screen_pos, self._iso_to_screen(nr, nc)))
                    path_ended = True
                    break

                if (nr, nc) in crystal_map:
                    hit_crystal = crystal_map[(nr, nc)]
                    
                    beam_key = tuple(sorted((start_id, hit_crystal.id))) + (direction,)
                    if beam_key in visited_beams:
                        path_ended = True
                        break
                    visited_beams.add(beam_key)
                    
                    self.beams.append((start_crystal.screen_pos, hit_crystal.screen_pos))
                    
                    in_dir = (direction + 4) % 8
                    out_dirs = self._get_reflections(hit_crystal, in_dir)
                    
                    if out_dirs:
                        if hit_crystal.id not in lit_crystal_ids:
                           lit_crystal_ids.add(hit_crystal.id)
                           # Sound placeholder: pygame.mixer.Sound.play(self.crystal_lit_sound)
                        
                        for out_dir in out_dirs:
                            queue.append((hit_crystal.id, out_dir))
                            
                    path_ended = True
                    break
            
            if not path_ended: # Beam went off-screen without hitting anything
                nr, nc = r + max(self.GRID_ROWS, self.GRID_COLS) * dr, c + max(self.GRID_ROWS, self.GRID_COLS) * dc
                self.beams.append((start_crystal.screen_pos, self._iso_to_screen(nr, nc)))

        for c in self.crystals:
            if c.id in lit_crystal_ids:
                c.is_lit = True
        
        self.lit_crystal_count = len(lit_crystal_ids)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_beams()
        
        # Draw selection marker first so it's underneath
        selected_crystal = self._get_selected_crystal()
        if selected_crystal:
            pos = selected_crystal.screen_pos
            radius = int(self.TILE_WIDTH * 0.6)
            for i in range(5):
                alpha = 150 - i * 30
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius + i, (*self.COLOR_SELECT, alpha))

        for crystal in sorted(self.crystals, key=lambda c: c.screen_pos[1]):
            self._draw_crystal(crystal)

    def _draw_beams(self):
        for start_pos, end_pos in self.beams:
            # Glow effect
            pygame.draw.line(self.screen, (*self.COLOR_BEAM, 50), start_pos, end_pos, 7)
            pygame.draw.line(self.screen, (*self.COLOR_BEAM, 100), start_pos, end_pos, 5)
            # Main beam
            pygame.draw.aaline(self.screen, self.COLOR_BEAM, start_pos, end_pos)

    def _draw_crystal(self, crystal):
        pos = crystal.screen_pos
        radius = int(self.TILE_WIDTH / 2.5)
        
        is_source = crystal.is_source
        is_lit = crystal.is_lit

        if is_source:
            color = self.COLOR_SOURCE
            border_color = self.COLOR_SOURCE_BORDER
        elif is_lit:
            color = self.COLOR_LIT
            border_color = self.COLOR_LIT_BORDER
        else:
            color = self.COLOR_UNLIT
            border_color = self.COLOR_UNLIT_BORDER

        # Glow for lit crystals
        if is_lit or is_source:
            for i in range(10):
                alpha = 100 - i * 10
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius + i, (*color, alpha))
        
        # Crystal shape (octagon)
        points = []
        for i in range(8):
            angle = math.pi / 4 * i + (math.pi / 8) # Add offset for flat top
            points.append((pos[0] + radius * math.cos(angle), pos[1] + radius * math.sin(angle)))
        
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, border_color)

        # Orientation marker
        o = crystal.orientation
        marker_len = radius * 0.7
        
        # Different markers for different mirror types
        shape_type = o % 4
        if shape_type == 0: # '\'
            p1 = (pos[0] - marker_len/2, pos[1] - marker_len/2)
            p2 = (pos[0] + marker_len/2, pos[1] + marker_len/2)
        elif shape_type == 1: # '|'
            p1 = (pos[0], pos[1] - marker_len/2)
            p2 = (pos[0], pos[1] + marker_len/2)
        elif shape_type == 2: # '/'
            p1 = (pos[0] - marker_len/2, pos[1] + marker_len/2)
            p2 = (pos[0] + marker_len/2, pos[1] - marker_len/2)
        else: # '-'
            p1 = (pos[0] - marker_len/2, pos[1])
            p2 = (pos[0] + marker_len/2, pos[1])
            
        pygame.draw.aaline(self.screen, border_color, p1, p2, 2)


    def _render_ui(self):
        # Moves left
        moves_text = f"MOVES: {self.remaining_moves}"
        text_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Lit count
        lit_text = f"LIT: {self.lit_crystal_count} / {self.total_crystals}"
        text_surf = self.font_ui.render(lit_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.screen_size[0] - text_surf.get_width() - 10, 10))

        # Game over message
        if self.game_over:
            msg_surf = self.font_msg.render(self.win_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.screen_size[0] / 2, self.screen_size[1] / 2))
            
            # Add a dark background for readability
            bg_rect = msg_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.remaining_moves,
            "lit_crystals": self.lit_crystal_count,
            "total_crystals": self.total_crystals,
        }

    def _check_termination(self):
        win = self.lit_crystal_count == self.total_crystals
        lose = self.remaining_moves <= 0
        self.game_over = win or lose
        return self.game_over
    
    def _get_selected_crystal(self):
        if not self.row_keys: return None
        row_key = self.row_keys[self.selected_row_idx]
        return self.crystals_by_row[row_key][self.selected_col_idx]

    def _iso_to_screen(self, r, c):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = (self.screen_size[0] / 2) + (c - r) * (self.TILE_WIDTH / 2)
        screen_y = (self.screen_size[1] / 4) + (c + r) * (self.TILE_HEIGHT / 2)
        return screen_x, screen_y

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode(env.screen_size)
    pygame.display.set_caption(env.game_description)

    last_action = [0, 0, 0]
    
    print(env.user_guide)

    while running:
        action = [0, 0, 0] # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space for manual play
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

        # Only step if an action is taken (for turn-based manual play)
        if action != [0,0,0]:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
            if terminated:
                print("Game Over!")
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        game_window.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(15) # Limit frame rate for manual play responsiveness
        
        if env.game_over:
            pygame.time.wait(3000)
            obs, info = env.reset()


    env.close()