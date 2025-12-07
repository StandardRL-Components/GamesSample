
# Generated: 2025-08-28T00:59:05.141694
# Source Brief: brief_03966.md
# Brief Index: 3966

        
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
        "Controls: Arrows to move cursor. Space to pick up/drop a crystal. Shift to rotate a crystal. "
        "Dropping or rotating a crystal costs 1 move."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Illuminate all the green exit nodes by moving and rotating reflective crystals. "
        "You have a limited number of moves and time!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 65)
    COLOR_WALL = (70, 80, 110)
    COLOR_WALL_TOP = (90, 100, 130)
    COLOR_SOURCE = (255, 255, 180)
    COLOR_EXIT_INACTIVE = (200, 50, 80)
    COLOR_EXIT_ACTIVE = (50, 220, 120)
    COLOR_BEAM = (255, 255, 200)
    COLOR_BEAM_GLOW = (255, 255, 200, 40)
    COLOR_CURSOR = (255, 255, 255)
    CRYSTAL_COLORS = [
        (100, 200, 255), (255, 100, 200), (200, 255, 100),
        (255, 150, 50), (150, 100, 255)
    ]

    # Grid & Tile Dimensions
    GRID_WIDTH = 20
    GRID_HEIGHT = 14
    TILE_W = 40
    TILE_H = 20
    TILE_Z = 12 # Height of walls/blocks

    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Game Parameters
    MAX_MOVES = 20
    MAX_TIME = 60.0 # seconds
    FPS = 30
    
    # Grid Content IDs
    EMPTY, WALL, SOURCE, EXIT = 0, 1, 2, 3
    
    # Directions for light path
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    DIR_VECTORS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    # Crystal orientation -> incoming direction -> outgoing direction
    REFRACTION_MAP = {
        0: {UP: RIGHT, RIGHT: UP, DOWN: LEFT, LEFT: DOWN}, # '/'
        1: {UP: LEFT, RIGHT: DOWN, DOWN: RIGHT, LEFT: UP}  # '\'
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 60)
        
        self.grid_offset = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - self.GRID_HEIGHT * self.TILE_H / 2 + 30)

        # Initialize state variables
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.crystals = []
        self.exits = []
        self.source_pos = (0, 0)
        self.light_beams = []
        self.illuminated_exits = set()
        
        self.cursor_pos = (0, 0)
        self.held_crystal_idx = None
        self.held_crystal_original_pos = None

        self.was_space_held = False
        self.was_shift_held = False

        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.time_left = 0.0
        self.game_over = False
        self.win = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = self.MAX_MOVES
        self.time_left = self.MAX_TIME
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.held_crystal_idx = None
        self.was_space_held = False
        self.was_shift_held = False
        
        self._generate_level()
        self._calculate_light_paths()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.grid.fill(self.EMPTY)
        # Walls
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL
        
        # Source
        self.source_pos = (1, self.GRID_HEIGHT // 2)
        self.grid[self.source_pos] = self.SOURCE
        
        # Exits
        self.exits = []
        num_exits = self.np_random.integers(2, 4)
        exit_y_positions = self.np_random.choice(range(2, self.GRID_HEIGHT - 2), num_exits, replace=False)
        for y in exit_y_positions:
            pos = (self.GRID_WIDTH - 2, y)
            self.grid[pos] = self.EXIT
            self.exits.append(pos)
            
        # Crystals
        self.crystals = []
        num_crystals = self.np_random.integers(3, 6)
        for i in range(num_crystals):
            while True:
                pos = (self.np_random.integers(2, self.GRID_WIDTH - 2), self.np_random.integers(2, self.GRID_HEIGHT - 2))
                if self.grid[pos] == self.EMPTY and not any(c['pos'] == pos for c in self.crystals):
                    break
            self.crystals.append({
                'pos': pos,
                'orientation': self.np_random.integers(0, 2),
                'color': self.CRYSTAL_COLORS[i % len(self.CRYSTAL_COLORS)]
            })

    def step(self, action):
        reward = 0
        self.time_left -= 1.0 / self.FPS
        self.steps += 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = self._process_input(action)
        action_taken = False
        
        # 1. Handle cursor movement
        if movement != 0:
            dx, dy = self.DIR_VECTORS[movement-1] if movement in [1,3] else self.DIR_VECTORS[movement-1] # Remap Up/Down
            if movement == 1: dx, dy = 0, -1 # Up
            if movement == 2: dx, dy = 0, 1 # Down
            if movement == 3: dx, dy = -1, 0 # Left
            if movement == 4: dx, dy = 1, 0 # Right
            
            new_x = max(0, min(self.GRID_WIDTH - 1, self.cursor_pos[0] + dx))
            new_y = max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + dy))
            self.cursor_pos = (new_x, new_y)

        # 2. Handle actions (Space/Shift)
        crystal_at_cursor_idx = self._get_crystal_at(self.cursor_pos)

        # Rotate action
        if shift_pressed and self.held_crystal_idx is None and crystal_at_cursor_idx is not None:
            # sfx: rotate_crystal.wav
            self.crystals[crystal_at_cursor_idx]['orientation'] = 1 - self.crystals[crystal_at_cursor_idx]['orientation']
            action_taken = True
        
        # Pickup/Drop action
        elif space_pressed:
            if self.held_crystal_idx is not None: # Try to drop
                if self.grid[self.cursor_pos] == self.EMPTY and self._get_crystal_at(self.cursor_pos) is None:
                    # sfx: drop_crystal.wav
                    self.crystals[self.held_crystal_idx]['pos'] = self.cursor_pos
                    self.held_crystal_idx = None
                    action_taken = True
                else:
                    # sfx: action_fail.wav
                    pass # Invalid drop location
            elif crystal_at_cursor_idx is not None: # Try to pick up
                # sfx: pickup_crystal.wav
                self.held_crystal_idx = crystal_at_cursor_idx
                self.held_crystal_original_pos = self.crystals[crystal_at_cursor_idx]['pos']

        # 3. Update game state if a move was made
        if action_taken:
            self.moves_left -= 1
            prev_illuminated = self.illuminated_exits.copy()
            
            self._calculate_light_paths()
            
            newly_lit = self.illuminated_exits - prev_illuminated
            newly_unlit = prev_illuminated - self.illuminated_exits
            
            reward += len(newly_lit) * 5.0
            reward += len(newly_unlit) * -1.0
            
            self.score += reward

        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100
                self.score += 100
            else:
                reward -= 100
                self.score -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        space_pressed = space_held and not self.was_space_held
        shift_pressed = shift_held and not self.was_shift_held
        
        self.was_space_held = space_held
        self.was_shift_held = shift_held
        
        return movement, space_pressed, shift_pressed

    def _check_termination(self):
        if self.game_over:
            return True
        
        if len(self.illuminated_exits) == len(self.exits):
            self.game_over = True
            self.win = True
            return True
            
        if self.moves_left <= 0 or self.time_left <= 0:
            self.game_over = True
            self.win = False
            return True
            
        return False

    def _calculate_light_paths(self):
        self.light_beams = []
        self.illuminated_exits = set()
        # Initial beams from source
        for direction in [self.UP, self.RIGHT, self.DOWN, self.LEFT]:
             self._trace_beam(self.source_pos, direction, 0)
    
    def _trace_beam(self, start_pos, direction, depth):
        if depth > self.GRID_WIDTH * self.GRID_HEIGHT: return # Prevent infinite loops

        pos = start_pos
        path_segment = [self._grid_to_world(pos)]
        
        while True:
            dx, dy = self.DIR_VECTORS[direction]
            next_pos = (pos[0] + dx, pos[1] + dy)

            if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                break # Out of bounds

            crystal_idx = self._get_crystal_at(next_pos)
            
            if self.grid[next_pos] == self.WALL:
                break # Hit a wall
            
            elif self.grid[next_pos] == self.EXIT:
                self.illuminated_exits.add(next_pos)
                pos = next_pos
            
            elif crystal_idx is not None and (self.held_crystal_idx is None or crystal_idx != self.held_crystal_idx):
                path_segment.append(self._grid_to_world(next_pos))
                self.light_beams.append(path_segment)
                
                orientation = self.crystals[crystal_idx]['orientation']
                new_direction = self.REFRACTION_MAP[orientation][direction]
                
                # sfx: light_reflect.wav
                self._trace_beam(next_pos, new_direction, depth + 1)
                return # Stop current beam, new one is traced
            
            else: # Empty space
                pos = next_pos
        
        path_segment.append(self._grid_to_world(pos))
        self.light_beams.append(path_segment)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left, "time_left": self.time_left}

    def _get_crystal_at(self, pos):
        for i, crystal in enumerate(self.crystals):
            if crystal['pos'] == pos:
                return i
        return None

    def _grid_to_world(self, pos):
        x, y = pos
        screen_x = self.grid_offset[0] + (x - y) * self.TILE_W / 2
        screen_y = self.grid_offset[1] + (x + y) * self.TILE_H / 2
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw grid tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                pos = (x, y)
                wx, wy = self._grid_to_world(pos)
                tile_points = [
                    (wx, wy),
                    (wx + self.TILE_W / 2, wy + self.TILE_H / 2),
                    (wx, wy + self.TILE_H),
                    (wx - self.TILE_W / 2, wy + self.TILE_H / 2)
                ]
                pygame.draw.polygon(self.screen, self.COLOR_GRID, tile_points)

                if self.grid[pos] == self.WALL:
                    self._render_iso_block(pos, self.COLOR_WALL, self.COLOR_WALL_TOP)
                elif self.grid[pos] == self.SOURCE:
                    self._render_iso_block(pos, self.COLOR_SOURCE, self.COLOR_SOURCE)
                elif self.grid[pos] == self.EXIT:
                    color = self.COLOR_EXIT_ACTIVE if pos in self.illuminated_exits else self.COLOR_EXIT_INACTIVE
                    self._render_iso_block(pos, color, color)

        # Draw crystals
        for i, crystal in enumerate(self.crystals):
            if self.held_crystal_idx is not None and i == self.held_crystal_idx:
                continue
            self._render_crystal(crystal)

        # Draw light beams
        for beam in self.light_beams:
            if len(beam) > 1:
                pygame.draw.lines(self.screen, self.COLOR_BEAM_GLOW, False, beam, width=7)
                pygame.draw.lines(self.screen, self.COLOR_BEAM, False, beam, width=2)
        
        # Draw cursor and held crystal
        self._render_cursor()
        if self.held_crystal_idx is not None:
            crystal = self.crystals[self.held_crystal_idx]
            self._render_crystal(crystal, override_pos=self.cursor_pos, is_held=True)

    def _render_iso_block(self, pos, side_color, top_color):
        wx, wy = self._grid_to_world(pos)
        top_points = [
            (wx, wy), (wx + self.TILE_W / 2, wy + self.TILE_H / 2),
            (wx, wy + self.TILE_H), (wx - self.TILE_W / 2, wy + self.TILE_H / 2)
        ]
        pygame.draw.polygon(self.screen, top_color, top_points)
        pygame.gfxdraw.aapolygon(self.screen, top_points, top_color)

    def _render_crystal(self, crystal, override_pos=None, is_held=False):
        pos = override_pos if override_pos else crystal['pos']
        wx, wy = self._grid_to_world(pos)
        color = crystal['color']
        
        size = self.TILE_W / 2.5
        points = [
            (wx, wy + self.TILE_H / 2 - size), (wx + size, wy + self.TILE_H / 2),
            (wx, wy + self.TILE_H / 2 + size), (wx - size, wy + self.TILE_H / 2)
        ]
        
        if is_held:
            s = pygame.Surface((self.TILE_W*2, self.TILE_H*2), pygame.SRCALPHA)
            s.set_alpha(150)
            pygame.draw.polygon(s, color, [(p[0]-wx+self.TILE_W, p[1]-wy+self.TILE_H) for p in points])
            self.screen.blit(s, (wx-self.TILE_W, wy-self.TILE_H))
        else:
            pygame.draw.polygon(self.screen, color, points)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Orientation indicator
        if crystal['orientation'] == 0: # '/'
            p1 = (points[3][0] + 5, points[3][1] + 2)
            p2 = (points[1][0] - 5, points[1][1] - 2)
        else: # '\'
            p1 = (points[0][0] - 5, points[0][1] + 2)
            p2 = (points[2][0] + 5, points[2][1] - 2)
        pygame.draw.line(self.screen, self.COLOR_BG, p1, p2, 2)

    def _render_cursor(self):
        wx, wy = self._grid_to_world(self.cursor_pos)
        points = [
            (wx, wy), (wx + self.TILE_W / 2, wy + self.TILE_H / 2),
            (wx, wy + self.TILE_H), (wx - self.TILE_W / 2, wy + self.TILE_H / 2)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, points, 2)

    def _render_ui(self):
        # Moves
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_CURSOR)
        self.screen.blit(moves_text, (20, 20))
        
        # Time
        time_text = self.font_ui.render(f"Time: {max(0, self.time_left):.1f}", True, self.COLOR_CURSOR)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 20, 20))
        
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_CURSOR)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 20))

        # Game Over Message
        if self.game_over:
            msg_text = "LEVEL CLEAR" if self.win else "GAME OVER"
            msg_color = self.COLOR_EXIT_ACTIVE if self.win else self.COLOR_EXIT_INACTIVE
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            text_surf = self.font_msg.render(msg_text, True, msg_color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0)

    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Buttons
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']}")
            obs, info = env.reset() # Auto-reset for continuous play
        
        # --- Rendering ---
        # The observation is already the rendered frame
        # We just need to get it back into a pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    pygame.quit()