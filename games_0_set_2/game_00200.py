
# Generated: 2025-08-27T12:54:51.742572
# Source Brief: brief_00200.md
# Brief Index: 200

        
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
    """
    A match-3 puzzle game where the player swaps adjacent tiles to create
    matches of three or more. The goal is to reach a target score within
    a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile, "
        "move to an adjacent tile, and press Space again to swap. Press Shift to deselect."
    )
    game_description = (
        "A colorful match-3 puzzle game. Swap adjacent tiles to create lines of three or more. "
        "Plan your moves to create cascading combos and reach the target score before you run out of moves!"
    )

    # Frame advance behavior
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.NUM_COLORS = 6
        self.TARGET_SCORE = 1000
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000 # Safety limit

        # --- Colors ---
        self.COLOR_BG = (25, 28, 44)
        self.COLOR_GRID = (50, 55, 75)
        self.TILE_COLORS = [
            (255, 89, 94),   # Red
            (138, 201, 38),  # Green
            (25, 130, 196),  # Blue
            (255, 202, 58),  # Yellow
            (132, 65, 208),  # Purple
            (255, 127, 80),  # Orange
        ]
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_SELECTED = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        
        # Turn/Animation state management
        self.turn_in_progress = False
        self.animation_state = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_pos = None
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.turn_in_progress = False
        self.animation_state = None

        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        
        # A "turn" (swap and its consequences) might take multiple steps to animate.
        # If a turn is in progress, we don't process new actions, just advance the animation.
        if self.turn_in_progress:
            reward += self._update_turn_state()
        else:
            reward += self._handle_input(action)
        
        self._update_particles()

        terminated = self._check_termination()
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        if terminated and not self.game_over:
            if self.score >= self.TARGET_SCORE:
                reward += 100 # Win bonus
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action
        reward = 0

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1)

        # --- Selection Logic (Space Button) ---
        is_space_press = space_held and not self.prev_space_held
        if is_space_press:
            if self.selected_pos is None:
                # Select a tile
                self.selected_pos = list(self.cursor_pos)
                # sfx: select_tile.wav
            else:
                # Attempt to swap with the selected tile
                if self._is_adjacent(self.selected_pos, self.cursor_pos):
                    # Start the turn
                    self.turn_in_progress = True
                    self.moves_left -= 1
                    self.animation_state = {
                        "type": "swap_attempt",
                        "pos1": self.selected_pos,
                        "pos2": list(self.cursor_pos),
                        "progress": 0
                    }
                    self.selected_pos = None
                    # sfx: swap.wav
                else:
                    # Invalid swap, just select the new tile
                    self.selected_pos = list(self.cursor_pos)
                    # sfx: invalid_move.wav

        # --- Deselection Logic (Shift Button) ---
        is_shift_press = shift_held and not self.prev_shift_held
        if is_shift_press and self.selected_pos is not None:
            self.selected_pos = None
            # sfx: deselect.wav

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return reward

    def _update_turn_state(self):
        """Manages the state machine for a single game turn (swap, match, cascade)."""
        if self.animation_state is None:
            self.turn_in_progress = False
            return 0

        state_type = self.animation_state["type"]
        reward = 0
        
        if state_type == "swap_attempt":
            self.animation_state["progress"] += 0.2
            if self.animation_state["progress"] >= 1.0:
                p1 = self.animation_state["pos1"]
                p2 = self.animation_state["pos2"]
                self._swap_tiles(p1, p2)
                matches = self._find_all_matches()
                if not matches:
                    # Invalid move, swap back
                    self.animation_state = {"type": "swap_back", "pos1": p1, "pos2": p2, "progress": 0}
                    reward = -0.1
                else:
                    self.animation_state = {"type": "clearing", "matches": matches, "progress": 0}
        
        elif state_type == "swap_back":
            self.animation_state["progress"] += 0.2
            if self.animation_state["progress"] >= 1.0:
                p1 = self.animation_state["pos1"]
                p2 = self.animation_state["pos2"]
                self._swap_tiles(p1, p2) # Swap back
                self.animation_state = None # End turn
                self.turn_in_progress = False

        elif state_type == "clearing":
            self.animation_state["progress"] += 0.15
            if self.animation_state["progress"] >= 1.0:
                matches = self.animation_state["matches"]
                reward += self._clear_and_score(matches)
                self.animation_state = {"type": "dropping", "progress": 0}

        elif state_type == "dropping":
            self.animation_state["progress"] += 0.25
            if self.animation_state["progress"] >= 1.0:
                self._drop_and_refill_tiles()
                new_matches = self._find_all_matches()
                if new_matches:
                    # Cascade!
                    self.animation_state = {"type": "clearing", "matches": new_matches, "progress": 0}
                    # sfx: cascade.wav
                else:
                    self.animation_state = None # End turn
                    self.turn_in_progress = False

        return reward

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_ROWS, self.GRID_COLS))
            if not self._find_all_matches():
                break

    def _is_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2) == 1

    def _swap_tiles(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2] and self.grid[r,c] != -1:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        # Vertical
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c] and self.grid[r,c] != -1:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _clear_and_score(self, matches):
        num_cleared = len(matches)
        if num_cleared == 0:
            return 0
        
        # sfx: match_clear.wav
        for r, c in matches:
            self._spawn_particles(r, c, self.grid[r, c])
            self.grid[r, c] = -1 # Mark as empty
        
        # Calculate score and reward
        self.score += num_cleared * 10
        reward = num_cleared # Base reward
        if num_cleared == 4: reward += 5
        if num_cleared >= 5: reward += 10
        return reward

    def _drop_and_refill_tiles(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self._swap_tiles((r, c), (empty_row, c))
                    empty_row -= 1
            # Refill
            for r in range(empty_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_COLORS)
                # sfx: tile_fall.wav

    def _check_termination(self):
        return self.moves_left <= 0 or self.score >= self.TARGET_SCORE

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Calculate grid dimensions and offsets
        grid_pixel_height = self.HEIGHT - 40
        self.tile_size = min((self.WIDTH - 40) // self.GRID_COLS, grid_pixel_height // self.GRID_ROWS)
        grid_w = self.tile_size * self.GRID_COLS
        grid_h = self.tile_size * self.GRID_ROWS
        self.offset_x = (self.WIDTH - grid_w) // 2
        self.offset_y = (self.HEIGHT - grid_h) // 2

        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.offset_x, self.offset_y, grid_w, grid_h), border_radius=5)

        # Draw tiles
        self._render_tiles()

        # Draw cursor and selection
        if not self.turn_in_progress:
            self._render_cursor()
            self._render_selection()
            
        self._render_particles()

    def _render_tiles(self):
        tile_margin = int(self.tile_size * 0.1)
        tile_inner_size = self.tile_size - 2 * tile_margin
        
        # Create a dictionary to hold tile positions for animation
        render_positions = {}
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                render_positions[(r, c)] = (
                    self.offset_x + c * self.tile_size + tile_margin,
                    self.offset_y + r * self.tile_size + tile_margin
                )

        # Handle animations that affect tile positions
        if self.animation_state and self.animation_state["type"] in ["swap_attempt", "swap_back"]:
            p = self.animation_state["progress"]
            p1_r, p1_c = self.animation_state["pos1"]
            p2_r, p2_c = self.animation_state["pos2"]
            
            start_x1, start_y1 = render_positions[(p1_r, p1_c)]
            start_x2, start_y2 = render_positions[(p2_r, p2_c)]
            
            render_positions[(p1_r, p1_c)] = (
                start_x1 + (start_x2 - start_x1) * p,
                start_y1 + (start_y2 - start_y1) * p,
            )
            render_positions[(p2_r, p2_c)] = (
                start_x2 + (start_x1 - start_x2) * p,
                start_y2 + (start_y1 - start_y2) * p,
            )

        # Draw all tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.grid[r, c]
                if color_idx == -1:
                    continue

                x, y = render_positions[(r, c)]
                size = tile_inner_size
                
                # Handle clearing animation
                if self.animation_state and self.animation_state["type"] == "clearing":
                    if (r, c) in self.animation_state["matches"]:
                        p = self.animation_state["progress"]
                        size = tile_inner_size * (1 - p)
                        x += (tile_inner_size - size) / 2
                        y += (tile_inner_size - size) / 2
                
                # Handle dropping animation
                if self.animation_state and self.animation_state["type"] == "dropping":
                     # A simple visual pop-in is sufficient here
                     pass

                tile_rect = pygame.Rect(int(x), int(y), int(max(0, size)), int(max(0, size)))
                pygame.draw.rect(self.screen, self.TILE_COLORS[color_idx], tile_rect, border_radius=int(self.tile_size * 0.2))

    def _render_cursor(self):
        r, c = self.cursor_pos
        x = self.offset_x + c * self.tile_size
        y = self.offset_y + r * self.tile_size
        cursor_rect = pygame.Rect(x, y, self.tile_size, self.tile_size)
        
        s = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, (x, y))
        pygame.draw.rect(self.screen, self.COLOR_SELECTED, cursor_rect, 2, border_radius=int(self.tile_size*0.15))

    def _render_selection(self):
        if self.selected_pos is not None:
            r, c = self.selected_pos
            x = self.offset_x + c * self.tile_size
            y = self.offset_y + r * self.tile_size
            
            # Pulsating effect
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
            thickness = 2 + int(pulse * 2)
            
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, (x, y, self.tile_size, self.tile_size), thickness, border_radius=int(self.tile_size*0.15))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Moves
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 15, 10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win_status = "You Win!" if self.score >= self.TARGET_SCORE else "Game Over"
            msg_text = self.font_main.render(win_status, True, self.COLOR_SELECTED)
            msg_rect = msg_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(msg_text, msg_rect)

            reset_text = self.font_small.render("Call reset() to play again", True, self.COLOR_TEXT)
            reset_rect = reset_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(reset_text, reset_rect)

    def _spawn_particles(self, r, c, color_idx):
        x = self.offset_x + c * self.tile_size + self.tile_size / 2
        y = self.offset_y + r * self.tile_size + self.tile_size / 2
        color = self.TILE_COLORS[color_idx]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(20, 40)
            self.particles.append([x, y, vx, vy, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1
        self.particles = [p for p in self.particles if p[4] > 0]

    def _render_particles(self):
        for x, y, vx, vy, lifetime, color in self.particles:
            radius = int(max(0, lifetime * 0.15))
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), radius, color)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Match-3 Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0
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
                print("Resetting environment...")
                obs, info = env.reset()

        # In a turn-based game, we only step when there's an action or a turn is resolving
        if movement != 0 or env.prev_space_held != space_held or env.prev_shift_held != shift_held or env.turn_in_progress:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")
            if terminated:
                print("Game Over!")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    pygame.quit()