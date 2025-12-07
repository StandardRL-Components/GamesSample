
# Generated: 2025-08-28T02:17:26.767609
# Source Brief: brief_04398.md
# Brief Index: 4398

        
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

    user_guide = (
        "Controls: ↑↓←→ to move the selector. Space to select a tile. Shift to deselect."
    )

    game_description = (
        "Swap adjacent tiles to create matches of three or more. Clear the entire board before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_COLORS = 5
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Visuals ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (45, 50, 55)
        self.COLOR_TEXT = (220, 220, 220)
        self.TILE_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.TILE_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.TILE_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.TILE_SIZE) // 2
        self.EMPTY_TILE = -1

        # --- State Variables ---
        self.board = None
        self.selector_pos = None
        self.selected_tile = None
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.just_cleared_info = [] # Stores {'pos': (x,y), 'color': (r,g,b)}

        self.reset()
        
        # Self-validation
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = self._generate_board()
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile = None
        self.moves_left = self.MAX_MOVES
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.just_cleared_info = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.just_cleared_info.clear() # Clear effects from previous step

        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0
        action_taken = any([movement > 0, space_press, shift_press])

        if shift_press and self.selected_tile:
            self.selected_tile = None
            # sfx: deselect sound

        if movement > 0:
            if movement == 1: self.selector_pos[1] -= 1  # Up
            elif movement == 2: self.selector_pos[1] += 1  # Down
            elif movement == 3: self.selector_pos[0] -= 1  # Left
            elif movement == 4: self.selector_pos[0] += 1  # Right
            self.selector_pos[0] = np.clip(self.selector_pos[0], 0, self.GRID_SIZE - 1)
            self.selector_pos[1] = np.clip(self.selector_pos[1], 0, self.GRID_SIZE - 1)
            # sfx: selector move tick

        if space_press:
            current_selection = tuple(self.selector_pos)
            if self.selected_tile is None:
                self.selected_tile = current_selection
                # sfx: select sound
            elif self.selected_tile == current_selection:
                self.selected_tile = None # Deselect if same tile
                # sfx: deselect sound
            else:
                # This is the second selection (a swap attempt)
                first = self.selected_tile
                second = current_selection
                self.selected_tile = None # Clear selection after attempt

                # Check for adjacency
                if abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1:
                    # sfx: swap attempt
                    if self._try_swap(first, second):
                        self.moves_left -= 1
                        cascade_reward, cleared_info = self._process_cascades()
                        reward += cascade_reward
                        self.just_cleared_info = cleared_info
                    else:
                        reward -= 0.1 # Penalty for invalid swap
                else:
                    reward -= 0.1 # Penalty for non-adjacent selection
                    # sfx: error sound

        if not action_taken:
            reward = 0 # No penalty for no-op

        # Check for termination conditions
        terminated = False
        is_board_clear = np.all(self.board == self.EMPTY_TILE)
        
        if is_board_clear:
            reward += 100
            terminated = True
            # sfx: win fanfare
        elif self.moves_left <= 0 or self.steps >= self.MAX_STEPS:
            reward -= 100 if not is_board_clear else 0
            terminated = True
            # sfx: lose sound
        
        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "moves_left": self.moves_left,
        }

    # --- Game Logic Helpers ---

    def _generate_board(self):
        board = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))
        while True:
            matches = self._find_matches(board)
            if not matches:
                break
            for x, y in matches:
                board[y, x] = self.np_random.integers(0, self.NUM_COLORS)
        return board

    def _find_matches(self, board):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if board[r,c] == self.EMPTY_TILE: continue
                # Horizontal
                if c < self.GRID_SIZE - 2 and board[r, c] == board[r, c+1] == board[r, c+2]:
                    matches.update([(c, r), (c+1, r), (c+2, r)])
                # Vertical
                if r < self.GRID_SIZE - 2 and board[r, c] == board[r+1, c] == board[r+2, c]:
                    matches.update([(c, r), (c, r+1), (c, r+2)])
        return list(matches)
    
    def _try_swap(self, pos1, pos2):
        temp_board = self.board.copy()
        temp_board[pos1[1], pos1[0]], temp_board[pos2[1], pos2[0]] = \
            temp_board[pos2[1], pos2[0]], temp_board[pos1[1], pos1[0]]
        
        if self._find_matches(temp_board):
            self.board = temp_board # Commit the swap
            return True
        return False

    def _process_cascades(self):
        total_reward = 0
        all_cleared_info = []
        
        while True:
            matches = self._find_matches(self.board)
            if not matches:
                break
            
            # sfx: match clear sound
            num_cleared = len(matches)
            total_reward += num_cleared # +1 for each tile
            if num_cleared > 3:
                total_reward += 5 # Combo bonus

            # Store info for particle effects
            for x, y in matches:
                if self.board[y, x] != self.EMPTY_TILE:
                    color_index = self.board[y, x]
                    all_cleared_info.append({'pos': (x, y), 'color': self.TILE_COLORS[color_index]})

            # Clear tiles
            for x, y in matches:
                self.board[y, x] = self.EMPTY_TILE
            
            # Apply gravity
            for c in range(self.GRID_SIZE):
                empty_row = self.GRID_SIZE - 1
                for r in range(self.GRID_SIZE - 1, -1, -1):
                    if self.board[r, c] != self.EMPTY_TILE:
                        self.board[empty_row, c], self.board[r, c] = self.board[r, c], self.board[empty_row, c]
                        empty_row -= 1
            
            # Refill board
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    if self.board[r, c] == self.EMPTY_TILE:
                        self.board[r, c] = self.np_random.integers(0, self.NUM_COLORS)
                        # sfx: tile fall/refill sound
        
        return total_reward, all_cleared_info

    # --- Rendering Helpers ---

    def _render_game(self):
        self._render_grid()
        self._render_tiles()
        self._render_selector()
        self._spawn_and_render_particles()

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y + self.GRID_SIZE * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_SIZE * self.TILE_SIZE, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_tiles(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.board[r, c]
                if color_index != self.EMPTY_TILE:
                    self._draw_tile(c, r, self.TILE_COLORS[color_index])

    def _draw_tile(self, c, r, color):
        rect = pygame.Rect(
            self.GRID_OFFSET_X + c * self.TILE_SIZE + 2,
            self.GRID_OFFSET_Y + r * self.TILE_SIZE + 2,
            self.TILE_SIZE - 4,
            self.TILE_SIZE - 4
        )
        border_color = tuple(max(0, val - 40) for val in color)
        
        pygame.gfxdraw.box(self.screen, rect, color)
        pygame.gfxdraw.rectangle(self.screen, rect, border_color)

    def _render_selector(self):
        sel_x, sel_y = self.selector_pos
        
        # Pulse animation for glow
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
        glow_size = int(6 + pulse * 4)
        glow_alpha = int(50 + pulse * 40)

        # Glow
        glow_rect = pygame.Rect(
            self.GRID_OFFSET_X + sel_x * self.TILE_SIZE - glow_size // 2,
            self.GRID_OFFSET_Y + sel_y * self.TILE_SIZE - glow_size // 2,
            self.TILE_SIZE + glow_size,
            self.TILE_SIZE + glow_size
        )
        glow_color = (255, 255, 255, glow_alpha)
        
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, glow_color, shape_surf.get_rect(), border_radius=8)
        self.screen.blit(shape_surf, glow_rect)
        
        # Main selector border
        sel_rect = pygame.Rect(
            self.GRID_OFFSET_X + sel_x * self.TILE_SIZE,
            self.GRID_OFFSET_Y + sel_y * self.TILE_SIZE,
            self.TILE_SIZE,
            self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, (255, 255, 255), sel_rect, 3, border_radius=4)
        
        # Highlight first selected tile
        if self.selected_tile:
            first_sel_x, first_sel_y = self.selected_tile
            first_rect = pygame.Rect(
                self.GRID_OFFSET_X + first_sel_x * self.TILE_SIZE + 3,
                self.GRID_OFFSET_Y + first_sel_y * self.TILE_SIZE + 3,
                self.TILE_SIZE - 6,
                self.TILE_SIZE - 6
            )
            pygame.draw.rect(self.screen, (255, 255, 255, 150), first_rect, 2, border_radius=4)

    def _spawn_and_render_particles(self):
        # Since auto_advance=False, particles must live and die in one frame.
        # This simulates their entire lifecycle and renders them.
        if self.just_cleared_info:
            for info in self.just_cleared_info:
                px = self.GRID_OFFSET_X + info['pos'][0] * self.TILE_SIZE + self.TILE_SIZE // 2
                py = self.GRID_OFFSET_Y + info['pos'][1] * self.TILE_SIZE + self.TILE_SIZE // 2
                
                for _ in range(15): # 15 particles per cleared tile
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    lifetime = self.np_random.integers(10, 20)
                    
                    vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
                    color = info['color']
                    
                    # Simulate and draw the particle's life as a fading streak
                    for t in range(lifetime):
                        current_alpha = 255 * (1 - t / lifetime)
                        if current_alpha > 0:
                            x = int(px + vx * t)
                            y = int(py + vy * t)
                            pygame.gfxdraw.pixel(self.screen, x, y, (*color, int(current_alpha)))
        
    def _render_ui(self):
        # Moves Left
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Score
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            is_win = np.all(self.board == self.EMPTY_TILE)
            msg = "BOARD CLEARED!" if is_win else "OUT OF MOVES"
            
            end_text = self.font_large.render(msg, True, (255, 255, 100))
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen to be a display surface
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Match-3 Gym Environment")
    
    done = False
    total_reward = 0
    
    # Game loop
    while not done:
        action = [0, 0, 0] # Default no-op
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        # Only step if an action was taken
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Moves: {info['moves_left']}")

            if terminated:
                print("--- GAME OVER ---")
                print(f"Final Score: {info['score']}")

        # Render the current state to the display
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Limit FPS for human play

    env.close()