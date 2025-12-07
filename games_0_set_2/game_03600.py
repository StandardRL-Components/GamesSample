
# Generated: 2025-08-27T23:50:52.980750
# Source Brief: brief_03600.md
# Brief Index: 3600

        
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
        "Controls: Arrow keys to move cursor. Space to select a block. "
        "Select a second, adjacent, matching block to clear a cluster. "
        "Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Connect adjacent blocks of the same color to clear them. "
        "Clear the whole board or get the highest score before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 7
    NUM_COLORS = 5
    INITIAL_MOVES = 50

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECT_GLOW = (255, 255, 100)
    
    BLOCK_COLORS = [
        (231, 76, 60),    # Red
        (52, 152, 219),   # Blue
        (46, 204, 113),   # Green
        (155, 89, 182),   # Purple
        (241, 196, 15),   # Yellow
    ]

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        # Calculate layout
        self.block_size = 40
        self.grid_width = self.GRID_SIZE * self.block_size
        self.grid_height = self.GRID_SIZE * self.block_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.selected_block_pos = None
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.INITIAL_MOVES
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_block_pos = None
        self.particles = []

        self._generate_board()
        
        return self._get_observation(), self._get_info()

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            if self._check_for_matches():
                break

    def _check_for_matches(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color = self.grid[r, c]
                if color == 0: continue
                # Check right
                if c < self.GRID_SIZE - 1 and self.grid[r, c+1] == color:
                    return True
                # Check down
                if r < self.GRID_SIZE - 1 and self.grid[r+1, c] == color:
                    return True
        return False

    def step(self, action):
        reward = 0
        self.game_over = False
        
        # Unpack factorized action
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)

        # 2. Handle deselect (highest priority)
        if shift_press and self.selected_block_pos:
            self.selected_block_pos = None
            # No move cost, no reward
        
        # 3. Handle select/match
        elif space_press:
            cursor_r, cursor_c = self.cursor_pos
            
            # Can't select an empty cell
            if self.grid[cursor_r, cursor_c] == 0:
                pass
            
            # First selection
            elif not self.selected_block_pos:
                self.selected_block_pos = (cursor_r, cursor_c)
            
            # Second selection (this action costs a move)
            else:
                self.moves_remaining -= 1
                
                sel_r, sel_c = self.selected_block_pos
                
                # Check for valid match: same color, adjacent, not same block
                is_adjacent = abs(sel_r - cursor_r) + abs(sel_c - cursor_c) == 1
                is_same_color = self.grid[sel_r, sel_c] == self.grid[cursor_r, cursor_c]

                if is_adjacent and is_same_color:
                    # --- Valid Match ---
                    # Find entire cluster
                    cluster = self._find_cluster(sel_r, sel_c)
                    cleared_count = len(cluster)
                    
                    # Calculate reward
                    reward += cleared_count  # +1 per block
                    if cleared_count >= 4:
                        reward += 5  # Bonus for large cluster
                    
                    self.score += reward
                    
                    # Clear blocks and create particles
                    for r, c in cluster:
                        self._create_particles(r, c)
                        self.grid[r, c] = 0
                        # sfx: block_clear.wav
                    
                    self._apply_gravity()
                    self.selected_block_pos = None

                else:
                    # --- Invalid Match ---
                    reward -= 0.1
                    self.score -= 0.1
                    self.selected_block_pos = None
                    # sfx: error.wav
        
        # 4. Update particles
        self._update_particles()
        
        # 5. Check termination conditions
        board_cleared = np.all(self.grid == 0)
        
        if self.moves_remaining <= 0 or board_cleared:
            self.game_over = True
            if board_cleared:
                reward += 50
                self.score += 50
                # sfx: win_jingle.wav
            else:
                # sfx: lose_sound.wav
                pass
        
        self.steps += 1
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _find_cluster(self, start_r, start_c):
        color_to_match = self.grid[start_r, start_c]
        if color_to_match == 0:
            return set()

        q = [(start_r, start_c)]
        visited = set(q)
        
        while q:
            r, c = q.pop(0)
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                    if (nr, nc) not in visited and self.grid[nr, nc] == color_to_match:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return visited

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
    
    def _create_particles(self, r, c):
        color_index = self.grid[r, c]
        if color_index == 0: return
        
        px = self.grid_offset_x + c * self.block_size + self.block_size / 2
        py = self.grid_offset_y + r * self.block_size + self.block_size / 2
        base_color = self.BLOCK_COLORS[color_index - 1]

        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = random.randint(10, 20)
            size = random.uniform(2, 5)
            self.particles.append([px, py, vx, vy, life, size, base_color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[1]  # x += vx
            p[1] += p[2]  # y += vy
            p[4] -= 1     # life -= 1

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Grid Lines ---
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.grid_offset_x + i * self.block_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.block_size, self.grid_offset_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.block_size)
            end_pos = (self.grid_offset_x + self.grid_width, self.grid_offset_y + i * self.block_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # --- Render Blocks & Selections ---
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.grid[r, c]
                if color_index > 0:
                    rect = pygame.Rect(
                        self.grid_offset_x + c * self.block_size + 2,
                        self.grid_offset_y + r * self.block_size + 2,
                        self.block_size - 4,
                        self.block_size - 4
                    )
                    pygame.draw.rect(self.screen, self.BLOCK_COLORS[color_index - 1], rect, border_radius=4)

        # --- Render Selected Block Glow ---
        if self.selected_block_pos:
            r, c = self.selected_block_pos
            rect = pygame.Rect(
                self.grid_offset_x + c * self.block_size,
                self.grid_offset_y + r * self.block_size,
                self.block_size,
                self.block_size
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECT_GLOW, rect, width=3, border_radius=6)

        # --- Render Cursor ---
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cursor_c * self.block_size,
            self.grid_offset_y + cursor_r * self.block_size,
            self.block_size,
            self.block_size
        )
        # Pulsating effect for cursor
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=line_width, border_radius=6)

        # --- Render Particles ---
        for x, y, vx, vy, life, size, color in self.particles:
            alpha = int(255 * (life / 20.0))
            if alpha > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, alpha), (size, size), size)
                self.screen.blit(s, (int(x-size), int(y-size)))

        # --- Render UI ---
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        moves_text = self.font_small.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 20))

        # --- Render Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render("GAME OVER", True, self.COLOR_UI_TEXT)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": list(self.cursor_pos),
            "selected_pos": self.selected_block_pos,
        }

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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Matcher")
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    
    # Game loop
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                
        action = [movement, space, shift]
        
        # Only step if an action was taken
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_remaining']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS for human play

    env.close()