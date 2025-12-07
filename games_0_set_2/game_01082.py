
# Generated: 2025-08-27T15:49:26.574991
# Source Brief: brief_01082.md
# Brief Index: 1082

        
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
    """
    A tile-swapping puzzle game where the player must clear a 5x5 grid.

    The player controls a cursor to select and swap adjacent tiles. Swapping
    tiles to form horizontal or vertical lines of 3 or more identical tiles
    causes them to be cleared from the board. New tiles fall from the top to
    fill the gaps. The goal is to clear the entire board within a limited
    number of moves.

    The environment prioritizes visual feedback and game feel, with smooth
    animations for selections, matches, and particle effects for clearing tiles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a tile, "
        "then move to an adjacent tile and press Space again to swap."
    )

    # User-facing description of the game
    game_description = (
        "Swap adjacent tiles to form matching lines of 3 or more. "
        "Clear the entire board before you run out of moves!"
    )

    # The game is turn-based, so it only advances on action.
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 5, 5
    NUM_TILE_TYPES = 5
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_MOVES = 10
    MAX_STEPS = 1000 # Gym standard, but game ends with moves/win

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    TILE_COLORS = [
        (255, 89, 94),   # Red
        (255, 202, 58),  # Yellow
        (138, 201, 38),  # Green
        (25, 130, 196),  # Blue
        (106, 76, 147),  # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 255, 255)
    
    # Rewards
    REWARD_PER_TILE = 1
    REWARD_MATCH_3 = 5
    REWARD_MATCH_4 = 10
    REWARD_MATCH_5 = 20
    REWARD_WIN = 100
    REWARD_LOSE = -50
    PENALTY_INVALID_SWAP = -0.1

    class Particle:
        """A simple particle for visual effects."""
        def __init__(self, x, y, color):
            self.x = x
            self.y = y
            self.color = color
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed
            self.life = random.randint(15, 30) # life in frames
            self.radius = random.uniform(2, 5)

        def update(self):
            self.x += self.vx
            self.y += self.vy
            self.life -= 1
            self.radius -= 0.1
            return self.life > 0 and self.radius > 0

        def draw(self, surface):
            if self.life > 0 and self.radius > 0:
                pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), int(self.radius), self.color)

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_score = pygame.font.SysFont("Consolas", 32, bold=True)

        self.grid_rect = None
        self.tile_size = 0
        self.grid_offset_x = 0
        self.grid_offset_y = 0

        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.particles = []
        self.last_swap = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_pos = None
        self.particles = []
        self.last_swap = None

        self._calculate_grid_layout()
        self._generate_valid_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_press, _ = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0
        terminated = False
        self.last_swap = None

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1) # Down
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1) # Right

        # 2. Handle selection and swap logic
        if space_press:
            # Sound: select_tile.wav
            if self.selected_pos is None:
                self.selected_pos = list(self.cursor_pos)
            else:
                if self._is_adjacent(self.selected_pos, self.cursor_pos):
                    # --- SWAP LOGIC ---
                    self.moves_left -= 1
                    pos1, pos2 = tuple(self.selected_pos), tuple(self.cursor_pos)
                    
                    # Store original values in case of invalid swap
                    tile1_val, tile2_val = self.grid[pos1], self.grid[pos2]
                    self.grid[pos1], self.grid[pos2] = tile2_val, tile1_val
                    self.last_swap = (pos1, pos2)
                    
                    # --- MATCH AND CASCADE LOGIC ---
                    total_cleared_this_turn = 0
                    chain_reaction_bonus = 0
                    
                    matches = self._find_all_matches()
                    
                    if not matches:
                        # Invalid swap, swap back and penalize
                        self.grid[pos1], self.grid[pos2] = tile1_val, tile2_val
                        reward += self.PENALTY_INVALID_SWAP
                        # Sound: invalid_swap.wav
                    else:
                        # Valid swap, proceed with clearing
                        while matches:
                            # Sound: match_clear.wav
                            num_cleared = len(matches)
                            total_cleared_this_turn += num_cleared
                            
                            step_reward = num_cleared * self.REWARD_PER_TILE
                            step_reward += chain_reaction_bonus
                            
                            for match_set in self._group_matches(matches):
                                if len(match_set) == 3: step_reward += self.REWARD_MATCH_3
                                elif len(match_set) == 4: step_reward += self.REWARD_MATCH_4
                                elif len(match_set) >= 5: step_reward += self.REWARD_MATCH_5

                            reward += step_reward
                            self.score += int(step_reward)

                            for r, c in matches:
                                self._create_particles_at((r, c))
                            
                            self._clear_tiles(matches)
                            self._apply_gravity_and_refill()
                            chain_reaction_bonus += 5 # Bonus for cascades
                            
                            matches = self._find_all_matches()
                    
                    self.selected_pos = None # Reset selection after swap attempt
                else: # Not adjacent, just deselect
                    self.selected_pos = None

        # 3. Check for termination
        if self.moves_left <= 0:
            terminated = True
            reward += self.REWARD_LOSE
            self.score += self.REWARD_LOSE
            # Sound: game_over.wav
        
        if np.all(self.grid == -1): # All tiles cleared
            terminated = True
            reward += self.REWARD_WIN
            self.score += self.REWARD_WIN
            # Sound: game_win.wav

        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid_background()
        self._update_and_draw_particles()
        self._draw_tiles()
        self._draw_cursor_and_selection()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    # --- Drawing and Rendering ---

    def _calculate_grid_layout(self):
        grid_h = self.SCREEN_HEIGHT * 0.8
        self.tile_size = int(grid_h / self.GRID_HEIGHT)
        grid_w = self.tile_size * self.GRID_WIDTH
        self.grid_offset_x = (self.SCREEN_WIDTH - grid_w) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.tile_size * self.GRID_HEIGHT) // 2
        self.grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, grid_w, self.tile_size * self.GRID_HEIGHT)

    def _draw_grid_background(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.tile_size,
                    self.grid_offset_y + r * self.tile_size,
                    self.tile_size, self.tile_size
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _draw_tiles(self):
        padding = self.tile_size * 0.1
        radius = (self.tile_size - 2 * padding) / 2
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                tile_type = self.grid[r, c]
                if tile_type != -1:
                    center_x = self.grid_offset_x + c * self.tile_size + self.tile_size // 2
                    center_y = self.grid_offset_y + r * self.tile_size + self.tile_size // 2
                    color = self.TILE_COLORS[tile_type]
                    
                    current_radius = radius
                    # Simple "pop" animation for swapped tiles
                    if self.last_swap and ((r, c) in self.last_swap):
                        current_radius *= 1.15
                    
                    pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), int(current_radius), color)
                    pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), int(current_radius), color)

    def _draw_cursor_and_selection(self):
        # Draw selection first
        if self.selected_pos:
            r, c = self.selected_pos
            rect = pygame.Rect(
                self.grid_offset_x + c * self.tile_size,
                self.grid_offset_y + r * self.tile_size,
                self.tile_size, self.tile_size
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 3, border_radius=5)

        # Draw cursor on top
        r, c = self.cursor_pos
        rect = pygame.Rect(
            self.grid_offset_x + c * self.tile_size,
            self.grid_offset_y + r * self.tile_size,
            self.tile_size, self.tile_size
        )
        # Animate cursor alpha for a pulsing effect
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        cursor_color = (*self.COLOR_CURSOR, alpha)
        
        # Create a temporary surface for transparency
        cursor_surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, cursor_color, cursor_surface.get_rect(), 4, border_radius=5)
        self.screen.blit(cursor_surface, rect.topleft)

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, (200, 200, 220))
        self.screen.blit(moves_text, (20, 20))
        
        # Score
        score_text = self.font_score.render(f"{self.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(centerx=self.SCREEN_WIDTH / 2, y=10)
        self.screen.blit(score_text, score_rect)

    def _update_and_draw_particles(self):
        self.particles = [p for p in self.particles if p.update()]
        for p in self.particles:
            p.draw(self.screen)

    # --- Game Logic Helpers ---

    def _generate_valid_board(self):
        self.grid = self.np_random.integers(0, self.NUM_TILE_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while self._find_all_matches():
            # If matches exist, find the matched tiles and replace them
            matches = self._find_all_matches()
            for r, c in matches:
                self.grid[r,c] = self.np_random.integers(0, self.NUM_TILE_TYPES)

    def _is_adjacent(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2) == 1

    def _find_all_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    for i in range(3): matches.add((r, c+i))
                    # Check for 4 and 5 matches
                    if c + 3 < self.GRID_WIDTH and self.grid[r, c] == self.grid[r, c+3]:
                        matches.add((r, c+3))
                        if c + 4 < self.GRID_WIDTH and self.grid[r, c] == self.grid[r, c+4]:
                            matches.add((r, c+4))
        # Vertical
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if self.grid[r, c] != -1 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    for i in range(3): matches.add((r+i, c))
                    # Check for 4 and 5 matches
                    if r + 3 < self.GRID_HEIGHT and self.grid[r, c] == self.grid[r+3, c]:
                        matches.add((r+3, c))
                        if r + 4 < self.GRID_HEIGHT and self.grid[r, c] == self.grid[r+4, c]:
                            matches.add((r+4, c))
        return matches

    def _group_matches(self, match_coords):
        """Groups coordinates into contiguous lines for scoring."""
        if not match_coords: return []
        visited = set()
        groups = []
        for r_start, c_start in match_coords:
            if (r_start, c_start) not in visited:
                # Horizontal group
                h_group = set()
                for c in range(c_start, self.GRID_WIDTH):
                    if (r_start, c) in match_coords: h_group.add((r_start, c))
                    else: break
                for c in range(c_start - 1, -1, -1):
                    if (r_start, c) in match_coords: h_group.add((r_start, c))
                    else: break
                if len(h_group) >= 3:
                    groups.append(h_group)
                    visited.update(h_group)
                
                # Vertical group
                v_group = set()
                for r in range(r_start, self.GRID_HEIGHT):
                    if (r, c_start) in match_coords: v_group.add((r, c_start))
                    else: break
                for r in range(r_start - 1, -1, -1):
                    if (r, c_start) in match_coords: v_group.add((r, c_start))
                    else: break
                if len(v_group) >= 3:
                    groups.append(v_group)
                    visited.update(v_group)
        return groups

    def _clear_tiles(self, tiles_to_clear):
        for r, c in tiles_to_clear:
            self.grid[r, c] = -1 # -1 represents an empty space

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
            
            # Refill empty spaces at the top
            for r in range(empty_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_TILE_TYPES)
                # Sound: tile_refill.wav

    def _create_particles_at(self, pos):
        r, c = pos
        center_x = self.grid_offset_x + c * self.tile_size + self.tile_size // 2
        center_y = self.grid_offset_y + r * self.tile_size + self.tile_size // 2
        tile_type = self.grid[r, c]
        if tile_type != -1:
            color = self.TILE_COLORS[tile_type]
            for _ in range(15): # Number of particles per tile
                self.particles.append(self.Particle(center_x, center_y, color))
    
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set up Pygame window for human play
    pygame.display.set_caption("Tile Swap Puzzle")
    screen = pygame.display.set_mode((640, 400))
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    while running:
        # --- Human Controls ---
        movement, space_press = 0, 0 # 0=none, 1=up, 2=down, 3=left, 4=right
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_press = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        # Construct the action for the environment
        action = [movement, space_press, 0] # Shift is not used
        
        # Step the environment only when there is an action
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}, Terminated: {terminated}")
            if terminated:
                print("--- GAME OVER ---")
        
        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS

    pygame.quit()