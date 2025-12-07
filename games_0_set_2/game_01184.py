import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to swap with the tile in the last direction moved."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent colored tiles to create matches of 3 or more in a grid-based puzzle to clear the board within a limited number of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_COLORS = 5
    TILE_SIZE = 48
    GRID_LINE_WIDTH = 2
    SELECTOR_WIDTH = 4
    
    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_SELECTOR = (255, 255, 0)
    TILE_COLORS = [
        (220, 50, 50),   # Red
        (50, 220, 50),   # Green
        (50, 150, 220),  # Blue
        (220, 150, 50),  # Orange
        (150, 50, 220),  # Purple
    ]
    COLOR_EMPTY = (30, 45, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_FLASH = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Calculate grid rendering offsets
        self.grid_pixel_width = self.GRID_WIDTH * self.TILE_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.TILE_SIZE
        self.grid_offset_x = (self.screen.get_width() - self.grid_pixel_width) // 2
        self.grid_offset_y = (self.screen.get_height() - self.grid_pixel_height) // 2

        # Initialize state variables
        self.grid = None
        self.selector_pos = None
        self.last_move_dir = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.steps = None
        self.particles = []
        self._flash_animation_data = [] # Tiles to flash this frame

        # Run validation check
        # self.validate_implementation() # This is called by the test harness, no need to call it here.

    def _generate_board(self):
        """Generates a new board, ensuring no initial matches and at least one possible move."""
        while True:
            # Create a random board
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            
            # Resolve any starting matches
            while self._check_and_clear_matches(spawn_particles=False)[0]:
                self._apply_gravity_and_refill()

            # Check if any moves are possible
            if self._find_possible_moves():
                break # Valid board found

    def _find_possible_moves(self):
        """Checks the entire grid for any possible swaps that result in a match."""
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Check swap right
                if x < self.GRID_WIDTH - 1:
                    self._swap_tiles((x, y), (x + 1, y))
                    if self._check_matches():
                        self._swap_tiles((x, y), (x + 1, y)) # Swap back
                        return True
                    self._swap_tiles((x, y), (x + 1, y)) # Swap back
                
                # Check swap down
                if y < self.GRID_HEIGHT - 1:
                    self._swap_tiles((x, y), (x, y + 1))
                    if self._check_matches():
                        self._swap_tiles((x, y), (x, y + 1)) # Swap back
                        return True
                    self._swap_tiles((x, y), (x, y + 1)) # Swap back
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_board()
        
        self.selector_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_move_dir = [0, -1] # Default up
        self.moves_left = 20
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.particles = []
        self._flash_animation_data = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self._flash_animation_data = []

        movement = action[0]
        space_pressed = action[1] == 1
        
        # 1. Handle selector movement
        if movement > 0:
            if movement == 1: # Up
                self.selector_pos[1] -= 1
                self.last_move_dir = [0, -1]
            elif movement == 2: # Down
                self.selector_pos[1] += 1
                self.last_move_dir = [0, 1]
            elif movement == 3: # Left
                self.selector_pos[0] -= 1
                self.last_move_dir = [-1, 0]
            elif movement == 4: # Right
                self.selector_pos[0] += 1
                self.last_move_dir = [1, 0]
            
            # Wrap selector around grid
            self.selector_pos[0] %= self.GRID_WIDTH
            self.selector_pos[1] %= self.GRID_HEIGHT

        # 2. Handle swap action
        if space_pressed:
            self.moves_left -= 1
            
            pos1 = tuple(self.selector_pos)
            target_pos = (pos1[0] + self.last_move_dir[0], pos1[1] + self.last_move_dir[1])

            # Check if swap is valid (within bounds)
            if 0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT:
                self._swap_tiles(pos1, target_pos)
                
                # Check for matches and resolve them in a chain
                total_tiles_cleared, chain_bonus = self._resolve_all_matches()

                if total_tiles_cleared > 0:
                    # Successful match
                    reward += total_tiles_cleared * 1 # +1 per tile
                    if total_tiles_cleared > 3:
                        reward += 5 # Bonus for clearing more than 3
                    reward += chain_bonus
                    self.score += reward
                else:
                    # Invalid swap (no match), swap back
                    self._swap_tiles(pos1, target_pos)
                    reward -= 0.1 # Penalty for invalid swap
            else:
                # Attempted swap out of bounds
                reward -= 0.1

        # 3. Check for termination
        terminated = False
        if np.all(self.grid == 0): # Win condition
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0: # Lose condition
            reward -= 50
            self.score -= 50
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _swap_tiles(self, pos1, pos2):
        """Swaps two tiles in the grid."""
        self.grid[pos1], self.grid[pos2] = self.grid[pos2], self.grid[pos1]

    def _check_matches(self):
        """Finds all horizontal and vertical matches of 3 or more."""
        matches = set()
        # Horizontal
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                if self.grid[x, y] == self.grid[x+1, y] == self.grid[x+2, y] != 0:
                    for i in range(3): matches.add((x+i, y))
                    # Check for longer matches
                    for i in range(3, self.GRID_WIDTH - x):
                        if self.grid[x, y] == self.grid[x+i, y]: matches.add((x+i, y))
                        else: break

        # Vertical
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                if self.grid[x, y] == self.grid[x, y+1] == self.grid[x, y+2] != 0:
                    for i in range(3): matches.add((x, y+i))
                    # Check for longer matches
                    for i in range(3, self.GRID_HEIGHT - y):
                        if self.grid[x, y] == self.grid[x, y+i]: matches.add((x, y+i))
                        else: break
        return matches

    def _check_and_clear_matches(self, spawn_particles=True):
        """Checks for matches, clears them, and returns if any were found."""
        matched_tiles = self._check_matches()
        if not matched_tiles:
            return False, 0
        
        for x, y in matched_tiles:
            # Spawn particles for visual effect
            if spawn_particles:
                self._spawn_particles((x, y), self.TILE_COLORS[self.grid[x, y] - 1])
            self.grid[x, y] = 0 # Clear tile
        
        self._flash_animation_data.extend(list(matched_tiles))
        return True, len(matched_tiles)

    def _apply_gravity_and_refill(self):
        """Makes tiles fall down and refills empty top rows."""
        for x in range(self.GRID_WIDTH):
            empty_slots = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[x, y + empty_slots] = self.grid[x, y]
                    self.grid[x, y] = 0
            # Refill top empty slots
            for y in range(empty_slots):
                self.grid[x, y] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _resolve_all_matches(self):
        """Continuously finds and clears matches until none are left (chain reaction)."""
        total_tiles_cleared = 0
        chain_count = 0
        chain_bonus = 0
        while True:
            found_match, num_cleared = self._check_and_clear_matches()
            if not found_match:
                break
            
            total_tiles_cleared += num_cleared
            if chain_count > 0:
                chain_bonus += 10 # Chain reaction bonus
            chain_count += 1
            
            # Let tiles fall and refill before checking for new matches
            self._apply_gravity_and_refill()
        return total_tiles_cleared, chain_bonus

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                tile_val = self.grid[x, y]
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.TILE_SIZE,
                    self.grid_offset_y + y * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                
                # Draw background cell
                pygame.draw.rect(self.screen, self.COLOR_EMPTY, rect)

                # Draw tile
                if tile_val > 0:
                    color_index = tile_val - 1
                    color = self.TILE_COLORS[color_index]
                    
                    # Flash effect for matched tiles
                    if (x, y) in self._flash_animation_data:
                        color = self.COLOR_FLASH
                    
                    # Draw rounded rectangle for tile
                    pygame.draw.rect(self.screen, color, rect, border_radius=8)

        # Draw grid lines over tiles
        for i in range(self.GRID_WIDTH + 1):
            start_pos = (self.grid_offset_x + i * self.TILE_SIZE - self.GRID_LINE_WIDTH // 2, self.grid_offset_y)
            end_pos = (self.grid_offset_x + i * self.TILE_SIZE - self.GRID_LINE_WIDTH // 2, self.grid_offset_y + self.grid_pixel_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            start_pos = (self.grid_offset_x, self.grid_offset_y + i * self.TILE_SIZE - self.GRID_LINE_WIDTH // 2)
            end_pos = (self.grid_offset_x + self.grid_pixel_width, self.grid_offset_y + i * self.TILE_SIZE - self.GRID_LINE_WIDTH // 2)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, self.GRID_LINE_WIDTH)

        # Draw selector
        sel_x, sel_y = self.selector_pos
        selector_rect = pygame.Rect(
            self.grid_offset_x + sel_x * self.TILE_SIZE,
            self.grid_offset_y + sel_y * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, self.SELECTOR_WIDTH, border_radius=8)

    def _spawn_particles(self, grid_pos, color):
        """Spawns particles at a grid location."""
        # sfx: tile_clear.wav
        center_x = self.grid_offset_x + grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.grid_offset_y + grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = random.randint(3, 7)
            lifespan = random.randint(20, 40)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'size': size, 'lifespan': lifespan, 'color': color})

    def _render_particles(self):
        """Updates and draws all particles."""
        remaining_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['size'] *= 0.95 # Shrink
            if p['lifespan'] > 0 and p['size'] > 1:
                rect = pygame.Rect(int(p['pos'][0]), int(p['pos'][1]), int(p['size']), int(p['size']))
                pygame.draw.rect(self.screen, p['color'], rect)
                remaining_particles.append(p)
        self.particles = remaining_particles

    def _render_ui(self):
        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.screen.get_width() - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = np.all(self.grid == 0)
            msg = "Board Cleared!" if win_condition else "Out of Moves!"
            
            end_text = self.font_main.render(msg, True, self.COLOR_SELECTOR)
            end_rect = end_text.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))
            self.screen.blit(end_text, end_rect)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset to initialize attributes
        self.reset()
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset return values
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Tile Matcher")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
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
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # Since auto_advance is False, we only step when an action is taken
        if np.any(action > 0):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}, Terminated: {terminated}")

            if terminated:
                print("Game Over! Press 'R' to restart or 'ESC' to quit.")

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    pygame.quit()