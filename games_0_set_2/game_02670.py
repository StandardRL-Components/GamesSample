
# Generated: 2025-08-27T21:04:35.775519
# Source Brief: brief_02670.md
# Brief Index: 2670

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to select a tile. "
        "Move to an adjacent tile and press Space again to swap. "
        "Press Shift to deselect."
    )

    game_description = (
        "A vibrant match-3 puzzle game. Swap adjacent tiles to create lines of "
        "3 or more. Trigger cascades for huge scores and try to reach the "
        "target score before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_SIZE = 8
        self.NUM_COLORS = 6
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.TARGET_SCORE = 1000
        self.MAX_STEPS = 1000

        # Visuals
        self.TILE_SIZE = 40
        self.GRID_WIDTH = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.TILE_SIZE
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2

        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 60, 80)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_SELECTED = (255, 255, 255)
        self.TILE_COLORS = [
            (0, 0, 0),  # 0 is empty
            (220, 50, 50),   # Red
            (50, 220, 50),   # Green
            (50, 150, 220),  # Blue
            (220, 220, 50),  # Yellow
            (180, 50, 220),  # Purple
            (240, 120, 50),  # Orange
        ]
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # State variables
        self.grid = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.selected_tile = None
        self.particles = []
        self.np_random = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile = None
        self.particles = []

        # Generate a board with at least one valid move
        while True:
            self._generate_board()
            if self._find_possible_moves():
                break
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        if shift_pressed and self.selected_tile:
            # Sound: Deselect
            self.selected_tile = None

        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dy) % self.GRID_SIZE
            self.cursor_pos[1] = (self.cursor_pos[1] + dx) % self.GRID_SIZE
            # Sound: Cursor move

        if space_pressed:
            cursor_r, cursor_c = self.cursor_pos
            if self.selected_tile is None:
                self.selected_tile = [cursor_r, cursor_c]
                # Sound: Select
            else:
                sel_r, sel_c = self.selected_tile
                # Check for adjacency
                if abs(sel_r - cursor_r) + abs(sel_c - cursor_c) == 1:
                    # Attempt swap
                    reward = self._attempt_swap(sel_r, sel_c, cursor_r, cursor_c)
                    self.selected_tile = None
                else: # Not adjacent, treat as new selection
                    self.selected_tile = [cursor_r, cursor_c]
                    # Sound: Select

        self.steps += 1
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.score >= self.TARGET_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif not self._find_possible_moves():
            reward -= 10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_swap(self, r1, c1, r2, c2):
        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
        
        # Check for matches
        total_reward = 0
        combo = 1
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            # Sound: Match
            num_cleared = len(matches)
            
            # Calculate reward for this wave
            wave_reward = num_cleared  # +1 per tile
            if num_cleared == 4: wave_reward += 5
            elif num_cleared >= 5: wave_reward += 10
            total_reward += wave_reward * combo

            # Clear tiles and create particles
            for r, c in matches:
                self._create_particles(r, c, self.grid[r, c])
                self.grid[r, c] = 0
            
            # Apply gravity and refill
            self._apply_gravity()
            self._refill_board()
            
            combo += 1
        
        if total_reward == 0:
            # No match found, swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            # Sound: Invalid swap
            return -0.1 # Penalty for invalid move
        
        return total_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Helper Functions ---

    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        # Ensure no initial matches
        while self._find_matches():
            matches = self._find_matches()
            for r, c in matches:
                self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1
    
    def _refill_board(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)
    
    def _find_possible_moves(self):
        moves = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Swap right
                if c < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches(): moves.append(((r, c), (r, c+1)))
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Swap down
                if r < self.GRID_SIZE - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches(): moves.append(((r, c), (r+1, c)))
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return moves

    def _create_particles(self, r, c, color_index):
        px, py = self._grid_to_screen(r, c)
        px += self.TILE_SIZE // 2
        py += self.TILE_SIZE // 2
        color = self.TILE_COLORS[color_index]
        for _ in range(15): # Create 15 particles per tile
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([ [px, py], vel, lifetime, color ])

    def _update_and_draw_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0] # Update x
            p[0][1] += p[1][1] # Update y
            p[2] -= 1 # Decrease lifetime
            
            # Draw particle
            size = max(0, p[2] / 5)
            pos = (int(p[0][0]), int(p[0][1]))
            pygame.draw.circle(self.screen, p[3], pos, int(size))
        
        # Remove dead particles
        self.particles = [p for p in self.particles if p[2] > 0]

    def _grid_to_screen(self, r, c):
        x = self.GRID_X_OFFSET + c * self.TILE_SIZE
        y = self.GRID_Y_OFFSET + r * self.TILE_SIZE
        return x, y

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_X_OFFSET + i * self.TILE_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + i * self.TILE_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)
            # Horizontal
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.TILE_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + i * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.grid[r, c]
                if color_index > 0:
                    x, y = self._grid_to_screen(r, c)
                    color = self.TILE_COLORS[color_index]
                    rect = pygame.Rect(x + 3, y + 3, self.TILE_SIZE - 6, self.TILE_SIZE - 6)
                    pygame.gfxdraw.box(self.screen, rect, (*color, 200)) # Semi-transparent fill
                    pygame.gfxdraw.rectangle(self.screen, rect, color) # Outline
        
        # Draw particles
        self._update_and_draw_particles()

        # Draw selected tile highlight
        if self.selected_tile:
            r, c = self.selected_tile
            x, y = self._grid_to_screen(r, c)
            rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 4, border_radius=5)

        # Draw cursor
        r, c = self.cursor_pos
        x, y = self._grid_to_screen(r, c)
        rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=5)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 20))
        
        # Target score display
        target_text = self.font_small.render(f"Target: {self.TARGET_SCORE}", True, (200, 200, 200))
        self.screen.blit(target_text, (20, 60))

        # Step display
        step_text = self.font_small.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, (200, 200, 200))
        step_rect = step_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(step_text, step_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "You Win!" if self.score >= self.TARGET_SCORE else "Game Over"
            end_text = self.font_large.render(msg, True, self.COLOR_CURSOR)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        # Can't call _get_observation before reset, so we do a light check
        assert self.observation_space.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert self.observation_space.dtype == np.uint8
        
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a display window
    pygame.display.set_caption("Match-3 Gym Environment")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    game_over_display = False

    while running:
        # Action defaults
        movement = 0 # No-op
        space_pressed = 0
        shift_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    game_over_display = False
                    continue
                if game_over_display:
                    continue

                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_pressed = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_pressed = 1
        
        if not game_over_display:
            action = [movement, space_pressed, shift_pressed]
            
            # Only step if an action was taken
            if any(action):
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
                if terminated:
                    game_over_display = True

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    env.close()