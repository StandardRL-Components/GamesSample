
# Generated: 2025-08-27T16:44:00.118649
# Source Brief: brief_01312.md
# Brief Index: 1312

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    """
    A match-3 style puzzle game where the player selects groups of same-colored
    blocks to clear them from a grid. The goal is to reach a target score by
    creating matches and triggering cascading combos. The game ends upon
    reaching the score, running out of possible moves, or exceeding the step limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a block group. Press Shift to restart."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match groups of 3 or more same-colored blocks to score points. Plan your moves to create cascading combos and reach the target score before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 6
        self.NUM_COLORS = 3
        self.TARGET_SCORE = 500
        self.MAX_STEPS = 1000

        # --- Visual Constants ---
        self.COLOR_BG = pygame.Color("#1A1A2E")
        self.COLOR_GRID = pygame.Color("#16213E")
        self.COLOR_UI_TEXT = pygame.Color("#E94560")
        self.COLOR_CURSOR = pygame.Color("#F0E3FF")
        self.COLORS = {
            0: pygame.Color("#1A1A2E"),  # Empty
            1: pygame.Color("#E94560"),  # Red/Pink
            2: pygame.Color("#00A8CC"),  # Green/Cyan
            3: pygame.Color("#F9D56E"),  # Blue/Yellow
        }
        self.BLOCK_SIZE = 52
        self.GRID_LINE_WIDTH = 4
        self.BLOCK_MARGIN = 8
        self.BLOCK_BORDER_RADIUS = 8
        self.GRID_WIDTH = self.GRID_SIZE * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH) + self.GRID_LINE_WIDTH
        self.GRID_HEIGHT = self.GRID_SIZE * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH) + self.GRID_LINE_WIDTH
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_score = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 48, bold=True)
        
        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.last_space_held = None
        self.particles = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.particles = []

        # Generate a valid initial grid
        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            if self._find_all_matches():
                break
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        # Action 1: Reset via Shift key
        if shift_held:
            terminated = True
            # No reward for resetting, observation will be from the new state after reset
            return self._get_observation(), 0, terminated, False, self._get_info()

        # Action 2: Move cursor
        self._move_cursor(movement)

        # Action 3: Select block with Space key (on press)
        space_press = space_held and not self.last_space_held
        if space_press:
            reward = self._handle_selection()
        
        self.last_space_held = space_held
        
        # Update game logic
        self.steps += 1
        self._update_particles()
        
        # Check for termination conditions
        if self.score >= self.TARGET_SCORE:
            terminated = True
            reward += 100  # Win bonus
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        elif not self._find_all_matches():
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_selection(self):
        """Processes a block selection, including matching, cascades, and rewards."""
        r, c = self.cursor_pos
        if self.grid[r, c] == 0:
            return -0.1  # Penalty for selecting empty space

        component = self._find_component(r, c)
        
        if len(component) < 3:
            # sfx: invalid_move_sound
            return -0.1  # Penalty for non-match

        # sfx: match_sound
        total_reward = 0
        coords_to_process = component

        # Cascade loop
        while coords_to_process:
            self._add_particles(coords_to_process)
            match_reward = self._calculate_match_reward(coords_to_process)
            total_reward += match_reward
            
            self._clear_blocks(coords_to_process)
            self._apply_gravity()
            self._fill_empty_top_rows()

            # sfx: cascade_sound
            coords_to_process = self._find_all_matches()

        return total_reward

    def _calculate_match_reward(self, matched_coords):
        num_blocks = len(matched_coords)
        base_reward = num_blocks
        bonus_reward = 0
        if num_blocks == 4:
            bonus_reward = 10
        elif num_blocks >= 5:
            bonus_reward = 20
        
        self.score += (base_reward + bonus_reward)
        return float(base_reward + bonus_reward)

    def _move_cursor(self, movement):
        r, c = self.cursor_pos
        if movement == 1:  # Up
            self.cursor_pos[0] = (r - 1) % self.GRID_SIZE
        elif movement == 2:  # Down
            self.cursor_pos[0] = (r + 1) % self.GRID_SIZE
        elif movement == 3:  # Left
            self.cursor_pos[1] = (c - 1) % self.GRID_SIZE
        elif movement == 4:  # Right
            self.cursor_pos[1] = (c + 1) % self.GRID_SIZE

    def _find_component(self, start_r, start_c):
        """Finds all connected blocks of the same color using BFS."""
        target_color = self.grid[start_r, start_c]
        if target_color == 0:
            return set()

        q = deque([(start_r, start_c)])
        visited = set([(start_r, start_c)])
        component = set([(start_r, start_c)])

        while q:
            r, c = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and \
                   (nr, nc) not in visited and self.grid[nr, nc] == target_color:
                    visited.add((nr, nc))
                    q.append((nr, nc))
                    component.add((nr, nc))
        return component

    def _find_all_matches(self):
        """Finds all groups of 3 or more on the entire grid."""
        all_matched_coords = set()
        visited = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r, c) not in visited:
                    component = self._find_component(r, c)
                    visited.update(component)
                    if len(component) >= 3:
                        all_matched_coords.update(component)
        return all_matched_coords

    def _clear_blocks(self, coords_to_clear):
        for r, c in coords_to_clear:
            self.grid[r, c] = 0

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1

    def _fill_empty_top_rows(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_WIDTH, self.GRID_HEIGHT))

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                block_color = self.COLORS.get(self.grid[r, c], self.COLOR_BG)
                
                # Calculate block position
                x = self.GRID_OFFSET_X + self.GRID_LINE_WIDTH + c * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
                y = self.GRID_OFFSET_Y + self.GRID_LINE_WIDTH + r * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
                
                # Draw block
                block_rect = pygame.Rect(x + self.BLOCK_MARGIN // 2, y + self.BLOCK_MARGIN // 2, self.BLOCK_SIZE - self.BLOCK_MARGIN, self.BLOCK_SIZE - self.BLOCK_MARGIN)
                pygame.draw.rect(self.screen, block_color, block_rect, border_radius=self.BLOCK_BORDER_RADIUS)
        
        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_x = self.GRID_OFFSET_X + self.GRID_LINE_WIDTH + cursor_c * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
        cursor_y = self.GRID_OFFSET_Y + self.GRID_LINE_WIDTH + cursor_r * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH)
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=4, border_radius=self.BLOCK_BORDER_RADIUS + 2)
        
        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['radius']))

    def _render_ui(self):
        # Render score
        score_text = self.font_score.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Render steps
        steps_text = self.font_score.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 20, 10))

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_or_lose_text = "YOU WIN!" if self.score >= self.TARGET_SCORE else "GAME OVER"
            text_surface = self.font_game_over.render(win_or_lose_text, True, self.COLOR_UI_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _add_particles(self, coords):
        for r, c in coords:
            color = self.COLORS[self.grid[r, c]]
            center_x = self.GRID_OFFSET_X + self.GRID_LINE_WIDTH + c * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH) + self.BLOCK_SIZE / 2
            center_y = self.GRID_OFFSET_Y + self.GRID_LINE_WIDTH + r * (self.BLOCK_SIZE + self.GRID_LINE_WIDTH) + self.BLOCK_SIZE / 2
            
            for _ in range(5): # 5 particles per block
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 5)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                self.particles.append({
                    'pos': [center_x, center_y],
                    'vel': vel,
                    'radius': self.np_random.uniform(3, 7),
                    'color': color,
                    'life': 20 # frames
                })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] *= 0.95
            if p['life'] <= 0 or p['radius'] < 1:
                self.particles.remove(p)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": list(self.cursor_pos),
            "is_game_over": self.game_over,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Block Matcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    
    while running:
        # --- Human Controls ---
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        # --- Environment Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']}")
            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
        else:
            # If game is over, allow reset with Shift key
            if shift:
                obs, info = env.reset()
                terminated = False

        # --- Rendering ---
        # Pygame surface is managed internally by env, just need to get it and blit
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()