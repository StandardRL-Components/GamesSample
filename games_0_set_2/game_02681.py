
# Generated: 2025-08-28T05:38:55.228987
# Source Brief: brief_02681.md
# Brief Index: 2681

        
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
        "Controls: Arrow keys to move the selector. Press space to cycle the selected tile's color."
    )

    game_description = (
        "A strategic puzzle game. Match 3 or more adjacent tiles of the same color to clear them. "
        "Clear the entire board within 20 moves to win!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.MAX_MOVES = 20
        self.NUM_COLORS = 6

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_LINES = (50, 60, 80)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.COLOR_EMPTY = (30, 35, 50)
        self.COLORS = [
            (220, 50, 50),    # Red
            (50, 220, 50),    # Green
            (50, 120, 220),   # Blue
            (220, 220, 50),   # Yellow
            (150, 50, 220),   # Purple
            (220, 120, 50),   # Orange
        ]

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)

        # --- Sizing & Layout ---
        self.GRID_PIXEL_SIZE = min(self.HEIGHT - 40, self.WIDTH - 40)
        self.TILE_SIZE = self.GRID_PIXEL_SIZE // self.GRID_SIZE
        self.GRID_PIXEL_SIZE = self.TILE_SIZE * self.GRID_SIZE # Recalculate to avoid gaps
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_PIXEL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_PIXEL_SIZE) // 2
        
        # --- Game State ---
        self.grid = None
        self.selector_pos = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.steps = None
        self.particles = []
        self.last_cleared_tiles = []

        self.reset()
        
        # This check is for development and ensures compliance.
        # self.validate_implementation() 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.particles = []
        self.last_cleared_tiles = []
        
        self._create_initial_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        action_taken = False
        reward = 0

        # 1. Handle Movement
        if movement != 0:
            action_taken = True
            if movement == 1: self.selector_pos[0] -= 1  # Up
            elif movement == 2: self.selector_pos[0] += 1  # Down
            elif movement == 3: self.selector_pos[1] -= 1  # Left
            elif movement == 4: self.selector_pos[1] += 1  # Right
            
            # Wrap around grid
            self.selector_pos[0] %= self.GRID_SIZE
            self.selector_pos[1] %= self.GRID_SIZE

        # 2. Handle Color Change
        if space_pressed:
            action_taken = True
            r, c = self.selector_pos
            if self.grid[r, c] != -1:
                # Cycle color
                self.grid[r, c] = (self.grid[r, c] + 1) % self.NUM_COLORS
                # Process matches and calculate reward
                reward += self._process_matches()

        # 3. Update Game State
        if action_taken:
            self.moves_left -= 1
        
        self.score += reward
        self.steps += 1
        
        # 4. Check Termination Conditions
        win = np.all(self.grid == -1)
        loss = self.moves_left <= 0 and not win
        terminated = win or loss
        
        if win:
            reward += 100
            self.score += 100
            self.game_over = True
        elif loss:
            reward -= 100
            self.score -= 100
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_initial_grid(self):
        """Generates a grid and ensures at least one 3-in-a-row match exists."""
        while True:
            self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))
            
            # Check for existing matches
            matches = self._find_matches()
            if not matches:
                break # Grid is valid if it has no initial matches
        
        # To guarantee a solvable state, we ensure a move can create a match.
        # We find two adjacent tiles and place a third one nearby that can be changed.
        # For simplicity in this context, we will just force a 3-in-a-row.
        # This also fulfills the "at least one immediate match" for a random agent.
        row = self.np_random.integers(0, self.GRID_SIZE)
        col = self.np_random.integers(0, self.GRID_SIZE - 2)
        color = self.np_random.integers(0, self.NUM_COLORS)
        self.grid[row, col] = self.grid[row, col+1] = self.grid[row, col+2] = color

    def _process_matches(self):
        """Repeatedly finds matches, clears tiles, applies gravity, and accumulates reward."""
        total_reward = 0
        total_cleared_this_turn = 0
        self.last_cleared_tiles.clear()

        while True:
            matches_coords = self._find_matches()
            if not matches_coords:
                break

            num_cleared = len(matches_coords)
            total_cleared_this_turn += num_cleared

            for r, c in matches_coords:
                self.last_cleared_tiles.append((r, c, self.grid[r,c]))
                self.grid[r, c] = -1
            
            self._apply_gravity_and_refill()

        if total_cleared_this_turn > 0:
            total_reward += total_cleared_this_turn * 0.1
            if total_cleared_this_turn > 3:
                total_reward += 5
        
        return total_reward

    def _find_matches(self):
        """Finds all groups of 3 or more same-colored tiles."""
        to_clear = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1:
                    continue
                
                # Horizontal check
                if c < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    to_clear.add((r, c)); to_clear.add((r, c+1)); to_clear.add((r, c+2))
                
                # Vertical check
                if r < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    to_clear.add((r, c)); to_clear.add((r+1, c)); to_clear.add((r+2, c))
        return to_clear

    def _apply_gravity_and_refill(self):
        """Shifts tiles down to fill empty spaces and adds new tiles at the top."""
        for c in range(self.GRID_SIZE):
            col = self.grid[:, c]
            non_empty = col[col != -1]
            num_empty = self.GRID_SIZE - len(non_empty)
            
            new_tiles = self.np_random.integers(0, self.NUM_COLORS, size=num_empty)
            self.grid[:, c] = np.concatenate((new_tiles, non_empty))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.TILE_SIZE, self.GRID_OFFSET_Y + self.GRID_PIXEL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_PIXEL_SIZE, self.GRID_OFFSET_Y + i * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)

        # Create particles for recently cleared tiles
        for r, c, color_idx in self.last_cleared_tiles:
            for _ in range(5): # 5 particles per tile
                px = self.GRID_OFFSET_X + c * self.TILE_SIZE + self.TILE_SIZE / 2
                py = self.GRID_OFFSET_Y + r * self.TILE_SIZE + self.TILE_SIZE / 2
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                self.particles.append([
                    [px, py], vel, self.COLORS[color_idx], self.np_random.uniform(5, 10)
                ])
        self.last_cleared_tiles.clear()

        # Update and draw particles
        for p in self.particles[:]:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[3] -= 0.5 # lifespan
            if p[3] <= 0:
                self.particles.remove(p)
            else:
                pygame.draw.circle(self.screen, p[2], (int(p[0][0]), int(p[0][1])), int(p[3]))

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.grid[r, c]
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.TILE_SIZE,
                    self.GRID_OFFSET_Y + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                
                color = self.COLOR_EMPTY if color_idx == -1 else self.COLORS[color_idx]
                border_color = tuple(min(255, x + 40) for x in color) if color_idx != -1 else self.COLOR_GRID_LINES

                pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))
                pygame.gfxdraw.rectangle(self.screen, rect.inflate(-4, -4), border_color)


        # Draw selector
        sel_r, sel_c = self.selector_pos
        selector_rect = pygame.Rect(
            self.GRID_OFFSET_X + sel_c * self.TILE_SIZE,
            self.GRID_OFFSET_Y + sel_r * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, 4)

    def _render_ui(self):
        # Moves Left
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, (255, 255, 255))
        self.screen.blit(moves_surf, (20, 10))

        # Score
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, (255, 255, 255))
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(score_surf, score_rect)
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            win = np.all(self.grid == -1)
            text = "YOU WIN!" if win else "GAME OVER"
            color = (100, 255, 100) if win else (255, 100, 100)
            
            go_surf = self.font_game_over.render(text, True, color)
            go_rect = go_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(go_surf, go_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Color Grid")
    
    running = True
    game_over = False
    
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not game_over:
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
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    game_over = False
                    continue
                
                obs, reward, terminated, truncated, info = env.step(action)
                game_over = terminated
                
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}, Terminated: {terminated}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
    env.close()