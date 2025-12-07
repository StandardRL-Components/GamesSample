import os
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame


# Set the SDL video driver to "dummy" to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    """
    A puzzle game where the player matches colored blocks in a grid to score points.
    The goal is to reach a target score before running out of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press space to select a block. Hold shift to reset."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match colored blocks in a grid to reach a target score before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_COLORS = 5
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WIN_SCORE = 1000
    MAX_MOVES = 20
    MATCH_MIN_SIZE = 3

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_LINES = (40, 60, 80)
    BLOCK_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 120, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    # --- Enums for grid state ---
    EMPTY_CELL = 0
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game state variables are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.win = None
        self.steps = None
        self.particles = []

        # Calculate grid rendering properties
        self.grid_area_height = self.SCREEN_HEIGHT - 20
        self.block_size = min(
            (self.SCREEN_WIDTH - 40) // self.GRID_WIDTH, 
            self.grid_area_height // self.GRID_HEIGHT
        )
        self.grid_render_width = self.block_size * self.GRID_WIDTH
        self.grid_render_height = self.block_size * self.GRID_HEIGHT
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_render_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_render_height) // 2
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.steps = 0
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles = []

        # Generate a valid starting grid
        self._generate_initial_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        reward = 0
        terminated = False

        if shift_pressed:
            # Special action: reset the environment
            obs, info = self.reset()
            return obs, 0, False, False, info

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Actions ---
        # 1. Movement
        if movement == 1:  # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2:  # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3:  # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4:  # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)
        
        # 2. Selection
        if space_pressed:
            self.moves_left -= 1
            match_reward, match_found = self._process_selection()
            reward += match_reward
            if match_found:
                self.score += int(match_reward)
        
        # --- Check Termination Conditions ---
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100  # Win bonus
        elif self.moves_left <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 10 # Lose penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_selection(self):
        """Handles the logic when the player selects a block."""
        x, y = self.cursor_pos
        block_type = self.grid[y, x]

        if block_type == self.EMPTY_CELL:
            return -0.1, False # Penalty for selecting empty space

        # Find connected blocks of the same color
        matched_blocks = self._find_matches(x, y)
        
        if len(matched_blocks) < self.MATCH_MIN_SIZE:
            # Not a valid match
            return -0.1, False # Penalty for failed match
        
        # --- Process a successful match ---
        
        # Calculate reward
        reward = len(matched_blocks)  # Base reward
        if len(matched_blocks) > 5:
            reward += 10 # Combo bonus
        
        # Clear matched blocks and create particles
        for pos_y, pos_x in matched_blocks:
            self._create_particles(pos_x, pos_y, self.grid[pos_y, pos_x])
            self.grid[pos_y, pos_x] = self.EMPTY_CELL
        
        # Apply gravity and refill grid
        self._apply_gravity()
        self._refill_grid()
        
        # Reshuffle if no moves left
        if not self._has_valid_moves():
            self._generate_initial_grid() 
            
        return reward, True

    def _find_matches(self, start_x, start_y):
        """Finds all connected blocks of the same color using BFS."""
        target_color = self.grid[start_y, start_x]
        if target_color == self.EMPTY_CELL:
            return []

        q = deque([(start_y, start_x)])
        visited = set([(start_y, start_x)])
        matches = []

        while q:
            y, x = q.popleft()
            matches.append((y, x))

            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and
                        (ny, nx) not in visited and self.grid[ny, nx] == target_color):
                    visited.add((ny, nx))
                    q.append((ny, nx))
        
        return matches

    def _apply_gravity(self):
        """Makes blocks fall down to fill empty spaces."""
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] != self.EMPTY_CELL:
                    if empty_row != y:
                        self.grid[empty_row, x], self.grid[y, x] = self.grid[y, x], self.grid[empty_row, x]
                    empty_row -= 1
    
    def _refill_grid(self):
        """Fills empty cells at the top with new random blocks."""
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] == self.EMPTY_CELL:
                    self.grid[y, x] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _has_valid_moves(self):
        """Checks if there is at least one group of MATCH_MIN_SIZE or more."""
        visited = set()
        for y_start in range(self.GRID_HEIGHT):
            for x_start in range(self.GRID_WIDTH):
                if (y_start, x_start) in visited:
                    continue
                
                color = self.grid[y_start, x_start]
                if color == self.EMPTY_CELL:
                    continue

                q = deque([(y_start, x_start)])
                group = set([(y_start, x_start)])
                
                while q:
                    y, x = q.popleft()
                    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and
                                (ny, nx) not in group and self.grid[ny, nx] == color):
                            group.add((ny, nx))
                            q.append((ny, nx))
                
                visited.update(group)
                
                if len(group) >= self.MATCH_MIN_SIZE:
                    return True
                    
        return False

    def _generate_initial_grid(self):
        """Generates a new grid and ensures it has at least one valid move."""
        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if self._has_valid_moves():
                break

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_particles()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "steps": self.steps
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_offset_x + i * self.block_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_render_height))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_offset_y + i * self.block_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_render_width, y))

        # Draw blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                block_type = self.grid[y, x]
                if block_type != self.EMPTY_CELL:
                    color = self.BLOCK_COLORS[block_type - 1]
                    rect = pygame.Rect(
                        self.grid_offset_x + x * self.block_size,
                        self.grid_offset_y + y * self.block_size,
                        self.block_size,
                        self.block_size
                    )
                    # Draw a slightly smaller, filled inner rect for a border effect
                    inner_rect = rect.inflate(-4, -4)
                    pygame.draw.rect(self.screen, color, inner_rect, border_radius=4)
        
        # Draw cursor
        cursor_x = self.grid_offset_x + self.cursor_pos[0] * self.block_size
        cursor_y = self.grid_offset_y + self.cursor_pos[1] * self.block_size
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.block_size, self.block_size)
        
        # Pulsating effect for cursor
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # Varies between 0 and 1
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=line_width, border_radius=6)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surface = font.render(text, True, color)
            self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surface, pos)

        # Draw Score
        score_text = f"Score: {self.score}"
        draw_text(score_text, self.font_main, self.COLOR_TEXT, (20, 20))

        # Draw Moves Left
        moves_text = f"Moves: {self.moves_left}"
        text_width = self.font_main.size(moves_text)[0]
        draw_text(moves_text, self.font_main, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 20, 20))

        # Draw Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _create_particles(self, grid_x, grid_y, block_type):
        """Creates a burst of particles for a matched block."""
        px = self.grid_offset_x + (grid_x + 0.5) * self.block_size
        py = self.grid_offset_y + (grid_y + 0.5) * self.block_size
        color = self.BLOCK_COLORS[block_type - 1]
        
        for _ in range(10): # Number of particles per block
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30) # Frames
            size = self.np_random.uniform(2, 5)
            self.particles.append(Particle(px, py, vx, vy, size, lifetime, color))
            
    def _update_and_draw_particles(self):
        """Updates particle physics and draws them."""
        active_particles = []
        for p in self.particles:
            p.update()
            if p.is_alive():
                p.draw(self.screen)
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()

class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, x, y, vx, vy, size, lifetime, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1 # gravity
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)

    def is_alive(self):
        return self.lifetime > 0

    def draw(self, surface):
        if self.is_alive():
            # Fade out effect
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            r, g, b = self.color
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(
                temp_surf, int(self.size), int(self.size), int(self.size), (r, g, b, alpha)
            )
            surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)))

# Example usage:
if __name__ == '__main__':
    # To run with a display, comment out the os.environ line at the top
    # and uncomment the following line.
    # os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    pygame.display.set_caption("Block Matcher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # no-op, no space, no shift

    print("\n" + env.user_guide)
    
    # To make the game turn-based for manual play, we only step on an action
    action_taken = False

    while running:
        movement = 0 # no-op
        space_pressed = 0
        shift_pressed = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    action_taken = False # Reset doesn't count as a turn
                elif event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_pressed = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    shift_pressed = 1
                else:
                    action_taken = False # No relevant key pressed

        if action_taken and not terminated:
            action = [movement, space_pressed, shift_pressed]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

            if terminated:
                print("Game Over! Press 'r' to restart.")
            
            action_taken = False
        
        # We still need to render even if no action is taken (e.g., for cursor animation)
        # In a real agent loop, step() would be called every time.
        # For manual play, we call _get_observation to update the visuals without advancing game state.
        if not action_taken:
            obs = env._get_observation()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate
        
    env.close()