
# Generated: 2025-08-27T15:19:55.947808
# Source Brief: brief_00960.md
# Brief Index: 960

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a tile, then move to an adjacent tile and press space again to swap. Press shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more of the same color. Clear the board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_COLORS = 5
        self.INITIAL_MOVES = 20
        self.MAX_STEPS = 1000

        # Visuals
        self.TILE_SIZE = 40
        self.BOARD_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.TILE_SIZE) // 2
        self.BOARD_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.TILE_SIZE) // 2
        self.TILE_BORDER_RADIUS = 8
        self.PARTICLE_LIFESPAN = 15

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.GRID_COLOR = (40, 45, 55)
        self.TILE_COLORS = [
            (230, 50, 50),   # Red
            (50, 200, 50),   # Green
            (60, 130, 255),  # Blue
            (240, 220, 80),  # Yellow
            (180, 80, 240),  # Purple
        ]
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECT = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables
        self.board = None
        self.cursor_pos = None
        self.selected_tile = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.steps = None
        self.last_match_details = {}
        self.particles = []
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.game_over = False
        self.steps = 0
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_tile = None
        self.last_match_details = {}
        self.particles = []

        while True:
            self._generate_board()
            if self._find_possible_moves():
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.last_match_details = {}

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._move_cursor(movement)

        if shift_held and self.selected_tile:
            self.selected_tile = None
        elif space_held:
            current_tile_val = self.board[self.cursor_pos[0], self.cursor_pos[1]]
            if not self.selected_tile:
                if current_tile_val != 0:
                    self.selected_tile = tuple(self.cursor_pos)
            else:
                if self._is_adjacent(self.cursor_pos, self.selected_tile) and current_tile_val != 0:
                    reward = self._handle_swap()
                    self.moves_left -= 1
                self.selected_tile = None

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if np.all(self.board == 0): # Win
                reward += 100
            elif self.moves_left <= 0: # Loss
                reward -= 50

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _handle_swap(self):
        p1 = self.selected_tile
        p2 = tuple(self.cursor_pos)
        self.board[p1], self.board[p2] = self.board[p2], self.board[p1]

        total_reward = 0
        chain_count = 0
        
        while True:
            matches = self._find_matches()
            if not matches:
                if chain_count == 0: # Initial swap resulted in no match
                    return -0.1
                else: # Chain reaction ended
                    break
            
            # --- Match found ---
            chain_count += 1
            num_matched = len(matches)

            # Calculate reward
            match_reward = num_matched
            if num_matched > 3:
                match_reward += 5
            if chain_count > 1: # Chain reaction bonus
                match_reward += 10
            
            total_reward += match_reward
            self.score += int(match_reward)

            # Spawn particles for visual effect
            # sound: gem_shatter.wav
            for r, c in matches:
                self._spawn_particles(r, c, self.board[r, c])

            # Update board state
            for r, c in matches:
                self.board[r, c] = 0
            
            self._apply_gravity()
            self._refill_board()

            if np.all(self.board == 0):
                break
        
        return total_reward

    def _generate_board(self):
        self.board = self.rng.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        while True:
            matches = self._find_matches()
            if not matches:
                break
            for r, c in matches:
                self.board[r, c] = self.rng.integers(1, self.NUM_COLORS + 1)

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.board[r, c] != 0 and self.board[r, c] == self.board[r, c+1] == self.board[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.board[r, c] != 0 and self.board[r, c] == self.board[r+1, c] == self.board[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

    def _find_possible_moves(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                temp_board = np.copy(self.board)
                # Try swapping right
                if c < self.GRID_SIZE - 1:
                    temp_board[r,c], temp_board[r,c+1] = temp_board[r,c+1], temp_board[r,c]
                    if self._find_matches_on_board(temp_board): return True
                    temp_board[r,c], temp_board[r,c+1] = temp_board[r,c+1], temp_board[r,c] # Swap back
                # Try swapping down
                if r < self.GRID_SIZE - 1:
                    temp_board[r,c], temp_board[r+1,c] = temp_board[r+1,c], temp_board[r,c]
                    if self._find_matches_on_board(temp_board): return True
        return False
    
    def _find_matches_on_board(self, board):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if board[r, c] != 0 and board[r, c] == board[r, c+1] == board[r, c+2]:
                    return True
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if board[r, c] != 0 and board[r, c] == board[r+1, c] == board[r+2, c]:
                    return True
        return False

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.board[r, c] != 0:
                    self.board[empty_row, c], self.board[r, c] = self.board[r, c], self.board[empty_row, c]
                    empty_row -= 1

    def _refill_board(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.board[r, c] == 0:
                    self.board[r, c] = self.rng.integers(1, self.NUM_COLORS + 1)
                    # sound: gem_fall.wav

    def _move_cursor(self, movement):
        if movement == 1: self.cursor_pos[0] -= 1  # Up
        elif movement == 2: self.cursor_pos[0] += 1  # Down
        elif movement == 3: self.cursor_pos[1] -= 1  # Left
        elif movement == 4: self.cursor_pos[1] += 1  # Right
        self.cursor_pos[0] = self.cursor_pos[0] % self.GRID_SIZE
        self.cursor_pos[1] = self.cursor_pos[1] % self.GRID_SIZE

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _check_termination(self):
        if self.moves_left <= 0:
            return True
        if np.all(self.board == 0):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        # Check for no more possible moves
        if self.moves_left > 0 and not self._find_possible_moves():
            # In a real game, we might shuffle the board. Here, we end the game.
            return True
        return False

    def _render_game(self):
        # Draw grid background
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(self.BOARD_OFFSET_X + c * self.TILE_SIZE,
                                   self.BOARD_OFFSET_Y + r * self.TILE_SIZE,
                                   self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.GRID_COLOR, rect)

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.board[r, c]
                if color_index != 0:
                    self._draw_tile(self.screen, r, c, color_index)

        # Draw selection and cursor
        self._draw_selection_indicators()
        
        # Update and draw particles
        self._update_and_draw_particles()

    def _draw_tile(self, surface, r, c, color_index):
        tile_rect = pygame.Rect(
            self.BOARD_OFFSET_X + c * self.TILE_SIZE + 2,
            self.BOARD_OFFSET_Y + r * self.TILE_SIZE + 2,
            self.TILE_SIZE - 4, self.TILE_SIZE - 4
        )
        base_color = self.TILE_COLORS[color_index - 1]
        shadow_color = tuple(max(0, val - 40) for val in base_color)
        highlight_color = tuple(min(255, val + 40) for val in base_color)

        # Draw 3D effect
        pygame.draw.rect(surface, shadow_color, tile_rect, border_radius=self.TILE_BORDER_RADIUS)
        inner_rect = tile_rect.inflate(-4, -4)
        inner_rect.topleft = (tile_rect.left + 2, tile_rect.top + 1)
        pygame.draw.rect(surface, base_color, inner_rect, border_radius=self.TILE_BORDER_RADIUS - 2)

        # Draw highlight
        pygame.gfxdraw.arc(surface, inner_rect.centerx, inner_rect.centery, inner_rect.width//2-2, 120, 240, highlight_color)

    def _draw_selection_indicators(self):
        # Draw selected tile indicator
        if self.selected_tile:
            r, c = self.selected_tile
            rect = pygame.Rect(
                self.BOARD_OFFSET_X + c * self.TILE_SIZE,
                self.BOARD_OFFSET_Y + r * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            alpha = int(100 + 155 * pulse)
            
            overlay_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(overlay_surf, self.COLOR_SELECT + (alpha,), 
                             (0, 0, self.TILE_SIZE, self.TILE_SIZE), 
                             border_radius=self.TILE_BORDER_RADIUS, width=4)
            self.screen.blit(overlay_surf, rect.topleft)

        # Draw cursor
        r, c = self.cursor_pos
        rect = pygame.Rect(
            self.BOARD_OFFSET_X + c * self.TILE_SIZE,
            self.BOARD_OFFSET_Y + r * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, border_radius=self.TILE_BORDER_RADIUS, width=3)
    
    def _spawn_particles(self, r, c, color_index):
        px = self.BOARD_OFFSET_X + c * self.TILE_SIZE + self.TILE_SIZE / 2
        py = self.BOARD_OFFSET_Y + r * self.TILE_SIZE + self.TILE_SIZE / 2
        color = self.TILE_COLORS[color_index - 1]
        for _ in range(20): # Spawn 20 particles
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * 2 + 1
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            size = self.rng.integers(2, 5)
            self.particles.append([px, py, vx, vy, size, self.PARTICLE_LIFESPAN, color])

    def _update_and_draw_particles(self):
        # Since auto_advance is False, particles only update/move on a step call
        # This is a compromise: they will appear static until the next action.
        # For a better effect with auto_advance=False, one might have an internal
        # animation loop within step(), but that complicates the gym API.
        # Here we just render them at their initial spawn location.
        for p in self.particles:
            pygame.draw.circle(self.screen, p[6], (int(p[0]), int(p[1])), p[4])
        self.particles.clear() # Clear after one frame render

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (20, 20), self.font_large)

        # Moves
        moves_text = f"MOVES: {self.moves_left}"
        text_surf = self.font_large.render(moves_text, True, self.COLOR_TEXT)
        self._draw_text(moves_text, (self.WIDTH - text_surf.get_width() - 20, 20), self.font_large)

        if self.game_over:
            if np.all(self.board == 0):
                end_text = "BOARD CLEARED!"
            else:
                end_text = "GAME OVER"
            
            center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2 - 80
            self._draw_text(end_text, (center_x, center_y), self.font_large, center=True)

    def _draw_text(self, text, pos, font, center=False):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, self.COLOR_TEXT)
        
        if center:
            text_rect = text_surf.get_rect(center=pos)
            shadow_rect = shadow_surf.get_rect(center=(pos[0]+2, pos[1]+2))
        else:
            text_rect = text_surf.get_rect(topleft=pos)
            shadow_rect = shadow_surf.get_rect(topleft=(pos[0]+2, pos[1]+2))
            
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        action = [0, 0, 0] # Default action: no-op
        
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
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
                    done = False
                    continue
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    
        # If a key was pressed, step the environment
        if any(a != 0 for a in action):
            if not done:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        # Rendering
        # The environment's observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        
        # We need a display to see the game
        display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Match-3 Gym Environment")
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS

    pygame.quit()