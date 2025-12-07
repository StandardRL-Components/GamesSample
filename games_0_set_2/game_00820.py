
# Generated: 2025-08-27T14:53:45.454164
# Source Brief: brief_00820.md
# Brief Index: 820

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to swap the selected tile "
        "with the tile in the direction you last moved. Hold shift to forfeit and end the game."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A match-3 puzzle game. Swap adjacent tiles to create lines of 3 or more of the same color. "
        "Clear the entire board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_TILE_TYPES = 6
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.TILE_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (40, 45, 50)
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # Calculate tile rendering properties
        self.board_rect = pygame.Rect(
            (self.WIDTH - self.HEIGHT) // 2, 0, self.HEIGHT, self.HEIGHT
        )
        self.TILE_SIZE = self.board_rect.width // self.GRID_SIZE
        self.GRID_OFFSET_X = self.board_rect.left
        self.GRID_OFFSET_Y = self.board_rect.top

        # Initialize state variables
        self.rng = None
        self.grid = None
        self.cursor_pos = None
        self.last_move_dir = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        
        self.reset()
        
        # This check is for development and ensures compliance.
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.grid = self._generate_board()
        self.cursor_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.last_move_dir = (0, -1)  # Default to Up
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        terminated = False
        
        # Action 1: Handle Shift to forfeit
        if shift_held:
            terminated = True
            reward = -50  # Penalty for forfeiting
        else:
            # Action 2: Handle cursor movement
            self._handle_movement(movement)

            # Action 3: Handle swap attempt
            if space_held:
                reward, terminated = self._handle_swap()

        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        """Updates cursor position based on movement action."""
        x, y = self.cursor_pos
        if movement == 1:  # Up
            y -= 1
            self.last_move_dir = (0, -1)
        elif movement == 2:  # Down
            y += 1
            self.last_move_dir = (0, 1)
        elif movement == 3:  # Left
            x -= 1
            self.last_move_dir = (-1, 0)
        elif movement == 4:  # Right
            x += 1
            self.last_move_dir = (1, 0)
        
        # Wrap around grid
        self.cursor_pos = (x % self.GRID_SIZE, y % self.GRID_SIZE)

    def _handle_swap(self):
        """Processes a swap action, resolves matches, and calculates rewards."""
        if self.moves_left <= 0:
            return 0.0, self.game_over

        self.moves_left -= 1
        
        p1_r, p1_c = self.cursor_pos
        p2_r, p2_c = ((p1_r + self.last_move_dir[1]), (p1_c + self.last_move_dir[0]))
        
        # Ensure target is within bounds (no wrapping for swaps)
        if not (0 <= p2_r < self.GRID_SIZE and 0 <= p2_c < self.GRID_SIZE):
            self.moves_left += 1 # No move consumed for invalid swap location
            return 0.0, False

        # Perform the swap in the data grid
        self.grid[p1_r, p1_c], self.grid[p2_r, p2_c] = self.grid[p2_r, p2_c], self.grid[p1_r, p1_c]
        # sfx: swap_sound
        
        total_reward, tiles_cleared = self._resolve_board()

        if tiles_cleared == 0:
            # Invalid move, swap back. Move is still consumed.
            self.grid[p1_r, p1_c], self.grid[p2_r, p2_c] = self.grid[p2_r, p2_c], self.grid[p1_r, p1_c]
            # sfx: invalid_swap_sound
            reward = -0.1
            terminated = self.moves_left <= 0
            if terminated: reward -= 50
            return reward, terminated

        # Check for win condition
        if np.all(self.grid == -1):
            total_reward += 100
            return total_reward, True

        # Check for loss condition
        if self.moves_left <= 0:
            total_reward -= 50
            return total_reward, True

        return total_reward, False

    def _resolve_board(self):
        """Repeatedly finds matches, clears them, and applies gravity until stable."""
        total_reward = 0
        total_tiles_cleared = 0
        
        while True:
            matches = self._find_matches()
            if not matches:
                break

            num_cleared = len(matches)
            total_tiles_cleared += num_cleared
            
            # Base reward
            reward_this_wave = num_cleared
            
            # Bonus rewards
            match_groups = self._group_matches(matches)
            for group in match_groups:
                if len(group) == 4:
                    reward_this_wave += 5
                elif len(group) >= 5:
                    reward_this_wave += 10
            
            total_reward += reward_this_wave
            
            # sfx: match_clear_sound
            for r, c in matches:
                if self.grid[r, c] != -1:
                    self._add_particles(r, c, self.TILE_COLORS[self.grid[r,c]])
                    self.grid[r, c] = -1  # Mark for removal
            
            self._apply_gravity_and_refill()
            # sfx: tiles_fall_sound
            
        return total_reward, total_tiles_cleared

    def _generate_board(self):
        """Generates a 10x10 board with no initial matches."""
        while True:
            grid = self.rng.integers(0, self.NUM_TILE_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_matches(grid):
                return grid

    def _find_matches(self, grid=None):
        """Finds all horizontal and vertical matches of 3 or more."""
        if grid is None:
            grid = self.grid
        
        matched_tiles = set()
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if grid[r, c] == -1: continue
                # Check horizontal
                if c < self.GRID_SIZE - 2 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    match_len = 3
                    while c + match_len < self.GRID_SIZE and grid[r, c] == grid[r, c+match_len]:
                        match_len += 1
                    for i in range(match_len):
                        matched_tiles.add((r, c + i))
                # Check vertical
                if r < self.GRID_SIZE - 2 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    match_len = 3
                    while r + match_len < self.GRID_SIZE and grid[r, c] == grid[r+match_len, c]:
                        match_len += 1
                    for i in range(match_len):
                        matched_tiles.add((r + i, c))
                        
        return matched_tiles

    def _group_matches(self, matches):
        """Groups a set of matched coordinates into contiguous lines."""
        groups = []
        matches_to_process = set(matches)
        while matches_to_process:
            start_node = matches_to_process.pop()
            q = [start_node]
            group = {start_node}
            head = 0
            while head < len(q):
                r, c = q[head]
                head += 1
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (r + dr, c + dc)
                    if neighbor in matches_to_process:
                        matches_to_process.remove(neighbor)
                        group.add(neighbor)
                        q.append(neighbor)
            groups.append(group)
        return groups

    def _apply_gravity_and_refill(self):
        """Shifts tiles down to fill empty spaces and adds new tiles at the top."""
        for c in range(self.GRID_SIZE):
            write_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    self.grid[write_row, c], self.grid[r, c] = self.grid[r, c], self.grid[write_row, c]
                    write_row -= 1
            
            for r in range(write_row, -1, -1):
                self.grid[r, c] = self.rng.integers(0, self.NUM_TILE_TYPES)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_render_particles()
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

    def _render_game(self):
        """Renders the grid, tiles, and cursor."""
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_x = self.GRID_OFFSET_X + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.GRID_OFFSET_Y), 
                             (start_x, self.GRID_OFFSET_Y + self.GRID_SIZE * self.TILE_SIZE))
            # Horizontal
            start_y = self.GRID_OFFSET_Y + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, start_y),
                             (self.GRID_OFFSET_X + self.GRID_SIZE * self.TILE_SIZE, start_y))

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_val = self.grid[r, c]
                if tile_val != -1:
                    color = self.TILE_COLORS[tile_val]
                    rect = pygame.Rect(
                        self.GRID_OFFSET_X + c * self.TILE_SIZE,
                        self.GRID_OFFSET_Y + r * self.TILE_SIZE,
                        self.TILE_SIZE, self.TILE_SIZE
                    )
                    
                    # Beveled effect
                    pygame.draw.rect(self.screen, color, rect.inflate(-2, -2))
                    brighter_color = tuple(min(255, x + 40) for x in color)
                    pygame.draw.rect(self.screen, brighter_color, rect.inflate(-8, -8))

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cursor_c * self.TILE_SIZE,
            self.GRID_OFFSET_Y + cursor_r * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        # Pulsating alpha for cursor glow
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        glow_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_CURSOR, alpha), (0, 0, self.TILE_SIZE, self.TILE_SIZE), border_radius=4)
        self.screen.blit(glow_surface, cursor_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)

    def _render_ui(self):
        """Renders the score, moves left, and game over messages."""
        # UI Panel on the left
        ui_panel = pygame.Rect(10, 10, self.GRID_OFFSET_X - 20, self.HEIGHT - 20)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_panel, border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_GRID, ui_panel, 2, border_radius=10)

        # Render Moves Left
        moves_text_surf = self.font_ui.render("Moves", True, self.COLOR_TEXT)
        moves_val_surf = self.font_ui.render(str(self.moves_left), True, self.COLOR_CURSOR)
        self.screen.blit(moves_text_surf, (ui_panel.centerx - moves_text_surf.get_width() // 2, 30))
        self.screen.blit(moves_val_surf, (ui_panel.centerx - moves_val_surf.get_width() // 2, 60))

        # Render Score
        score_text_surf = self.font_ui.render("Score", True, self.COLOR_TEXT)
        score_val_surf = self.font_ui.render(str(int(self.score)), True, self.COLOR_CURSOR)
        self.screen.blit(score_text_surf, (ui_panel.centerx - score_text_surf.get_width() // 2, 120))
        self.screen.blit(score_val_surf, (ui_panel.centerx - score_val_surf.get_width() // 2, 150))

        # Render Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            is_win = np.all(self.grid == -1)
            msg = "YOU WIN!" if is_win else "GAME OVER"
            color = self.TILE_COLORS[1] if is_win else self.TILE_COLORS[0]
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text_surf, text_rect)

    def _add_particles(self, r, c, color):
        """Creates a burst of particles at a given grid location."""
        center_x = self.GRID_OFFSET_X + c * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.GRID_OFFSET_Y + r * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(20, 40)
            radius = random.randint(3, 6)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color, 'radius': radius})

    def _update_and_render_particles(self):
        """Updates particle positions, lifespan, and renders them."""
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            # Fade out effect
            alpha = int(255 * (p['life'] / p['max_life']))
            color_with_alpha = (*p['color'], alpha)
            
            # Using gfxdraw for antialiased circles
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color_with_alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color_with_alpha)

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
    done = False
    
    # Key mapping for human play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame window for rendering
    pygame.display.set_caption("Match-3 Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Default action is no-op
        action = [0, 0, 0] # move, space, shift

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Check movement keys
        for key, move_action in key_map.items():
            if keys[key]:
                action[0] = move_action
                break # only one movement at a time

        # Check action keys
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_left']}")
        
        if terminated:
            print("Game Over!")
            print(f"Final Score: {info['score']:.2f}")
            # Render one last time to show the final state
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Wait for a moment before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()

        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit to 30 FPS for human play

    pygame.quit()