
# Generated: 2025-08-27T15:04:38.551954
# Source Brief: brief_00878.md
# Brief Index: 878

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    An isometric match-3 puzzle game where the player swaps adjacent tiles
    to create sets of 3 or more. The goal is to clear the board within a
    limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select/deselect a tile. "
        "Move cursor to an adjacent tile and press Space again to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent gems in an isometric grid to create matches of three or more. "
        "Clear the entire board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 5
    NUM_TILE_TYPES = 5
    INITIAL_MOVES = 20
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (25, 35, 55)
    COLOR_GRID = (45, 55, 75)
    TILE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECTED = (255, 165, 0)
    COLOR_TEXT = (220, 220, 220)

    # --- Isometric Projection ---
    TILE_W = 60
    TILE_H = TILE_W * 0.5
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 100

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.cursor_pos = [0, 0]
        self.selected_pos = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.particle_effects = []
        self.last_swap_feedback = ""
        self.feedback_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_board()
        self.moves_left = self.INITIAL_MOVES
        self.score = 0
        self.steps = 0
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_pos = None
        self.game_over = False
        self.prev_space_held = False
        self.particle_effects = []
        self.last_swap_feedback = ""
        self.feedback_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = self.game_over

        if not terminated:
            self._handle_action(action)
            reward, terminated = self._update_game_state()

        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        if movement != 0:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_SIZE
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_SIZE

        # --- Selection / Swap ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            cursor_tuple = tuple(self.cursor_pos)
            if self.selected_pos is None:
                self.selected_pos = cursor_tuple
            elif self.selected_pos == cursor_tuple:
                self.selected_pos = None
            else:
                # Check for adjacency
                dist = abs(self.selected_pos[0] - cursor_tuple[0]) + abs(self.selected_pos[1] - cursor_tuple[1])
                if dist == 1:
                    # Perform swap
                    self._swap_tiles(self.selected_pos, cursor_tuple)
                    self.selected_pos = None
                else:
                    self.selected_pos = cursor_tuple
        
        self.prev_space_held = space_held

    def _swap_tiles(self, pos1, pos2):
        self.moves_left -= 1
        self.grid[pos1], self.grid[pos2] = self.grid[pos2], self.grid[pos1]
        
        # This swap is now the event that needs to be processed
        self.swap_info = {'pos1': pos1, 'pos2': pos2}

    def _update_game_state(self):
        reward = 0
        terminated = False
        
        if hasattr(self, 'swap_info'):
            pos1, pos2 = self.swap_info['pos1'], self.swap_info['pos2']
            
            total_cleared_tiles, chain_reward = self._process_matches()
            
            if total_cleared_tiles > 0:
                reward += chain_reward
                self.score += int(chain_reward)
                self.last_swap_feedback = f"+{total_cleared_tiles} Clear!"
                self.feedback_timer = 30
            else:
                # No match, swap back
                self.grid[pos1], self.grid[pos2] = self.grid[pos2], self.grid[pos1]
                reward = -0.2
                self.last_swap_feedback = "Invalid Swap"
                self.feedback_timer = 30
            
            del self.swap_info

        # Check for win condition
        if np.all(self.grid == 0):
            reward += 100
            terminated = True
            self.last_swap_feedback = "Board Cleared!"
            self.feedback_timer = 60
        # Check for loss condition
        elif self.moves_left <= 0:
            reward -= 100
            terminated = True
            self.last_swap_feedback = "Out of Moves!"
            self.feedback_timer = 60
            
        return reward, terminated

    def _process_matches(self):
        total_cleared_tiles = 0
        total_reward = 0
        
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            num_cleared = len(matches)
            total_cleared_tiles += num_cleared
            
            # Base reward: +1 per tile
            reward_this_chain = num_cleared
            # Bonus reward: +5 for 4+ tiles
            if num_cleared >= 4:
                reward_this_chain += 5
            
            total_reward += reward_this_chain
            
            for r, c in matches:
                # Sound effect placeholder: # sfx_clear_tile()
                self._create_particles(r, c, self.grid[r, c])
                self.grid[r, c] = 0 # Mark as empty
            
            self._apply_gravity_and_refill()
            
        return total_cleared_tiles, total_reward

    def _find_matches(self):
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    continue
                # Horizontal match
                if c < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    matches.add((r, c)); matches.add((r, c+1)); matches.add((r, c+2))
                # Vertical match
                if r < self.GRID_SIZE - 2 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    matches.add((r, c)); matches.add((r+1, c)); matches.add((r+2, c))
        return matches

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_SIZE):
            empty_r = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[empty_r, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_r, c]
                    empty_r -= 1
            # Refill
            for r in range(empty_r, -1, -1):
                self.grid[r, c] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)

    def _generate_board(self):
        while True:
            grid = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            # Ensure no initial matches
            while self._find_matches_on_grid(grid):
                matches = self._find_matches_on_grid(grid)
                for r, c in matches:
                    grid[r,c] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
            # Ensure at least one move is possible
            if self._check_for_possible_moves(grid):
                return grid

    def _check_for_possible_moves(self, grid):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Swap right
                if c < self.GRID_SIZE - 1:
                    grid[r, c], grid[r, c+1] = grid[r, c+1], grid[r, c]
                    if self._find_matches_on_grid(grid):
                        grid[r, c], grid[r, c+1] = grid[r, c+1], grid[r, c]
                        return True
                    grid[r, c], grid[r, c+1] = grid[r, c+1], grid[r, c] # Swap back
                # Swap down
                if r < self.GRID_SIZE - 1:
                    grid[r, c], grid[r+1, c] = grid[r+1, c], grid[r, c]
                    if self._find_matches_on_grid(grid):
                        grid[r, c], grid[r+1, c] = grid[r+1, c], grid[r, c]
                        return True
                    grid[r, c], grid[r+1, c] = grid[r+1, c], grid[r, c] # Swap back
        return False

    def _find_matches_on_grid(self, grid):
        # A non-mutating version of _find_matches for board generation
        matches = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if grid[r, c] == 0: continue
                if c < self.GRID_SIZE - 2 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                if r < self.GRID_SIZE - 2 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return matches

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
            "cursor_pos": list(self.cursor_pos),
            "selected_pos": self.selected_pos,
        }

    def _iso_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * (self.TILE_W / 2)
        y = self.ORIGIN_Y + (c + r) * (self.TILE_H / 2)
        return int(x), int(y)

    def _draw_iso_tile(self, surface, r, c, color, highlight_color=None, highlight_width=2):
        x, y = self._iso_to_screen(r, c)
        
        points = [
            (x, y),
            (x + self.TILE_W / 2, y + self.TILE_H / 2),
            (x, y + self.TILE_H),
            (x - self.TILE_W / 2, y + self.TILE_H / 2)
        ]

        # Darken color for sides
        side_color = tuple(max(0, val - 40) for val in color)
        
        # Draw sides
        pygame.gfxdraw.filled_polygon(surface, [points[3], points[2], (points[2][0], points[2][1] + 10), (points[3][0], points[3][1] + 10)], side_color)
        pygame.gfxdraw.filled_polygon(surface, [points[2], points[1], (points[1][0], points[1][1] + 10), (points[2][0], points[2][1] + 10)], side_color)

        # Draw top
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

        if highlight_color:
            for i in range(highlight_width):
                h_points = [
                    (x, y-i),
                    (x + self.TILE_W / 2 + i, y + self.TILE_H / 2),
                    (x, y + self.TILE_H + i),
                    (x - self.TILE_W / 2 - i, y + self.TILE_H / 2)
                ]
                pygame.gfxdraw.aapolygon(surface, h_points, highlight_color)

    def _render_game(self):
        # Draw grid base
        for r in range(self.GRID_SIZE + 1):
            start = self._iso_to_screen(r - 0.5, -0.5)
            end = self._iso_to_screen(r - 0.5, self.GRID_SIZE - 0.5)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for c in range(self.GRID_SIZE + 1):
            start = self._iso_to_screen(-0.5, c - 0.5)
            end = self._iso_to_screen(self.GRID_SIZE - 0.5, c - 0.5)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_type = self.grid[r, c]
                if tile_type > 0:
                    color = self.TILE_COLORS[tile_type - 1]
                    self._draw_iso_tile(self.screen, r, c, color)
        
        # Draw cursor and selection
        cursor_r, cursor_c = self.cursor_pos
        self._draw_iso_tile(self.screen, cursor_r, cursor_c, (0,0,0,0), self.COLOR_CURSOR, 3)

        if self.selected_pos:
            sel_r, sel_c = self.selected_pos
            tile_type = self.grid[sel_r, sel_c]
            if tile_type > 0:
                 color = self.TILE_COLORS[tile_type - 1]
                 self._draw_iso_tile(self.screen, sel_r, sel_c, color, self.COLOR_SELECTED, 3)

        # Update and draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Score
        score_surf = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Moves
        moves_surf = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (self.SCREEN_WIDTH - moves_surf.get_width() - 20, 10))
        
        # Feedback text
        if self.feedback_timer > 0:
            feedback_surf = self.font_small.render(self.last_swap_feedback, True, self.COLOR_SELECTED)
            pos_x = (self.SCREEN_WIDTH - feedback_surf.get_width()) // 2
            self.screen.blit(feedback_surf, (pos_x, self.SCREEN_HEIGHT - 40))
            self.feedback_timer -= 1

    def _create_particles(self, r, c, tile_type):
        center_x, center_y = self._iso_to_screen(r, c)
        center_y += self.TILE_H / 2
        color = self.TILE_COLORS[tile_type - 1]
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particle_effects.append([center_x, center_y, vx, vy, 30, color])

    def _update_and_draw_particles(self):
        remaining_particles = []
        for p in self.particle_effects:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # lifetime--
            if p[4] > 0:
                remaining_particles.append(p)
                alpha = max(0, min(255, int(p[4] * (255/30))))
                size = max(1, int(p[4] / 10))
                color = p[5] + (alpha,)
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (int(p[0]) - size, int(p[1]) - size))
        self.particle_effects = remaining_particles

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part requires a window, so it won't run in a headless environment.
    # To run, you might need to comment out SDL_VIDEODRIVER="dummy" if set elsewhere.
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("IsoMatch Environment")
        
        obs, info = env.reset()
        terminated = False
        
        # Game loop
        running = True
        while running:
            movement = 0 # No-op
            space_held = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = True
            
            # For manual play, we want to react to key presses, not holds
            # so we only send an action if a key is pressed.
            action = [movement, 1 if space_held else 0, 0]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Draw the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
                pygame.time.wait(3000)
                obs, info = env.reset()

            env.clock.tick(30) # Limit frame rate for manual play
            
    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("This error is expected in a headless environment. The core Gym environment is still functional.")
    finally:
        env.close()