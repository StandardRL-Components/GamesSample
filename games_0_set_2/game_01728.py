
# Generated: 2025-08-27T18:05:52.204061
# Source Brief: brief_01728.md
# Brief Index: 1728

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to match the selected tile group. Use Shift to reshuffle the board (costs 1 move)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match 3 or more adjacent colored tiles to score points. Reach 1000 points to win, but you only have 20 moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        self.GRID_SIZE = 8
        self.NUM_COLORS = 5
        self.WIN_SCORE = 1000
        self.MAX_MOVES = 20
        self.MIN_MATCH_SIZE = 3

        # --- Visual Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 42
        self.TILE_MARGIN = 4
        self.GRID_WIDTH = self.GRID_SIZE * (self.TILE_SIZE + self.TILE_MARGIN) - self.TILE_MARGIN
        self.GRID_HEIGHT = self.GRID_SIZE * (self.TILE_SIZE + self.TILE_MARGIN) - self.TILE_MARGIN
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 30

        self.COLORS = [
            (230, 57, 70),   # Red
            (66, 186, 150),  # Green
            (57, 133, 230),  # Blue
            (244, 204, 91),  # Yellow
            (142, 68, 173)   # Purple
        ]
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_BG = (35, 40, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_VALUE = (255, 255, 255)
        
        self.font_main = pygame.font.SysFont("tahoma", 24, bold=True)
        self.font_small = pygame.font.SysFont("tahoma", 18)

        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.steps = None
        self.particles = []
        self.last_action_was_major = False # To prevent holding down space/shift from using all moves

        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.steps = 0
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.particles = []
        self.last_action_was_major = True # Prevent action on first frame

        self._initialize_board()
        while not self._has_possible_moves():
            self._initialize_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_val, shift_val = action
        space_held = space_val == 1
        shift_held = shift_val == 1
        
        reward = 0
        move_consumed = False

        # --- Action Handling Precedence ---
        # A "major" action (match or reshuffle) can only be taken if the previous action was not major.
        # This forces the agent to "release" the button.
        is_trying_major_action = space_held or shift_held
        
        if is_trying_major_action and not self.last_action_was_major:
            if shift_held:
                # --- Reshuffle Action ---
                self._reshuffle_board()
                # sound: reshuffle_sound.play()
                move_consumed = True
            elif space_held:
                # --- Match Action ---
                match_reward, match_made = self._attempt_match()
                if match_made:
                    reward += match_reward
                    move_consumed = True
                    # sound: match_success_sound.play()
                else:
                    # sound: match_fail_sound.play()
                    pass
            self.last_action_was_major = True
        elif not is_trying_major_action:
            self.last_action_was_major = False
            if movement > 0:
                # --- Cursor Movement ---
                self._move_cursor(movement)
                # sound: cursor_move_sound.play()

        if move_consumed:
            self.moves_left -= 1
        
        # --- Update Game State ---
        self.steps += 1
        self._update_particles()
        
        # --- Check for Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100 # Win bonus
            terminated = True
            # sound: win_game_sound.play()
        elif self.moves_left <= 0:
            terminated = True
            # sound: lose_game_sound.play()
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _initialize_board(self):
        self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))

    def _has_possible_moves(self):
        visited = np.zeros_like(self.grid, dtype=bool)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if not visited[r, c]:
                    group = self._find_match_group(c, r)
                    if len(group) >= self.MIN_MATCH_SIZE:
                        return True
                    for gr, gc in group:
                        visited[gr, gc] = True
        return False

    def _find_match_group(self, start_x, start_y):
        if not (0 <= start_x < self.GRID_SIZE and 0 <= start_y < self.GRID_SIZE):
            return set()
        
        target_color = self.grid[start_y, start_x]
        if target_color == -1: # Empty tile
            return set()

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        
        while q:
            x, y = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and \
                   (nx, ny) not in visited and self.grid[ny, nx] == target_color:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return visited

    def _attempt_match(self):
        cx, cy = self.cursor_pos
        match_group = self._find_match_group(cx, cy)
        
        if len(match_group) < self.MIN_MATCH_SIZE:
            return 0, False

        reward = len(match_group)
        if len(match_group) == 4:
            reward += 10
        elif len(match_group) >= 5:
            reward += 20
        
        self.score += reward
        
        for x, y in match_group:
            self._create_particles(x, y, self.grid[y, x])
            self.grid[y, x] = -1 # Mark as empty
        
        self._apply_gravity()
        
        # After gravity, check if the board is now stuck.
        if not self._has_possible_moves():
            self._reshuffle_board()
            
        return reward, True

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
            
            for r in range(empty_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(0, self.NUM_COLORS)

    def _reshuffle_board(self):
        flat_grid = self.grid.flatten().tolist()
        self.np_random.shuffle(flat_grid)
        self.grid = np.array(flat_grid).reshape((self.GRID_SIZE, self.GRID_SIZE))
        
        # Ensure the new board is solvable
        if not self._has_possible_moves():
            self._initialize_board() # Last resort, create a fresh board
            while not self._has_possible_moves():
                self._initialize_board()

    def _move_cursor(self, direction):
        # 1=up, 2=down, 3=left, 4=right
        if direction == 1: self.cursor_pos[1] -= 1
        elif direction == 2: self.cursor_pos[1] += 1
        elif direction == 3: self.cursor_pos[0] -= 1
        elif direction == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

    def _create_particles(self, grid_x, grid_y, color_index):
        px = self.GRID_X + grid_x * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_SIZE / 2
        py = self.GRID_Y + grid_y * (self.TILE_SIZE + self.TILE_MARGIN) + self.TILE_SIZE / 2
        color = self.COLORS[color_index]
        
        for _ in range(15): # Create 15 particles per tile
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background
        grid_bg_rect = pygame.Rect(self.GRID_X - 10, self.GRID_Y - 10, self.GRID_WIDTH + 20, self.GRID_HEIGHT + 20)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_bg_rect, border_radius=10)

        # Draw tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.grid[r, c]
                if color_index == -1:
                    continue
                
                tile_color = self.COLORS[color_index]
                tile_rect = pygame.Rect(
                    self.GRID_X + c * (self.TILE_SIZE + self.TILE_MARGIN),
                    self.GRID_Y + r * (self.TILE_SIZE + self.TILE_MARGIN),
                    self.TILE_SIZE,
                    self.TILE_SIZE
                )
                pygame.draw.rect(self.screen, tile_color, tile_rect, border_radius=8)
                
                # Add a subtle highlight for depth
                highlight_color = tuple(min(255, val + 30) for val in tile_color)
                pygame.draw.rect(self.screen, highlight_color, (tile_rect.x+3, tile_rect.y+3, tile_rect.width-6, tile_rect.height-10), border_radius=6)

        # Draw particles
        for p in self.particles:
            size = max(1, p['life'] / 6)
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(size))
            
        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X + cursor_x * (self.TILE_SIZE + self.TILE_MARGIN) - 4,
            self.GRID_Y + cursor_y * (self.TILE_SIZE + self.TILE_MARGIN) - 4,
            self.TILE_SIZE + 8,
            self.TILE_SIZE + 8
        )
        
        # Pulsing effect for cursor
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        cursor_surface = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), (0, 0, *cursor_rect.size), 4, border_radius=12)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render("SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_main.render(f"{self.score}", True, self.COLOR_UI_VALUE)
        self.screen.blit(score_text, (self.WIDTH // 4 - score_text.get_width() // 2, 10))
        self.screen.blit(score_val, (self.WIDTH // 4 - score_val.get_width() // 2, 40))

        # Moves display
        moves_text = self.font_main.render("MOVES", True, self.COLOR_UI_TEXT)
        moves_val = self.font_main.render(f"{self.moves_left}", True, self.COLOR_UI_VALUE)
        self.screen.blit(moves_text, (self.WIDTH * 3 // 4 - moves_text.get_width() // 2, 10))
        self.screen.blit(moves_val, (self.WIDTH * 3 // 4 - moves_val.get_width() // 2, 40))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            end_msg = self.font_main.render(win_text, True, self.COLOR_UI_VALUE)
            end_rect = end_msg.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_msg, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
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

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This setup allows a human to play the game.
    obs, info = env.reset()
    done = False
    
    # Use a real screen for manual play
    real_screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Tile Matcher")
    
    running = True
    while running:
        movement = 0 # none
        space = 0    # released
        shift = 0    # released
        
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
        
        # The environment step is only called when there is an action from the user
        # or if you want to advance the game state without user input.
        # Since auto_advance is False, we only step when an action is taken.
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward > 0:
            print(f"Reward: {reward}, Score: {info['score']}, Moves Left: {info['moves_left']}")
        
        if terminated or truncated:
            print("Game Over!")
            print(f"Final Score: {info['score']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for manual play

    env.close()