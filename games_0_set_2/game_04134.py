
# Generated: 2025-08-28T01:32:24.908961
# Source Brief: brief_04134.md
# Brief Index: 4134

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a block group. "
        "A group must have at least two adjacent blocks of the same color."
    )

    game_description = (
        "Clear the grid by selecting groups of same-colored blocks. Larger groups give more points. "
        "You have a limited number of moves. Plan ahead to create bigger groups!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 8
    NUM_COLORS = 6
    MAX_MOVES = 30
    SQUARE_SIZE = 40
    GRID_WIDTH = GRID_SIZE * SQUARE_SIZE
    GRID_HEIGHT = GRID_SIZE * SQUARE_SIZE
    GRID_TOP = (SCREEN_HEIGHT - GRID_HEIGHT) // 2
    GRID_LEFT = (SCREEN_WIDTH - GRID_WIDTH) // 2

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_UI_TEXT = (220, 220, 230)
    BLOCK_COLORS = [
        (227, 86, 86),   # Red
        (86, 227, 156),  # Green
        (86, 156, 227),  # Blue
        (227, 227, 86),  # Yellow
        (156, 86, 227),  # Purple
        (227, 156, 86),  # Orange
    ]
    
    # --- Animation ---
    ANIM_DURATION = 15 # frames

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid = []
        self.cursor_pos = [0, 0]
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False

        self.clearing_animation = []
        self.falling_animation = []
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        self.clearing_animation = []
        self.falling_animation = []

        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = self.game_over

        if terminated:
            return self._get_observation(), 0, terminated, False, self._get_info()

        # Handle animations first
        if self.clearing_animation or self.falling_animation:
            self._update_animations()
            if not self.clearing_animation and not self.falling_animation:
                # Animations just finished, check for game over state
                if not self._has_valid_moves():
                    terminated = True
                    self.game_over = True
                    reward = -10  # Loss: no more valid moves
                elif self._is_board_clear():
                    terminated = True
                    self.game_over = True
                    reward = 100 # Win: board cleared
        else:
            # --- Process Player Input ---
            # Restart on rising edge of Shift
            if shift_held and not self.prev_shift_held:
                terminated = True
                self.game_over = True
                # No specific reward for manual restart, handled by external logic
            else:
                self._handle_movement(movement)

                # Select on rising edge of Space
                if space_held and not self.prev_space_held:
                    reward = self._handle_selection()
                    if self.moves_left <= 0 and not self._is_board_clear():
                        terminated = True
                        self.game_over = True
                        reward += -5 # Loss: out of moves
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2: # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
        elif movement == 3: # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4: # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE

    def _handle_selection(self):
        self.moves_left -= 1
        x, y = self.cursor_pos
        
        if self.grid[y][x] == -1:
            return 0 # Selected an empty space

        group = self._find_connected_group(x, y)

        if len(group) < 2:
            # Sound: error_buzz.wav
            return -0.1 # Penalty for selecting isolated block

        # Sound: pop_N.wav where N is len(group)
        self.score += len(group)
        reward = len(group)
        if len(group) > 5:
            bonus = 5
            self.score += bonus
            reward += bonus
        
        for gx, gy in group:
            self.clearing_animation.append({
                "pos": (gx, gy),
                "color_idx": self.grid[gy][gx],
                "timer": self.ANIM_DURATION
            })
            self.grid[gy][gx] = -1
        
        return reward

    def _update_animations(self):
        # Update clearing animation
        if self.clearing_animation:
            for anim in self.clearing_animation:
                anim["timer"] -= 1
            self.clearing_animation = [a for a in self.clearing_animation if a["timer"] > 0]
            
            # If clearing just finished, trigger gravity
            if not self.clearing_animation:
                self._apply_gravity_and_refill()
                # Sound: blocks_fall.wav

        # Update falling animation
        if self.falling_animation:
            for anim in self.falling_animation:
                anim["timer"] -= 1
            self.falling_animation = [a for a in self.falling_animation if a["timer"] > 0]

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_SIZE):
            write_y = self.GRID_SIZE - 1
            for y in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[y][x] != -1:
                    color_idx = self.grid[y][x]
                    if y != write_y:
                        self.grid[write_y][x] = color_idx
                        self.grid[y][x] = -1
                        self.falling_animation.append({
                            "start_y": y,
                            "end_y": write_y,
                            "x": x,
                            "color_idx": color_idx,
                            "timer": self.ANIM_DURATION
                        })
                    write_y -= 1
            
            # Refill empty top cells
            for y in range(write_y, -1, -1):
                new_color = self.np_random.integers(0, self.NUM_COLORS)
                self.grid[y][x] = new_color
                # Add a "falling" animation for new blocks
                self.falling_animation.append({
                    "start_y": y - self.GRID_SIZE, # Start from above the screen
                    "end_y": y,
                    "x": x,
                    "color_idx": new_color,
                    "timer": self.ANIM_DURATION
                })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Grid and Static Blocks ---
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_LEFT + x * self.SQUARE_SIZE,
                    self.GRID_TOP + y * self.SQUARE_SIZE,
                    self.SQUARE_SIZE, self.SQUARE_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
                
                color_idx = self.grid[y][x]
                is_clearing = any(a["pos"] == (x, y) for a in self.clearing_animation)
                is_falling = any(a["end_y"] == y and a["x"] == x for a in self.falling_animation)

                if color_idx != -1 and not is_clearing and not is_falling:
                    self._draw_block(x, y, color_idx)

        # --- Draw Animated Blocks ---
        self._render_clearing_blocks()
        self._render_falling_blocks()

        # --- Draw Cursor and Pre-selection Highlight ---
        self._render_cursor_and_highlight()

    def _render_clearing_blocks(self):
        for anim in self.clearing_animation:
            x, y = anim["pos"]
            progress = anim["timer"] / self.ANIM_DURATION
            size = int(self.SQUARE_SIZE * progress)
            offset = (self.SQUARE_SIZE - size) // 2
            
            rect = pygame.Rect(
                self.GRID_LEFT + x * self.SQUARE_SIZE + offset,
                self.GRID_TOP + y * self.SQUARE_SIZE + offset,
                size, size
            )
            color = self.BLOCK_COLORS[anim["color_idx"]]
            alpha_color = (*color, int(255 * progress))
            
            temp_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surface, alpha_color, temp_surface.get_rect(), border_radius=4)
            self.screen.blit(temp_surface, rect.topleft)

    def _render_falling_blocks(self):
        for anim in self.falling_animation:
            progress = 1.0 - (anim["timer"] / self.ANIM_DURATION)
            # Ease-out interpolation
            eased_progress = 1 - pow(1 - progress, 3)

            current_y_float = anim["start_y"] + (anim["end_y"] - anim["start_y"]) * eased_progress
            
            self._draw_block(anim["x"], current_y_float, anim["color_idx"])

    def _render_cursor_and_highlight(self):
        # Highlight group under cursor if it's a valid move
        if not self.clearing_animation and not self.falling_animation:
            group = self._find_connected_group(self.cursor_pos[0], self.cursor_pos[1])
            if len(group) > 1:
                for x, y in group:
                    highlight_rect = pygame.Rect(
                        self.GRID_LEFT + x * self.SQUARE_SIZE + 2,
                        self.GRID_TOP + y * self.SQUARE_SIZE + 2,
                        self.SQUARE_SIZE - 4, self.SQUARE_SIZE - 4
                    )
                    pygame.draw.rect(self.screen, (255, 255, 255, 60), highlight_rect, border_radius=5)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_LEFT + cursor_x * self.SQUARE_SIZE,
            self.GRID_TOP + cursor_y * self.SQUARE_SIZE,
            self.SQUARE_SIZE, self.SQUARE_SIZE
        )
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 3, border_radius=5)


    def _draw_block(self, grid_x, grid_y, color_idx, size_mod=0):
        if color_idx < 0 or color_idx >= len(self.BLOCK_COLORS):
            return

        rect = pygame.Rect(
            self.GRID_LEFT + grid_x * self.SQUARE_SIZE + 2 + size_mod,
            self.GRID_TOP + grid_y * self.SQUARE_SIZE + 2 + size_mod,
            self.SQUARE_SIZE - 4 - size_mod*2, self.SQUARE_SIZE - 4 - size_mod*2
        )
        color = self.BLOCK_COLORS[color_idx]
        
        # Draw main block
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Draw subtle highlight for 3D effect
        highlight_color = tuple(min(255, c + 30) for c in color)
        pygame.gfxdraw.arc(self.screen, rect.left + 4, rect.top + 4, 3, 180, 270, highlight_color)


    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Moves Left
        moves_text = self.font_large.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(moves_text, moves_rect)
        
        # Game Over Text
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            status = ""
            if self._is_board_clear():
                status = "YOU WIN!"
            elif self.moves_left <= 0:
                status = "OUT OF MOVES"
            else:
                status = "NO MORE MOVES"

            over_text = self.font_large.render(status, True, (255, 255, 255))
            over_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, over_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
        }

    def _generate_board(self):
        while True:
            self.grid = [
                [self.np_random.integers(0, self.NUM_COLORS) for _ in range(self.GRID_SIZE)]
                for _ in range(self.GRID_SIZE)
            ]
            if self._has_valid_moves():
                break

    def _find_connected_group(self, start_x, start_y):
        if self.grid[start_y][start_x] == -1:
            return []
            
        target_color = self.grid[start_y][start_x]
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        group = []

        while q:
            x, y = q.popleft()
            group.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited:
                    if self.grid[ny][nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def _has_valid_moves(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color = self.grid[y][x]
                if color == -1: continue
                # Check right neighbor
                if x < self.GRID_SIZE - 1 and self.grid[y][x+1] == color:
                    return True
                # Check bottom neighbor
                if y < self.GRID_SIZE - 1 and self.grid[y+1][x] == color:
                    return True
        return False

    def _is_board_clear(self):
        return all(self.grid[y][x] == -1 for y in range(self.GRID_SIZE) for x in range(self.GRID_SIZE))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Clear Puzzle")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
    pygame.quit()