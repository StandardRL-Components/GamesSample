
# Generated: 2025-08-27T22:06:55.922944
# Source Brief: brief_03014.md
# Brief Index: 3014

        
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
        "Controls: Arrow keys to move selector. Space to match tiles. Shift to restart."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match 3 or more adjacent tiles of the same color to clear them. Clear the board or get the highest score in 20 moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 8
    TILE_SIZE = 40
    MAX_MOVES = 20
    MAX_STEPS = 1000 # Failsafe termination

    # Colors
    COLOR_BG = (40, 42, 54)
    COLOR_GRID = (68, 71, 90)
    COLOR_TEXT = (248, 248, 242)
    TILE_COLORS = [
        (255, 85, 85),   # Red
        (80, 250, 123),  # Green
        (139, 233, 253), # Cyan
        (255, 184, 108), # Orange
        (189, 147, 249), # Purple
    ]
    NUM_COLORS = len(TILE_COLORS)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
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
        
        self.grid_offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.TILE_SIZE) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.TILE_SIZE) // 2
        
        # State machine for animations in a turn-based system
        self.game_state = 'playing' # 'playing', 'clearing', 'falling', 'game_over'
        self.animation_timer = 0
        self.tiles_to_clear = []
        self.tiles_to_fall = {}

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.game_state = 'playing'
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.animation_timer = 0
        self.tiles_to_clear = []
        self.tiles_to_fall = {}

        # Generate a board with at least one possible match
        while True:
            self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if self._check_board_for_matches():
                break
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # If animating, the action is ignored and we just advance the animation state
        if self.game_state != 'playing':
            self._update_animations()
            reward = 0
            # Check for win condition after animations complete
            if self.game_state == 'playing' and np.all(self.grid == -1):
                reward += 100
                self.score += 100
                self.game_over = True

            terminated = self._check_termination()
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1
        
        reward = 0
        self.steps += 1

        if shift_pressed:
            self.game_over = True
        
        if not self.game_over:
            # Handle cursor movement (does not consume a move)
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)
            
            # Handle match attempt (consumes a move)
            if space_pressed and self.moves_remaining > 0:
                self.moves_remaining -= 1
                
                cx, cy = self.cursor_pos
                match_group = self._find_match_group(cx, cy)
                
                if len(match_group) >= 3:
                    # // SFX: Match success
                    reward += len(match_group)
                    if len(match_group) > 3:
                        reward += 5
                    self.score += reward
                    
                    self.tiles_to_clear = match_group
                    self.game_state = 'clearing'
                    self.animation_timer = 15 # frames for clearing animation
                else:
                    # // SFX: Match fail
                    pass # Just a wasted move

        terminated = self._check_termination()
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_animations(self):
        self.animation_timer -= 1
        if self.animation_timer > 0:
            return

        if self.game_state == 'clearing':
            # Mark tiles as empty
            for x, y in self.tiles_to_clear:
                self.grid[y, x] = -1
            self.tiles_to_clear = []
            
            self._prepare_fall()
            self.game_state = 'falling'
            self.animation_timer = 15 # frames for falling animation
            
        elif self.game_state == 'falling':
            self._apply_fall()
            self.game_state = 'playing'
            # After falling, check if any moves are left
            if not self._check_board_for_matches():
                self.game_over = True

    def _prepare_fall(self):
        self.tiles_to_fall = {}
        for x in range(self.GRID_WIDTH):
            empty_count = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] == -1:
                    empty_count += 1
                elif empty_count > 0:
                    self.tiles_to_fall[(x, y)] = empty_count
    
    def _apply_fall(self):
        # Move existing tiles down
        for x in range(self.GRID_WIDTH):
            col = self.grid[:, x]
            non_empty = col[col != -1]
            new_col = np.full(self.GRID_HEIGHT, -1)
            new_col[self.GRID_HEIGHT - len(non_empty):] = non_empty
            self.grid[:, x] = new_col

        # Refill empty spaces at the top with new tiles
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[y, x] == -1:
                    self.grid[y, x] = self.np_random.integers(0, self.NUM_COLORS)
        self.tiles_to_fall = {}

    def _find_match_group(self, start_x, start_y):
        if self.grid[start_y, start_x] == -1:
            return []
            
        target_color = self.grid[start_y, start_x]
        q = [(start_x, start_y)]
        visited = set(q)
        group = []

        while q:
            x, y = q.pop(0)
            group.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[ny, nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def _check_board_for_matches(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != -1:
                    if len(self._find_match_group(x, y)) >= 3:
                        return True
        return False

    def _check_termination(self):
        if self.game_over:
            return True
        if self.moves_remaining <= 0 and self.game_state == 'playing':
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
    
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
            "moves_remaining": self.moves_remaining
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_offset_x + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_offset_y), (x, self.grid_offset_y + self.GRID_HEIGHT * self.TILE_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_offset_y + i * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, y), (self.grid_offset_x + self.GRID_WIDTH * self.TILE_SIZE, y))

        # Draw tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.grid[y, x]
                if color_idx == -1:
                    continue
                
                tile_rect = pygame.Rect(
                    self.grid_offset_x + x * self.TILE_SIZE,
                    self.grid_offset_y + y * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                
                center_x, center_y = tile_rect.center
                size = self.TILE_SIZE
                
                # Handle animations
                if self.game_state == 'clearing' and (x, y) in self.tiles_to_clear:
                    progress = self.animation_timer / 15.0
                    size = int(self.TILE_SIZE * progress)
                elif self.game_state == 'falling' and (x, y) in self.tiles_to_fall:
                    fall_dist = self.tiles_to_fall[(x, y)]
                    progress = self.animation_timer / 15.0
                    offset = int(fall_dist * self.TILE_SIZE * progress)
                    tile_rect.y -= offset

                if size > 0:
                    draw_rect = pygame.Rect(center_x - size // 2, tile_rect.y, size, size)
                    pygame.draw.rect(self.screen, self.TILE_COLORS[color_idx], draw_rect, border_radius=5)
        
        # Draw cursor
        cursor_x = self.grid_offset_x + self.cursor_pos[0] * self.TILE_SIZE
        cursor_y = self.grid_offset_y + self.cursor_pos[1] * self.TILE_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.TILE_SIZE, self.TILE_SIZE)
        
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.005)) * 5
        pygame.draw.rect(self.screen, (255,255,255), cursor_rect.inflate(pulse, pulse), width=2, border_radius=7)

    def _render_ui(self):
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        moves_text = self.font_small.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 20))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win = np.all(self.grid == -1)
            end_text_str = "BOARD CLEARED!" if win else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Tile Matcher")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    # Track button presses to simulate single action per press
    last_action_time = 0
    action_cooldown = 150 # ms between actions

    while running:
        current_time = pygame.time.get_ticks()
        
        # --- Human Input ---
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        action_taken = False
        if current_time - last_action_time > action_cooldown:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1; action_taken = True
            elif keys[pygame.K_DOWN]: movement = 2; action_taken = True
            elif keys[pygame.K_LEFT]: movement = 3; action_taken = True
            elif keys[pygame.K_RIGHT]: movement = 4; action_taken = True
            
            if keys[pygame.K_SPACE]: space_held = 1; action_taken = True
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1; action_taken = True

        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        if terminated:
            if shift_held:
                obs, info = env.reset()
                terminated = False
                last_action_time = current_time
        elif action_taken or env.game_state != 'playing':
            obs, reward, terminated, truncated, info = env.step(action)
            if action_taken:
                last_action_time = current_time
            if reward > 0:
                print(f"Reward: {reward}, Score: {info['score']}")

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60)

    env.close()