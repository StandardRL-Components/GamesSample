
# Generated: 2025-08-28T05:07:02.805974
# Source Brief: brief_05464.md
# Brief Index: 5464

        
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
        "Controls: Use arrow keys to move the selector. Hold space to pick up a block, "
        "move to an adjacent empty square, and release space to drop it. Match 3 or more to clear!"
    )

    game_description = (
        "A fast-paced puzzle game. Match 3 or more colored blocks to clear them from the board. "
        "Clear the entire board before the 60-second timer runs out to win!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    BLOCK_SIZE = 40
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH * BLOCK_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT * BLOCK_SIZE) // 2 + 20

    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 255)
    
    BLOCK_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 150, 255),   # Blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
    ]
    NUM_COLORS = len(BLOCK_COLORS)

    MAX_STEPS = 1500
    GAME_DURATION_SECONDS = 60
    
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
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        self.board = None
        self.cursor_pos = None
        self.held_block = None
        self.prev_space_held = False
        self.timer = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.GAME_DURATION_SECONDS * self.FPS
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.held_block = None
        self.prev_space_held = False
        self.particles = []
        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        self.timer -= 1
        
        reward = -0.01  # Time penalty

        # --- Handle Input and Game Logic ---
        self._handle_cursor_movement(movement)
        match_reward = self._handle_block_interaction(space_held)
        reward += match_reward

        self._update_particles()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if np.sum(self.board) == 0: # Win condition
                reward += 100
            else: # Loss condition
                reward -= 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_board(self):
        # Brute-force a solvable board
        for _ in range(100): # Safety break
            self.board = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            # Ensure no initial matches
            while len(self._find_all_matches()) > 0:
                self.board = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
            
            if self._is_solvable():
                return
        # Fallback: if 100 attempts fail, just use the last generated board
        # print("Warning: Could not generate a guaranteed solvable board in 100 attempts.")

    def _is_solvable(self):
        # Check if any adjacent swap creates a match
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Horizontal swap
                if x < self.GRID_WIDTH - 1:
                    self._swap_blocks(x, y, x + 1, y)
                    if len(self._find_all_matches()) > 0:
                        self._swap_blocks(x, y, x + 1, y) # Swap back
                        return True
                    self._swap_blocks(x, y, x + 1, y) # Swap back
                
                # Vertical swap
                if y < self.GRID_HEIGHT - 1:
                    self._swap_blocks(x, y, x, y + 1)
                    if len(self._find_all_matches()) > 0:
                        self._swap_blocks(x, y, x, y + 1) # Swap back
                        return True
                    self._swap_blocks(x, y, x, y + 1) # Swap back
        return False

    def _swap_blocks(self, x1, y1, x2, y2):
        self.board[x1, y1], self.board[x2, y2] = self.board[x2, y2], self.board[x1, y1]

    def _handle_cursor_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

    def _handle_block_interaction(self, space_held):
        is_press = space_held and not self.prev_space_held
        is_release = not space_held and self.prev_space_held
        self.prev_space_held = space_held
        
        reward = 0
        
        # Pick up block
        if is_press and self.held_block is None:
            cx, cy = self.cursor_pos
            if self.board[cx, cy] > 0:
                # SFX: Block pickup
                self.held_block = {
                    'color': self.board[cx, cy],
                    'original_pos': (cx, cy)
                }
                self.board[cx, cy] = 0

        # Drop block
        elif is_release and self.held_block is not None:
            cx, cy = self.cursor_pos
            ox, oy = self.held_block['original_pos']
            
            is_adjacent = abs(cx - ox) + abs(cy - oy) == 1
            is_empty = self.board[cx, cy] == 0
            
            if is_adjacent and is_empty:
                # Valid drop
                # SFX: Block drop
                self.board[cx, cy] = self.held_block['color']
                self.held_block = None
                
                reward = self._handle_cascading_matches()
                if np.sum(self.board) == 0:
                    reward += 10 # Board clear bonus
            else:
                # Invalid drop, return block
                # SFX: Invalid move
                self.board[ox, oy] = self.held_block['color']
                self.held_block = None
        
        return reward

    def _handle_cascading_matches(self):
        total_reward = 0
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            
            num_cleared = len(matches)
            total_reward += num_cleared # 1 point per block cleared
            self.score += num_cleared

            # SFX: Match success
            for x, y in matches:
                self._spawn_particles(x, y, self.board[x, y])
                self.board[x, y] = 0
            
            self._apply_gravity()
        
        return total_reward

    def _find_all_matches(self):
        to_clear = set()
        # Horizontal
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                color = self.board[x, y]
                if color > 0 and color == self.board[x+1, y] and color == self.board[x+2, y]:
                    # Find full length of match
                    end_x = x + 2
                    while end_x + 1 < self.GRID_WIDTH and self.board[end_x + 1, y] == color:
                        end_x += 1
                    for i in range(x, end_x + 1):
                        to_clear.add((i, y))

        # Vertical
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                color = self.board[x, y]
                if color > 0 and color == self.board[x, y+1] and color == self.board[x, y+2]:
                    # Find full length of match
                    end_y = y + 2
                    while end_y + 1 < self.GRID_HEIGHT and self.board[x, end_y + 1] == color:
                        end_y += 1
                    for i in range(y, end_y + 1):
                        to_clear.add((x, i))
                        
        return to_clear

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[x, y] > 0:
                    if y != empty_row:
                        self.board[x, empty_row] = self.board[x, y]
                        self.board[x, y] = 0
                    empty_row -= 1

    def _spawn_particles(self, grid_x, grid_y, color_index):
        px = self.GRID_X + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        py = self.GRID_Y + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        color = self.BLOCK_COLORS[color_index - 1]
        
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.uniform(15, 30) # frames
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1

    def _check_termination(self):
        if self.timer <= 0:
            return True
        if np.sum(self.board) == 0 and self.held_block is None:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.timer / self.FPS),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(self.GRID_X + x * self.BLOCK_SIZE, self.GRID_Y + y * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_index = self.board[x, y]
                if color_index > 0:
                    self._draw_block(x, y, color_index)

        # Draw held block
        if self.held_block:
            cx, cy = self.cursor_pos
            self._draw_block(cx, cy, self.held_block['color'], is_held=True)
            
        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_X + cx * self.BLOCK_SIZE, self.GRID_Y + cy * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(6 * (p['life'] / p['max_life']))
            if size > 0:
                s = pygame.Surface((size, size), pygame.SRCALPHA)
                s.fill((*p['color'], alpha))
                self.screen.blit(s, (int(p['pos'][0] - size/2), int(p['pos'][1] - size/2)))

    def _draw_block(self, grid_x, grid_y, color_index, is_held=False):
        color = self.BLOCK_COLORS[color_index - 1]
        
        size = self.BLOCK_SIZE
        if is_held:
            size = int(self.BLOCK_SIZE * 1.2)
        
        x = self.GRID_X + grid_x * self.BLOCK_SIZE + (self.BLOCK_SIZE - size) / 2
        y = self.GRID_Y + grid_y * self.BLOCK_SIZE + (self.BLOCK_SIZE - size) / 2
        
        rect = pygame.Rect(int(x), int(y), size, size)
        
        # Draw bright block with a darker border
        border_color = tuple(max(0, c - 50) for c in color)
        pygame.draw.rect(self.screen, border_color, rect, 0, border_radius=8)
        inner_rect = rect.inflate(-6, -6)
        pygame.draw.rect(self.screen, color, inner_rect, 0, border_radius=6)
        
        if is_held:
            pygame.draw.rect(self.screen, (255, 255, 255, 100), rect, 3, border_radius=10)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        time_color = self.COLOR_UI_TEXT if time_left > 10 else (255, 100, 100)
        timer_text = self.font_large.render(f"TIME: {time_left:.1f}", True, time_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(timer_text, timer_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            win = np.sum(self.board) == 0
            message = "BOARD CLEARED!" if win else "TIME'S UP!"
            color = (100, 255, 100) if win else (255, 100, 100)
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, end_rect)

    def close(self):
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Puzzle")
    clock = pygame.time.Clock()
    
    # Game loop
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward != -0.01:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    env.close()