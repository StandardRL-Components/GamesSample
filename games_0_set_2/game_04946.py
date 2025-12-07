
# Generated: 2025-08-28T03:30:29.619077
# Source Brief: brief_04946.md
# Brief Index: 4946

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the selector. Press space to swap the selected gem "
        "with the gem in the direction of your last movement."
    )

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent gems to create lines of 3 or more. "
        "Chain reactions grant bonus points. Reach the target score before you run out of moves!"
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 8
    GEM_TYPES = 5
    BOARD_OFFSET_X, BOARD_OFFSET_Y = 160, 40
    GEM_SIZE = 40
    GRID_LINE_WIDTH = 2
    
    # --- Colors ---
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (40, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SCORE = (255, 223, 0)
    COLOR_MOVES = (100, 200, 255)
    
    GEM_COLORS = [
        (0, 0, 0),  # 0: Empty
        (255, 50, 50),   # 1: Red
        (50, 255, 50),   # 2: Green
        (80, 150, 255),  # 3: Blue
        (255, 255, 50),  # 4: Yellow
        (200, 80, 255),  # 5: Purple
    ]
    
    # --- Game Settings ---
    INITIAL_MOVES = 30
    TARGET_SCORE = 1000
    MAX_STEPS = 500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        self.board = None
        self.cursor_pos = None
        self.last_move_dir = None
        self.score = None
        self.moves_left = None
        self.steps = None
        self.game_over = None
        self.particles = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.last_move_dir = (-1, 0)  # Default to UP
        self.particles = []
        self.board = self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_press, _ = action
        space_pressed = space_press == 1

        # 1. Handle cursor movement
        moved = self._move_cursor(movement)

        # 2. Handle swap action
        if space_pressed:
            reward = self._process_swap()
            self.moves_left -= 1
        
        # 3. Update particles
        self._update_particles()

        # 4. Check for termination
        terminated = False
        if self.score >= self.TARGET_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0:
            reward -= 10
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _move_cursor(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dy) % self.GRID_SIZE
            self.cursor_pos[1] = (self.cursor_pos[1] + dx) % self.GRID_SIZE
            self.last_move_dir = (dy, dx)
            return True
        return False

    def _process_swap(self):
        row, col = self.cursor_pos
        dy, dx = self.last_move_dir
        
        target_row, target_col = row + dy, col + dx
        
        # Check if swap is within bounds
        if not (0 <= target_row < self.GRID_SIZE and 0 <= target_col < self.GRID_SIZE):
            return -0.2 # Invalid swap attempt off board

        # Perform the swap
        self.board[row, col], self.board[target_row, target_col] = \
            self.board[target_row, target_col], self.board[row, col]

        # Check for matches
        matches = self._find_matches()
        if not np.any(matches):
            # No match, swap back
            self.board[row, col], self.board[target_row, target_col] = \
                self.board[target_row, target_col], self.board[row, col]
            return -0.2
        
        # Process matches and cascades
        total_reward = 0
        chain_multiplier = 1.0
        while np.any(matches):
            num_cleared = np.sum(matches)
            
            # Calculate reward
            reward = num_cleared * chain_multiplier
            if num_cleared == 4: reward += 5
            if num_cleared >= 5: reward += 10
            total_reward += reward
            
            self.score += int(reward * 10) # Scale reward for score display

            # Create particles and clear gems
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    if matches[r, c]:
                        self._create_particles(r, c, self.board[r, c])
                        self.board[r, c] = 0
            
            # Sound effect placeholder
            # pygame.mixer.Sound("match.wav").play()

            self._apply_gravity()
            self._refill_board()
            
            matches = self._find_matches()
            chain_multiplier += 0.5 # Increase multiplier for next chain
        
        return total_reward

    def _generate_board(self):
        while True:
            board = self.np_random.integers(1, self.GEM_TYPES + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            
            # Remove initial matches
            while True:
                matches = self._find_matches(board)
                if not np.any(matches):
                    break
                for r in range(self.GRID_SIZE):
                    for c in range(self.GRID_SIZE):
                        if matches[r, c]:
                            board[r, c] = self.np_random.integers(1, self.GEM_TYPES + 1)
            
            # Ensure at least one move is possible
            if self._find_possible_moves(board):
                return board

    def _find_possible_moves(self, board):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                # Check swap right
                if c < self.GRID_SIZE - 1:
                    board[r, c], board[r, c + 1] = board[r, c + 1], board[r, c]
                    if np.any(self._find_matches(board)):
                        board[r, c], board[r, c + 1] = board[r, c + 1], board[r, c]
                        return True
                    board[r, c], board[r, c + 1] = board[r, c + 1], board[r, c]
                
                # Check swap down
                if r < self.GRID_SIZE - 1:
                    board[r, c], board[r + 1, c] = board[r + 1, c], board[r, c]
                    if np.any(self._find_matches(board)):
                        board[r + 1, c], board[r, c] = board[r, c], board[r + 1, c]
                        return True
                    board[r + 1, c], board[r, c] = board[r, c], board[r + 1, c]
        return False

    def _find_matches(self, board=None):
        if board is None:
            board = self.board
        
        matches = np.zeros_like(board, dtype=bool)
        
        # Horizontal matches
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if board[r, c] != 0 and board[r, c] == board[r, c + 1] == board[r, c + 2]:
                    matches[r, c:c + 3] = True
        
        # Vertical matches
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if board[r, c] != 0 and board[r, c] == board[r + 1, c] == board[r + 2, c]:
                    matches[r:r + 3, c] = True
        
        return matches

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
                    self.board[r, c] = self.np_random.integers(1, self.GEM_TYPES + 1)
    
    def _create_particles(self, row, col, gem_type):
        center_x = self.BOARD_OFFSET_X + col * self.GEM_SIZE + self.GEM_SIZE // 2
        center_y = self.BOARD_OFFSET_Y + row * self.GEM_SIZE + self.GEM_SIZE // 2
        color = self.GEM_COLORS[gem_type]

        for _ in range(15): # Number of particles per gem
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": [center_x, center_y],
                "vel": [vel_x, vel_y],
                "lifetime": lifetime,
                "max_lifetime": lifetime,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "steps": self.steps,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_gems()
        self._draw_cursor()
        self._draw_particles()

    def _draw_grid(self):
        board_width = self.GRID_SIZE * self.GEM_SIZE
        board_height = self.GRID_SIZE * self.GEM_SIZE
        pygame.draw.rect(self.screen, self.COLOR_GRID, 
                         (self.BOARD_OFFSET_X - self.GRID_LINE_WIDTH,
                          self.BOARD_OFFSET_Y - self.GRID_LINE_WIDTH,
                          board_width + self.GRID_LINE_WIDTH * 2,
                          board_height + self.GRID_LINE_WIDTH * 2), 
                         self.GRID_LINE_WIDTH, border_radius=5)

    def _draw_gems(self):
        radius = self.GEM_SIZE // 2 - 4
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.board[r, c]
                if gem_type == 0: continue
                
                center_x = int(self.BOARD_OFFSET_X + c * self.GEM_SIZE + self.GEM_SIZE / 2)
                center_y = int(self.BOARD_OFFSET_Y + r * self.GEM_SIZE + self.GEM_SIZE / 2)
                
                color = self.GEM_COLORS[gem_type]
                highlight = tuple(min(255, val + 60) for val in color)

                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                
                # Add a subtle highlight
                pygame.gfxdraw.filled_circle(self.screen, center_x - radius//3, center_y - radius//3, radius//3, highlight)


    def _draw_cursor(self):
        row, col = self.cursor_pos
        x = self.BOARD_OFFSET_X + col * self.GEM_SIZE
        y = self.BOARD_OFFSET_Y + row * self.GEM_SIZE
        
        # Pulsating alpha for glow effect
        alpha = 128 + int(127 * math.sin(pygame.time.get_ticks() * 0.005))
        
        rect = pygame.Rect(x, y, self.GEM_SIZE, self.GEM_SIZE)
        
        # Draw a thick, glowing border
        pygame.draw.rect(self.screen, (255, 255, 255, alpha), rect.inflate(4, 4), 3, border_radius=5)

    def _draw_particles(self):
        for p in self.particles:
            x, y = int(p["pos"][0]), int(p["pos"][1])
            alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
            color = (*p["color"], alpha)
            size = int(5 * (p["lifetime"] / p["max_lifetime"]))
            if size > 0:
                # Create a temporary surface for the particle to handle alpha
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (x - size, y - size))


    def _render_ui(self):
        # Score Display
        score_text = self.font_large.render("SCORE", True, self.COLOR_TEXT)
        score_val = self.font_large.render(f"{self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 50))
        self.screen.blit(score_val, (20, 85))

        # Moves Display
        moves_text = self.font_large.render("MOVES", True, self.COLOR_TEXT)
        moves_val = self.font_large.render(f"{self.moves_left}", True, self.COLOR_MOVES)
        self.screen.blit(moves_text, (20, 150))
        self.screen.blit(moves_val, (20, 185))
        
        # Target Score
        target_text = self.font_small.render(f"Target: {self.TARGET_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(target_text, (20, 240))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.TARGET_SCORE:
                end_text_str = "LEVEL CLEAR!"
                end_color = self.COLOR_SCORE
            else:
                end_text_str = "GAME OVER"
                end_color = self.COLOR_MOVES
            
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a screen for display if not running headless
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Match-3 Gym Environment")
    except pygame.error:
        # Running in a headless environment
        screen = None

    print(GameEnv.user_guide)
    
    action = [0, 0, 0] # No-op, no-space, no-shift
    
    while not done:
        # --- Manual Control ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                    print("--- Game Reset ---")

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        # Reset action after processing
        action = [0, 0, 0]

        # Render to the display window if it exists
        if screen:
            # Pygame uses (width, height), numpy uses (height, width, channels)
            # The observation is (height, width, channels), so we need to transpose it for pygame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS for manual play

    print("Game Over!")
    print(f"Final Info: {info}")
    env.close()