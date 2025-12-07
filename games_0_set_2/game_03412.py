
# Generated: 2025-08-27T23:17:08.313784
# Source Brief: brief_03412.md
# Brief Index: 3412

        
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
        "Controls: ←→ to select a column. Press space to pick up or place a block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Sort colored blocks into matching stacks before time runs out. "
        "Each move costs time. Plan your moves to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    NUM_COLUMNS = 5
    MAX_ROWS = 5
    NUM_BLOCKS_PER_COLOR = 4
    MAX_STEPS = 100

    # --- Colors ---
    COLOR_BG = (30, 30, 40)
    COLOR_GRID = (50, 50, 60)
    COLOR_TEXT = (220, 220, 230)
    COLOR_SELECTOR = (255, 255, 255, 60)
    COLOR_HELD_GLOW = (255, 255, 255)

    COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
    ]

    # --- Sizing ---
    GRID_WIDTH = 500
    GRID_HEIGHT = 250
    CELL_WIDTH = GRID_WIDTH // NUM_COLUMNS
    CELL_HEIGHT = GRID_HEIGHT // MAX_ROWS
    BLOCK_SIZE = int(min(CELL_WIDTH, CELL_HEIGHT) * 0.8)
    BLOCK_MARGIN_X = (CELL_WIDTH - BLOCK_SIZE) // 2
    BLOCK_MARGIN_Y = (CELL_HEIGHT - BLOCK_SIZE) // 2
    GRID_START_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_START_Y = SCREEN_HEIGHT - GRID_HEIGHT - 20

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_feedback = pygame.font.Font(None, 28)

        # Initialize state variables
        self.grid = []
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.selected_column = 0
        self.held_block = None
        self.last_space_state = False
        self.particles = []
        self.last_action_feedback = ""
        self.feedback_alpha = 0
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_left = self.MAX_STEPS
        self.game_over = False
        self.selected_column = self.NUM_COLUMNS // 2
        self.held_block = None
        self.last_space_state = False
        self.particles = []
        self.last_action_feedback = ""
        self.feedback_alpha = 0
        
        self.grid = [[] for _ in range(self.NUM_COLUMNS)]

        total_colors = len(self.COLORS)
        blocks_to_place = []
        for i in range(total_colors):
            blocks_to_place.extend([self.COLORS[i]] * self.NUM_BLOCKS_PER_COLOR)
        
        self.np_random.shuffle(blocks_to_place)

        for block in blocks_to_place:
            possible_cols = [i for i, col in enumerate(self.grid) if len(col) < self.MAX_ROWS]
            if not possible_cols:
                break 
            target_col_idx = self.np_random.choice(possible_cols)
            self.grid[target_col_idx].append(block)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_action, shift_action = action
        space_pressed = space_action == 1 and not self.last_space_state
        self.last_space_state = space_action == 1

        self.steps += 1
        self.time_left -= 1
        reward = 0

        # --- Handle Input and Game Logic ---
        # 1. Column Selection
        if movement == 3:  # Left
            self.selected_column = (self.selected_column - 1 + self.NUM_COLUMNS) % self.NUM_COLUMNS
        elif movement == 4:  # Right
            self.selected_column = (self.selected_column + 1) % self.NUM_COLUMNS

        # 2. Block Interaction
        if space_pressed:
            if self.held_block is None:
                # Try to pick up a block
                if self.grid[self.selected_column]:
                    self.held_block = self.grid[self.selected_column].pop()
                    self._set_feedback("Picked up!", 0)
                    # Sound: Block pick up
                else:
                    reward -= 0.01
                    self._set_feedback("Empty!", -0.01)
                    # Sound: Error buzz
            else:
                # Try to place a block
                target_col = self.grid[self.selected_column]
                is_empty = not target_col
                is_full = len(target_col) >= self.MAX_ROWS
                top_block_matches = not is_empty and target_col[-1] == self.held_block

                if not is_full and (is_empty or top_block_matches):
                    # --- Valid placement ---
                    was_complete_before = self._is_stack_complete(self.selected_column)
                    
                    target_col.append(self.held_block)
                    
                    is_complete_after = self._is_stack_complete(self.selected_column)

                    self.held_block = None
                    reward += 0.1
                    self._set_feedback("Placed!", 0.1)
                    # Sound: Block place
                    self._create_particles(
                        self.GRID_START_X + self.selected_column * self.CELL_WIDTH + self.CELL_WIDTH // 2,
                        self.GRID_START_Y + self.GRID_HEIGHT - (len(target_col) * self.CELL_HEIGHT) + self.CELL_HEIGHT // 2,
                        target_col[-1]
                    )
                    
                    if is_complete_after and not was_complete_before:
                        reward += 5.0
                        self._set_feedback("Stack Complete!", 5.0)
                        # Sound: Stack complete fanfare
                else:
                    # --- Invalid placement ---
                    reward -= 0.01
                    self._set_feedback("Can't place!", -0.01)
                    # Sound: Error buzz
        
        self.score += reward
        
        # --- Check Termination Conditions ---
        win_condition = self._check_win()
        time_out = self.time_left <= 0
        terminated = win_condition or time_out

        if terminated and not self.game_over:
            if win_condition:
                terminal_reward = 50.0
                self.score += terminal_reward
                reward += terminal_reward
                self._set_feedback("YOU WIN!", terminal_reward, 255)
            else: # Time out
                terminal_reward = -50.0
                self.score += terminal_reward
                reward += terminal_reward
                self._set_feedback("TIME UP!", terminal_reward, 255)
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_stack_complete(self, col_idx):
        column = self.grid[col_idx]
        if len(column) != self.NUM_BLOCKS_PER_COLOR:
            return False
        first_color = column[0]
        return all(block == first_color for block in column)

    def _check_win(self):
        for column in self.grid:
            if not column:
                continue
            first_color = column[0]
            for block in column:
                if block != first_color:
                    return False  # Found a mixed-color stack
        return True # All non-empty stacks are monochromatic

    def _set_feedback(self, text, value, alpha=150):
        if value > 0:
            self.last_action_feedback = f"{text} (+{value:.2f})"
        elif value < 0:
            self.last_action_feedback = f"{text} ({value:.2f})"
        else:
            self.last_action_feedback = text
        self.feedback_alpha = alpha

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(1, self.NUM_COLUMNS):
            x = self.GRID_START_X + i * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_START_Y), (x, self.GRID_START_Y + self.GRID_HEIGHT), 2)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_START_X, self.GRID_START_Y, self.GRID_WIDTH, self.GRID_HEIGHT), 2)

        # Draw selector
        selector_rect = pygame.Rect(
            self.GRID_START_X + self.selected_column * self.CELL_WIDTH,
            self.GRID_START_Y,
            self.CELL_WIDTH,
            self.GRID_HEIGHT
        )
        s = pygame.Surface((self.CELL_WIDTH, self.GRID_HEIGHT), pygame.SRCALPHA)
        s.fill(self.COLOR_SELECTOR)
        self.screen.blit(s, (selector_rect.x, selector_rect.y))

        # Draw blocks in grid
        for i, column in enumerate(self.grid):
            for j, color in enumerate(column):
                x = self.GRID_START_X + i * self.CELL_WIDTH + self.BLOCK_MARGIN_X
                y = self.GRID_START_Y + self.GRID_HEIGHT - (j + 1) * self.CELL_HEIGHT + self.BLOCK_MARGIN_Y
                self._draw_block(self.screen, (x, y), color)

        # Draw held block
        if self.held_block:
            x = self.GRID_START_X + self.selected_column * self.CELL_WIDTH + self.BLOCK_MARGIN_X
            y = self.GRID_START_Y - self.CELL_HEIGHT * 1.2
            self._draw_block(self.screen, (x, y), self.held_block, glow=True)
            
        # Update and draw particles
        self._update_and_draw_particles()

    def _draw_block(self, surface, pos, color, glow=False):
        rect = pygame.Rect(pos[0], pos[1], self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        # Darker color for 3D effect
        dark_color = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.rect(surface, dark_color, rect.move(0, 3))
        pygame.draw.rect(surface, dark_color, rect.move(3, 0))
        pygame.draw.rect(surface, color, rect)
        
        if glow:
            glow_rect = rect.inflate(6, 6)
            pygame.draw.rect(surface, self.COLOR_HELD_GLOW, glow_rect, 2, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Time
        time_text = self.font_main.render(f"Time: {self.time_left}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(time_text, time_rect)
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
        # Action Feedback Text
        if self.feedback_alpha > 0:
            feedback_surf = self.font_feedback.render(self.last_action_feedback, True, self.COLOR_TEXT)
            feedback_surf.set_alpha(self.feedback_alpha)
            feedback_rect = feedback_surf.get_rect(center=(self.SCREEN_WIDTH / 2, 35))
            self.screen.blit(feedback_surf, feedback_rect)
            self.feedback_alpha = max(0, self.feedback_alpha - 5)

    def _create_particles(self, x, y, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifespan = self.np_random.integers(20, 40)
            self.particles.append([x, y, vx, vy, lifespan, color])

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p[0] += p[1]  # x += vx
            p[1] += p[2]  # y += vy
            p[4] -= 1     # lifespan -= 1
            if p[4] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p[4] / 40))
                color = (*p[5], alpha)
                s = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (2, 2), 2)
                self.screen.blit(s, (int(p[0]), int(p[1])))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "held_block": self.held_block is not None,
            "is_win": self._check_win() if not self.game_over else self.time_left > 0
        }

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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # This part is for human testing and is not part of the gym environment
    
    print("Game: Block Sorter")
    print("Description:", GameEnv.game_description)
    print("Controls:", GameEnv.user_guide)
    
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    # Create a window to display the game
    pygame.display.set_caption("Block Sorter")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Win: {info['is_win']}")
            # Wait a bit then reset
            pygame.time.wait(3000)
            obs, info = env.reset()
            terminated = False
            total_reward = 0
            
        clock.tick(30) # Limit frame rate for human play

    env.close()