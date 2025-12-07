
# Generated: 2025-08-28T06:17:52.438565
# Source Brief: brief_02891.md
# Brief Index: 2891

        
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
        "Use Left/Right to select a column. Press Space to pick up or drop a ball. Use Shift to cancel a pickup."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Sort the colored balls into their matching columns. Plan your moves carefully as you only have a limited number to solve the puzzle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    NUM_COLS = 6
    COL_HEIGHT = 4
    MAX_MOVES = 25 

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SELECTOR = (255, 255, 0)
    
    # Ball colors (ID and RGB value)
    COLORS = {
        1: (255, 80, 80),   # Red
        2: (80, 255, 80),   # Green
        3: (80, 120, 255)   # Blue
    }
    
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
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Game State
        self.grid = []
        self.target_colors = []
        self.selector_pos = 0
        self.held_ball_data = None # (ball_id, original_col_index)
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        
        self.particles = []
        self.last_space_state = False
        self.last_shift_state = False

        self._setup_columns()
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def _setup_columns(self):
        # Assign target colors to columns
        num_colors = len(self.COLORS)
        cols_per_color = self.NUM_COLS // num_colors
        self.target_colors = []
        for i, color_id in enumerate(self.COLORS.keys()):
            self.target_colors.extend([color_id] * cols_per_color)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.moves_left = self.MAX_MOVES
        self.selector_pos = self.NUM_COLS // 2
        self.held_ball_data = None
        self.particles = []
        self.last_space_state = False
        self.last_shift_state = False
        
        # Generate and shuffle balls
        num_balls_per_color = (self.NUM_COLS * self.COL_HEIGHT) // len(self.COLORS)
        all_balls = []
        for color_id in self.COLORS.keys():
            all_balls.extend([color_id] * num_balls_per_color)
        
        self.np_random.shuffle(all_balls)
        
        # Distribute balls into the grid
        self.grid = [[] for _ in range(self.NUM_COLS)]
        for i, ball_id in enumerate(all_balls):
            col_index = i % self.NUM_COLS
            self.grid[col_index].append(ball_id)
            
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Use rising edge detection for button presses
        space_pressed = space_held and not self.last_space_state
        shift_pressed = shift_held and not self.last_shift_state

        reward = 0
        move_executed = False

        # --- Action Handling ---
        if movement == 3: # Left
            self.selector_pos = max(0, self.selector_pos - 1)
        elif movement == 4: # Right
            self.selector_pos = min(self.NUM_COLS - 1, self.selector_pos + 1)
        
        if space_pressed:
            if self.held_ball_data is None:
                if self.grid[self.selector_pos]: # Pick up
                    ball_id = self.grid[self.selector_pos].pop()
                    self.held_ball_data = (ball_id, self.selector_pos)
                    # sfx: pickup
            else:
                if len(self.grid[self.selector_pos]) < self.COL_HEIGHT: # Drop
                    ball_id, _ = self.held_ball_data
                    pre_move_sorted_state = self._get_sorted_columns_state()

                    self.grid[self.selector_pos].append(ball_id)
                    self.held_ball_data = None
                    self.moves_left -= 1
                    move_executed = True
                    
                    reward += self._calculate_continuous_reward()
                    reward += self._calculate_event_reward(pre_move_sorted_state)

                    drop_pos = self._get_ball_pos(self.selector_pos, len(self.grid[self.selector_pos]) - 1)
                    self._create_particles(drop_pos, self.COLORS[ball_id])
                    # sfx: drop
        
        if shift_pressed and self.held_ball_data is not None: # Cancel
            ball_id, origin_col = self.held_ball_data
            self.grid[origin_col].append(ball_id)
            self.held_ball_data = None
            # sfx: cancel

        if move_executed:
            self.steps += 1
            self.score += reward

        self.last_space_state = space_held
        self.last_shift_state = shift_held

        # --- Termination and Terminal Rewards ---
        is_win = self._is_win_condition()
        is_loss = self.moves_left <= 0 and not is_win
        terminated = is_win or is_loss
        
        if terminated:
            self.game_over = True
            if is_win:
                terminal_reward = 100
                reward += terminal_reward
                self.score += terminal_reward
                self.win_message = "PUZZLE SOLVED!"
                # sfx: win jingle
            if is_loss:
                # Per brief, terminal loss reward overrides others for the step
                reward = -100
                self.score += reward
                self.win_message = "OUT OF MOVES"
                # sfx: lose sound

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _calculate_continuous_reward(self):
        r = 0
        for col_idx, column in enumerate(self.grid):
            target_color_id = self.target_colors[col_idx]
            for ball_id in column:
                if ball_id == target_color_id:
                    r += 1
        return r

    def _calculate_event_reward(self, pre_move_sorted_state):
        r = 0
        post_move_sorted_state = self._get_sorted_columns_state()
        for i in range(self.NUM_COLS):
            if post_move_sorted_state[i] and not pre_move_sorted_state[i]:
                r += 10
        return r

    def _is_column_sorted(self, col_idx):
        column = self.grid[col_idx]
        if not column: return False
        target_color = self.target_colors[col_idx]
        return all(ball_id == target_color for ball_id in column)

    def _get_sorted_columns_state(self):
        return [self._is_column_sorted(i) for i in range(self.NUM_COLS)]

    def _is_win_condition(self):
        for i in range(self.NUM_COLS):
            if len(self.grid[i]) != self.COL_HEIGHT or not self._is_column_sorted(i):
                return False
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_column_backgrounds()
        self._render_balls()
        self._render_selector()
        self._render_held_ball()
        self._update_and_render_particles()

    def _get_column_rect(self, col_idx):
        col_width = self.SCREEN_WIDTH / (self.NUM_COLS + 1)
        spacing = col_width / self.NUM_COLS
        total_grid_width = self.NUM_COLS * col_width + (self.NUM_COLS - 1) * spacing
        start_x = (self.SCREEN_WIDTH - total_grid_width) / 2
        x = start_x + col_idx * (col_width + spacing)
        y = self.SCREEN_HEIGHT * 0.25
        h = self.SCREEN_HEIGHT * 0.7
        return pygame.Rect(x, y, col_width, h)

    def _get_ball_pos(self, col_idx, row_idx):
        col_rect = self._get_column_rect(col_idx)
        ball_radius = col_rect.width * 0.4
        x = col_rect.centerx
        y = col_rect.bottom - ball_radius - row_idx * (ball_radius * 2.1)
        return int(x), int(y)

    def _render_column_backgrounds(self):
        for i in range(self.NUM_COLS):
            rect = self._get_column_rect(i)
            target_color_id = self.target_colors[i]
            base_color = self.COLORS[target_color_id]
            bg_color = tuple(int(c * 0.2) for c in base_color)
            pygame.draw.rect(self.screen, bg_color, rect, border_radius=8)
            if self._is_column_sorted(i) and len(self.grid[i]) == self.COL_HEIGHT:
                pygame.draw.rect(self.screen, self.COLORS[target_color_id], rect, width=4, border_radius=8)

    def _render_balls(self):
        for col_idx, column in enumerate(self.grid):
            for row_idx, ball_id in enumerate(column):
                pos = self._get_ball_pos(col_idx, row_idx)
                col_rect = self._get_column_rect(col_idx)
                radius = int(col_rect.width * 0.4)
                color = self.COLORS[ball_id]
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
                highlight_pos = (pos[0] - radius // 3, pos[1] - radius // 3)
                pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], radius // 3, (255, 255, 255, 100))

    def _render_selector(self):
        rect = self._get_column_rect(self.selector_pos)
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
        color = (*self.COLOR_SELECTOR, alpha)
        temp_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(temp_surface, color, temp_surface.get_rect(), border_radius=12)
        self.screen.blit(temp_surface, rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, width=3, border_radius=12)

    def _render_held_ball(self):
        if self.held_ball_data:
            ball_id, _ = self.held_ball_data
            col_rect = self._get_column_rect(self.selector_pos)
            radius = int(col_rect.width * 0.4)
            pos_x, pos_y = col_rect.centerx, col_rect.top - radius * 1.5
            pos = (int(pos_x), int(pos_y))
            color = self.COLORS[ball_id]
            shadow_pos = (pos[0] + 5, pos[1] + 5)
            pygame.gfxdraw.filled_circle(self.screen, shadow_pos[0], shadow_pos[1], radius, (0,0,0,100))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            highlight_pos = (pos[0] - radius // 3, pos[1] - radius // 3)
            pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], radius // 3, (255, 255, 255, 100))

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['life'] -= 0.05
            if p['life'] <= 0: self.particles.remove(p); continue
            p['pos'][0] += p['vel'][0]; p['pos'][1] += p['vel'][1]; p['vel'][1] += 0.2
            alpha = int(255 * p['life']); color = (*p['color'], alpha)
            size = int(p['size'] * p['life'])
            if size > 0:
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (p['pos'][0]-size, p['pos'][1]-size))

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi); speed = random.uniform(1, 4)
            self.particles.append({'pos': list(pos), 'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2], 'life': 1.0, 'color': color, 'size': random.randint(4, 8)})

    def _render_ui(self):
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 20))
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        if self.game_over:
            msg_surf = self.font_msg.render(self.win_message, True, self.COLOR_SELECTOR)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 150))
            shadow_surf = self.font_msg.render(self.win_message, True, (0,0,0,150))
            self.screen.blit(shadow_surf, msg_rect.move(3,3))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

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
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    pygame.display.set_caption("Ball Sorter Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    while not done:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: obs, info = env.reset()
                if event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1

        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            if terminated:
                print("Game Over. Press 'R' to restart.")
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()