
# Generated: 2025-08-27T21:55:27.855303
# Source Brief: brief_02950.md
# Brief Index: 2950

        
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
        "Controls: Arrow keys to move the selector. Press Space to push the selected block in the last direction you moved."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A time-based puzzle game. Push blocks to clear a path and slide the target block to the goal before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 12
        self.GRID_HEIGHT = 8
        self.CELL_SIZE = 42
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2 + 20
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60

        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 62)
        self.COLOR_GOAL = (255, 215, 0)
        self.COLOR_GOAL_SHADOW = (200, 160, 0)
        self.COLOR_BLOCK_NORMAL = (100, 150, 255)
        self.COLOR_BLOCK_NORMAL_SHADOW = (70, 110, 205)
        self.COLOR_BLOCK_RISK = (60, 90, 220)
        self.COLOR_BLOCK_RISK_SHADOW = (40, 60, 170)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_CURSOR = (255, 100, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- State Variables (initialized in reset) ---
        self.grid = None
        self.goal_pos = None
        self.blocks = None
        self.cursor_pos = None
        self.last_move_direction = None
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.particles = []
        self.previous_space_held = False
        self.steps = 0
        self.win_state = False

        # --- Initial Reset ---
        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS
        self.game_over = False
        self.win_state = False
        self.cursor_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        self.last_move_direction = np.array([0, -1])  # Default up
        self.particles = []
        self.previous_space_held = False
        self.steps = 0

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), -1, dtype=int)
        self.blocks = []

        # Place goal
        goal_x = self.np_random.integers(1, self.GRID_WIDTH - 1)
        goal_y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
        self.goal_pos = np.array([goal_x, goal_y])

        # Place key block next to goal, ensuring a valid push spot
        offsets = [np.array([0, -1]), np.array([0, 1]), np.array([-1, 0]), np.array([1, 0])]
        self.np_random.shuffle(offsets)

        key_placed = False
        for offset in offsets:
            key_pos = self.goal_pos + offset
            push_pos = key_pos + offset
            if (0 <= key_pos[0] < self.GRID_WIDTH and 0 <= key_pos[1] < self.GRID_HEIGHT and
                0 <= push_pos[0] < self.GRID_WIDTH and 0 <= push_pos[1] < self.GRID_HEIGHT):
                self._add_block(key_pos[0], key_pos[1], is_risk=True)
                key_placed = True
                break
        
        if not key_placed: # Fallback if goal is in a corner
            self.reset() # Just try again
            return

        # Place other blocks
        num_blocks = self.np_random.integers(self.GRID_WIDTH, self.GRID_WIDTH * 2)
        for _ in range(num_blocks):
            for _ in range(50):  # Attempts to find empty spot
                bx, by = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
                if self.grid[bx, by] == -1 and not np.array_equal([bx, by], self.goal_pos):
                    is_risk = self.np_random.random() < 0.2
                    self._add_block(bx, by, is_risk=is_risk)
                    break

    def _add_block(self, x, y, is_risk):
        block_id = len(self.blocks)
        self.grid[x, y] = block_id
        self.blocks.append({
            'id': block_id,
            'pos': np.array([x, y]),
            'anim_pos': np.array([x, y], dtype=float),
            'is_risk': is_risk,
        })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Time penalty per frame

        # Unpack action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused

        # 1. Update cursor based on discrete movement actions
        if movement != 0:
            new_cursor_pos = self.cursor_pos.copy()
            if movement == 1: new_cursor_pos[1] -= 1; self.last_move_direction = np.array([0, -1])
            elif movement == 2: new_cursor_pos[1] += 1; self.last_move_direction = np.array([0, 1])
            elif movement == 3: new_cursor_pos[0] -= 1; self.last_move_direction = np.array([-1, 0])
            elif movement == 4: new_cursor_pos[0] += 1; self.last_move_direction = np.array([1, 0])
            
            self.cursor_pos[0] = np.clip(new_cursor_pos[0], 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(new_cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # 2. Handle push action on key press
        push_triggered = space_held and not self.previous_space_held
        if push_triggered:
            block_id = self.grid[self.cursor_pos[0], self.cursor_pos[1]]
            if block_id != -1:
                push_reward = self._push_block(block_id, self.last_move_direction)
                reward += push_reward
                self.score += push_reward
        self.previous_space_held = space_held

        # 3. Update game state
        self.time_remaining -= 1
        self._update_animations()
        self._update_particles()

        # 4. Check for termination
        terminated = self.game_over or self.time_remaining <= 0
        if terminated and not self.game_over:
            self.game_over = True # Mark as game over on timeout

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _push_block(self, block_id, direction):
        block = self.blocks[block_id]
        start_pos = block['pos'].copy()
        
        # Manhattan distance before push
        dist_before = np.sum(np.abs(start_pos - self.goal_pos))

        line_of_blocks = []
        current_pos = start_pos.copy()
        while 0 <= current_pos[0] < self.GRID_WIDTH and 0 <= current_pos[1] < self.GRID_HEIGHT:
            b_id = self.grid[current_pos[0], current_pos[1]]
            if b_id != -1:
                line_of_blocks.append(self.blocks[b_id])
                current_pos += direction
            else:
                break

        final_pos_of_line = current_pos
        can_move = 0 <= final_pos_of_line[0] < self.GRID_WIDTH and 0 <= final_pos_of_line[1] < self.GRID_HEIGHT

        if can_move:
            # Move all blocks
            for b in reversed(line_of_blocks):
                old_pos = b['pos']
                new_pos = old_pos + direction
                self.grid[old_pos[0], old_pos[1]] = -1
                self.grid[new_pos[0], new_pos[1]] = b['id']
                b['pos'] = new_pos
            
            # sfx: block_slide.wav
            self._create_slide_particles(start_pos, direction)

            # Check for win condition
            pushed_block_new_pos = start_pos + direction
            if np.array_equal(pushed_block_new_pos, self.goal_pos):
                self.game_over = True
                self.win_state = True
                # sfx: win_jingle.wav
                return 100

            dist_after = np.sum(np.abs(pushed_block_new_pos - self.goal_pos))
            reward = 0
            if dist_after < dist_before:
                reward += 1
                if block['is_risk']: reward += 5
            else:
                reward -= 1
                if block['is_risk']: reward -= 2
            return reward
        else:
            # sfx: thud.wav
            self._create_thud_particles(start_pos, direction)
            return -0.5

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_goal()
        self._render_blocks()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": int(self.score), "steps": self.steps, "time_left": self.time_remaining / self.FPS}

    def _update_animations(self):
        for block in self.blocks:
            block['anim_pos'] = block['anim_pos'] * 0.6 + np.array(block['pos']) * 0.4

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _world_to_screen(self, x, y):
        return (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)

    def _render_goal(self):
        x, y = self._world_to_screen(self.goal_pos[0], self.goal_pos[1])
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        shadow_rect = pygame.Rect(x, y + 4, self.CELL_SIZE, self.CELL_SIZE)
        
        pygame.draw.rect(self.screen, self.COLOR_GOAL_SHADOW, shadow_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, rect, border_radius=6)
        
        # Draw star
        star_points = []
        cx, cy = x + self.CELL_SIZE / 2, y + self.CELL_SIZE / 2
        outer_r = self.CELL_SIZE * 0.35
        inner_r = self.CELL_SIZE * 0.15
        for i in range(5):
            angle = math.pi / 2 + (2 * math.pi * i / 5)
            star_points.append((cx + outer_r * math.cos(angle), cy - outer_r * math.sin(angle)))
            angle += math.pi / 5
            star_points.append((cx + inner_r * math.cos(angle), cy - inner_r * math.sin(angle)))
        pygame.gfxdraw.aapolygon(self.screen, star_points, self.COLOR_GOAL_SHADOW)
        pygame.gfxdraw.filled_polygon(self.screen, star_points, self.COLOR_GOAL_SHADOW)

    def _render_blocks(self):
        for block in self.blocks:
            x, y = self._world_to_screen(block['anim_pos'][0], block['anim_pos'][1])
            is_risk = block['is_risk']
            color = self.COLOR_BLOCK_RISK if is_risk else self.COLOR_BLOCK_NORMAL
            shadow_color = self.COLOR_BLOCK_RISK_SHADOW if is_risk else self.COLOR_BLOCK_NORMAL_SHADOW
            
            shadow_rect = pygame.Rect(int(x) + 3, int(y) + 3, self.CELL_SIZE - 6, self.CELL_SIZE - 6)
            main_rect = pygame.Rect(int(x) + 3, int(y), self.CELL_SIZE - 6, self.CELL_SIZE - 6)

            pygame.draw.rect(self.screen, shadow_color, shadow_rect, border_radius=4)
            pygame.draw.rect(self.screen, color, main_rect, border_radius=4)

    def _render_cursor(self):
        x, y = self._world_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.01)
        cursor_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 3, border_radius=6)
        self.screen.blit(cursor_surface, (x, y))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        secs = int(self.time_remaining / self.FPS)
        mins = secs // 60
        secs %= 60
        time_text = self.font_ui.render(f"TIME: {mins:02}:{secs:02}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "LEVEL CLEARED!" if self.win_state else "TIME'S UP!"
        color = self.COLOR_GOAL if self.win_state else self.COLOR_CURSOR
        
        text = self.font_game_over.render(message, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def _create_slide_particles(self, pos, direction):
        px, py = self._world_to_screen(pos[0], pos[1])
        px += self.CELL_SIZE / 2
        py += self.CELL_SIZE / 2
        for _ in range(15):
            angle = math.atan2(-direction[1], -direction[0]) + (self.np_random.random() - 0.5) * math.pi / 2
            speed = 2 + self.np_random.random() * 3
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': 20, 'color': (200, 200, 220)})

    def _create_thud_particles(self, pos, direction):
        px, py = self._world_to_screen(pos[0] + direction[0], pos[1] + direction[1])
        px += self.CELL_SIZE / 2 - direction[0] * self.CELL_SIZE/2
        py += self.CELL_SIZE / 2 - direction[1] * self.CELL_SIZE/2
        for _ in range(5):
            angle = math.atan2(direction[1], direction[0]) + (self.np_random.random() - 0.5) * math.pi/4
            speed = 1 + self.np_random.random() * 2
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': 10, 'color': (150, 150, 150)})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            size = max(0, int(p['life'] * 0.2))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def validate_implementation(self):
        print("Running implementation validation...")
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

    def close(self):
        pygame.quit()