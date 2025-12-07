
# Generated: 2025-08-28T02:17:27.549248
# Source Brief: brief_01653.md
# Brief Index: 1653

        
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
        "Controls: Arrow keys to move cursor. Space to select a block group. Shift to restart."
    )

    game_description = (
        "Match groups of adjacent colored blocks to score points. Reach 1000 points to win. The game ends if no more matches are possible."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_COLORS = 5
        self.TARGET_SCORE = 1000
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (30, 35, 40)
        self.COLOR_GRID = (50, 55, 60)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Sizing and Layout ---
        self.GRID_AREA_SIZE = self.HEIGHT - 40
        self.BLOCK_SIZE = self.GRID_AREA_SIZE // self.GRID_SIZE
        self.GRID_START_X = (self.WIDTH - self.GRID_AREA_SIZE) // 2
        self.GRID_START_Y = (self.HEIGHT - self.GRID_AREA_SIZE) // 2

        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            if self._check_for_valid_moves():
                break

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, any action just returns the final state
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        self.steps += 1
        self.particles = [] # Clear particles from previous step

        if shift_press:
            terminated = True
            self.game_over = True
            reward = -10
            self.win_message = "Game Reset"

        elif space_press:
            # --- Block Selection Logic ---
            group = self._find_match_group(self.cursor_pos[0], self.cursor_pos[1])
            if len(group) > 1:
                # Successful match
                # SFX: Block clear sound
                num_cleared = len(group)
                reward += num_cleared
                if num_cleared >= 4:
                    reward += 10 # Bonus
                    # SFX: Bonus sound
                
                self.score += reward

                # Create particles and remove blocks
                for x, y in group:
                    self._create_particles(x, y)
                    self.grid[y, x] = 0
                
                self._apply_gravity_and_refill()

                if not self._check_for_valid_moves():
                    terminated = True
                    self.game_over = True
                    reward += -10
                    self.win_message = "No More Moves!"
            else:
                # Invalid move
                reward = -0.2
                self.score += reward
                # SFX: Error/buzz sound
        else:
            # --- Cursor Movement Logic ---
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
            # movement == 0 is a no-op, reward is 0

        # --- Check Termination Conditions ---
        if self.score >= self.TARGET_SCORE:
            terminated = True
            self.game_over = True
            reward += 100
            self.win_message = "You Win!"
        
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True
            self.win_message = "Time's Up!"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_match_group(self, start_x, start_y):
        if self.grid[start_y, start_x] == 0:
            return []

        target_color = self.grid[start_y, start_x]
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        group = []

        while q:
            x, y = q.popleft()
            group.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                    if (nx, ny) not in visited and self.grid[ny, nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_SIZE):
            col = self.grid[:, x]
            empty_cells = np.where(col == 0)[0]
            non_empty_cells = np.where(col != 0)[0]
            
            if len(empty_cells) > 0:
                new_col = np.concatenate((col[non_empty_cells], col[empty_cells]))
                # Invert to make blocks fall down
                self.grid[:, x] = new_col[::-1]
        
        # Refill top
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y, x] == 0:
                    self.grid[y, x] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _check_for_valid_moves(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                # Check right neighbor
                if x < self.GRID_SIZE - 1 and self.grid[y, x] == self.grid[y, x + 1] and self.grid[y,x] != 0:
                    return True
                # Check bottom neighbor
                if y < self.GRID_SIZE - 1 and self.grid[y, x] == self.grid[y + 1, x] and self.grid[y,x] != 0:
                    return True
        return False
        
    def _create_particles(self, grid_x, grid_y):
        px = self.GRID_START_X + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        py = self.GRID_START_Y + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        color_index = self.grid[grid_y, grid_x] - 1
        if color_index < 0: return
        color = self.BLOCK_COLORS[color_index]
        
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [px, py], 'vel': vel, 'radius': self.np_random.uniform(3, 6), 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_START_X + i * self.BLOCK_SIZE
            y = self.GRID_START_Y + i * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_START_Y), (x, self.GRID_START_Y + self.GRID_AREA_SIZE))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_START_X, y), (self.GRID_START_X + self.GRID_AREA_SIZE, y))

        # Draw blocks
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color_index = self.grid[y, x] - 1
                if color_index >= 0:
                    color = self.BLOCK_COLORS[color_index]
                    rect = pygame.Rect(
                        self.GRID_START_X + x * self.BLOCK_SIZE,
                        self.GRID_START_Y + y * self.BLOCK_SIZE,
                        self.BLOCK_SIZE, self.BLOCK_SIZE
                    )
                    
                    # 3D effect
                    shadow_color = tuple(max(0, c - 40) for c in color)
                    highlight_color = tuple(min(255, c + 40) for c in color)
                    
                    pygame.draw.rect(self.screen, shadow_color, rect.move(2, 2))
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, highlight_color, (rect.x, rect.y, rect.width, 2))
                    pygame.draw.rect(self.screen, highlight_color, (rect.x, rect.y, 2, rect.height))
                    pygame.gfxdraw.rectangle(self.screen, rect, (0,0,0,50))

        # Draw particles (since auto_advance=False, these only appear for one frame)
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

        # Draw cursor
        cursor_x = self.GRID_START_X + self.cursor_pos[0] * self.BLOCK_SIZE
        cursor_y = self.GRID_START_Y + self.cursor_pos[1] * self.BLOCK_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        # Pulsating effect for cursor
        pulse = (math.sin(self.steps * 0.5) + 1) / 2 # Use steps for deterministic pulse
        alpha = 100 + 155 * pulse
        
        # Create a temporary surface for alpha blending
        s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_WHITE, alpha), (0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE), 4, border_radius=3)
        self.screen.blit(s, (cursor_x, cursor_y))

    def _render_ui(self):
        # Draw score
        score_text = f"Score: {int(self.score)}"
        self._draw_text(score_text, self.font_medium, self.COLOR_WHITE, (10, 10))

        # Draw steps
        steps_text = f"Moves: {self.steps}"
        self._draw_text(steps_text, self.font_small, self.COLOR_WHITE, (self.WIDTH - 100, 10))

        # Draw game over message
        if self.game_over:
            self._draw_text(self.win_message, self.font_large, self.COLOR_WHITE, 
                            (self.WIDTH // 2, self.HEIGHT // 2), center=True, shadow=True)

    def _draw_text(self, text, font, color, pos, center=False, shadow=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos

        if shadow:
            shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surface, text_rect.move(3, 3))
        
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": list(self.cursor_pos),
            "is_game_over": self.game_over,
        }

    def close(self):
        pygame.quit()
        
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
    terminated = False
    
    # Override screen to be a display surface
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Matcher")
    clock = pygame.time.Clock()

    print(env.user_guide)

    while not terminated:
        # --- Human Controls ---
        movement = 0 # no-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift = 1
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()

        action = [movement, space, shift]
        
        # Only step if an action is taken, because auto_advance is False
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.1f}, Terminated: {terminated}")

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        # Pygame uses (width, height), numpy uses (height, width), so we transpose
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for human play

    print("Game Over!")
    # Keep window open for a bit to show final screen
    end_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - end_time < 3000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        pygame.display.flip()
        clock.tick(30)
        
    env.close()