
# Generated: 2025-08-27T14:28:12.516215
# Source Brief: brief_00686.md
# Brief Index: 686

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a block group. Hold shift to restart."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced color matching puzzle. Clear blocks by selecting groups of the same color to score points against the clock."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
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
        
        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (40, 44, 52)
        self.COLOR_GRID = (62, 68, 81)
        self.COLOR_CURSOR = (255, 255, 255)
        self.BLOCK_COLORS = [
            (224, 108, 117), # Red
            (152, 195, 121), # Green
            (97, 175, 239),  # Blue
            (229, 192, 123), # Yellow
            (198, 120, 221), # Purple
            (209, 154, 102), # Orange
        ]
        self.COLOR_EMPTY = (48, 52, 60)
        self.COLOR_TEXT = (220, 220, 220)

        # Game settings
        self.GRID_COLS = 16
        self.GRID_ROWS = 10
        self.BLOCK_SIZE = 40
        self.GRID_OFFSET_X = (640 - self.GRID_COLS * self.BLOCK_SIZE) // 2
        self.GRID_OFFSET_Y = (400 - self.GRID_ROWS * self.BLOCK_SIZE) // 2
        self.MAX_TIMER = 60.0
        self.FPS = 30
        self.WIN_PERCENTAGE = 0.5
        
        # Animation timings (in frames)
        self.CLEAR_ANIM_DUR = 8
        self.FALL_SPEED = 20 # pixels per frame

        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win_condition_met = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.game_state = None # 'IDLE', 'CLEARING', 'FALLING'
        self.particles = None
        self.animations = None
        self.np_random = None

        # Etc...        
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self.np_random.integers(0, len(self.BLOCK_COLORS), size=(self.GRID_COLS, self.GRID_ROWS))
        self.total_blocks = self.GRID_COLS * self.GRID_ROWS
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.timer = self.MAX_TIMER
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_condition_met = False
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.game_state = 'IDLE'
        self.particles = []
        self.animations = [] # {'type': 'clear'/'fall', 'data': ...}
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0.0
        self.steps += 1
        
        if not self.game_over:
            self.timer = max(0, self.timer - 1.0 / self.FPS)
        
        self._update_animations()

        if self.game_state == 'IDLE' and not self.game_over:
            # Handle cursor movement
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

            # Handle selection (on rising edge)
            if space_held and not self.prev_space_held:
                reward += self._handle_selection()

            # Handle restart (on rising edge)
            if shift_held and not self.prev_shift_held:
                self.game_over = True
                reward = -100.0 # Penalty for manual reset

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Check for termination
        cleared_pct = np.sum(self.grid == -1) / self.total_blocks if self.grid is not None else 0
        if not self.game_over and cleared_pct >= self.WIN_PERCENTAGE:
            self.game_over = True
            self.win_condition_met = True
            reward += 100.0
        elif not self.game_over and self.timer <= 0:
            self.game_over = True
            self.win_condition_met = False
            reward += -100.0

        terminated = self.game_over and self.game_state == 'IDLE'

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_selection(self):
        x, y = self.cursor_pos
        if self.grid[x, y] == -1:
            return 0

        color_to_match = self.grid[x, y]
        
        q = [(x, y)]
        visited = set([(x, y)])
        connected_blocks = []

        while q:
            cx, cy = q.pop(0)
            connected_blocks.append((cx, cy))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and \
                   (nx, ny) not in visited and self.grid[nx, ny] == color_to_match:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        
        if len(connected_blocks) > 1:
            # Sfx: Block clear sound
            for bx, by in connected_blocks:
                self.animations.append({
                    'type': 'clear', 'pos': (bx, by), 'color_idx': self.grid[bx, by], 'progress': 0
                })
                self.grid[bx, by] = -1
                self._spawn_particles(bx, by, color_to_match)

            self.game_state = 'CLEARING'
            
            # Calculate reward
            cleared_count = len(connected_blocks)
            self.score += cleared_count
            reward = float(cleared_count)
            if cleared_count > 10:
                reward += 10.0
                self.score += 10
            return reward
            
        return 0

    def _spawn_particles(self, grid_x, grid_y, color_idx):
        center_x = self.GRID_OFFSET_X + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        center_y = self.GRID_OFFSET_Y + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
        color = self.BLOCK_COLORS[color_idx]
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            life = self.np_random.integers(15, 30)
            self.particles.append([
                center_x, center_y, 
                math.cos(angle) * speed, math.sin(angle) * speed,
                life, color
            ])

    def _update_animations(self):
        # Update particles
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # life -= 1
        
        # Update main animations
        if self.game_state == 'CLEARING':
            self.animations = [a for a in self.animations if a['progress'] < self.CLEAR_ANIM_DUR]
            for anim in self.animations:
                if anim['type'] == 'clear':
                    anim['progress'] += 1
            if not any(a['type'] == 'clear' for a in self.animations):
                self._prepare_fall_and_refill()
                if any(a['type'] == 'fall' for a in self.animations):
                    self.game_state = 'FALLING'
                else:
                    self.game_state = 'IDLE'

        elif self.game_state == 'FALLING':
            all_settled = True
            for anim in self.animations:
                if anim['type'] == 'fall':
                    anim['y'] += self.FALL_SPEED
                    if anim['y'] >= anim['target_y']:
                        anim['y'] = anim['target_y']
                        self.grid[anim['pos'][0], anim['pos'][1]] = anim['color_idx']
                    else:
                        all_settled = False
            
            if all_settled:
                self.animations = []
                self.game_state = 'IDLE'

    def _prepare_fall_and_refill(self):
        # Sfx: Blocks falling
        new_grid = np.full((self.GRID_COLS, self.GRID_ROWS), -1, dtype=int)
        for x in range(self.GRID_COLS):
            write_y = self.GRID_ROWS - 1
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[x, y] != -1:
                    color_idx = self.grid[x, y]
                    new_grid[x, write_y] = color_idx
                    start_y = self.GRID_OFFSET_Y + y * self.BLOCK_SIZE
                    target_y = self.GRID_OFFSET_Y + write_y * self.BLOCK_SIZE
                    if start_y != target_y:
                        self.animations.append({
                            'type': 'fall', 'pos': (x, write_y), 'color_idx': color_idx,
                            'y': start_y, 'target_y': target_y
                        })
                    write_y -= 1
        
        # Refill
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if new_grid[x, y] == -1:
                    color_idx = self.np_random.integers(0, len(self.BLOCK_COLORS))
                    new_grid[x, y] = color_idx
                    start_y = self.GRID_OFFSET_Y + (y - self.GRID_ROWS) * self.BLOCK_SIZE
                    target_y = self.GRID_OFFSET_Y + y * self.BLOCK_SIZE
                    self.animations.append({
                        'type': 'fall', 'pos': (x, y), 'color_idx': color_idx,
                        'y': start_y, 'target_y': target_y
                    })

        self.grid = new_grid.copy()
        # Clear grid for falling blocks so they don't get drawn twice
        for anim in self.animations:
            if anim['type'] == 'fall':
                self.grid[anim['pos'][0], anim['pos'][1]] = -1

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_COLS * self.BLOCK_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_ROWS * self.BLOCK_SIZE))

        # Draw static blocks
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.grid[c, r]
                color = self.COLOR_EMPTY if color_idx == -1 else self.BLOCK_COLORS[color_idx]
                rect = (self.GRID_OFFSET_X + c * self.BLOCK_SIZE + 1, self.GRID_OFFSET_Y + r * self.BLOCK_SIZE + 1, self.BLOCK_SIZE - 2, self.BLOCK_SIZE - 2)
                pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Draw animations
        for anim in self.animations:
            if anim['type'] == 'clear':
                c, r = anim['pos']
                color = self.BLOCK_COLORS[anim['color_idx']]
                progress = anim['progress'] / self.CLEAR_ANIM_DUR
                size = self.BLOCK_SIZE * (1 - progress)
                offset = (self.BLOCK_SIZE - size) / 2
                rect = (self.GRID_OFFSET_X + c * self.BLOCK_SIZE + offset, self.GRID_OFFSET_Y + r * self.BLOCK_SIZE + offset, size, size)
                pygame.draw.rect(self.screen, color, rect, border_radius=int(4 * (1 - progress)))
            elif anim['type'] == 'fall':
                c, _ = anim['pos']
                color = self.BLOCK_COLORS[anim['color_idx']]
                rect = (self.GRID_OFFSET_X + c * self.BLOCK_SIZE + 1, anim['y'] + 1, self.BLOCK_SIZE - 2, self.BLOCK_SIZE - 2)
                pygame.draw.rect(self.screen, color, rect, border_radius=4)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p[4] / 30.0))))
            color = p[5]
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 2, (*color, alpha))
            
        # Draw cursor
        if self.game_state == 'IDLE' and not self.game_over:
            c, r = self.cursor_pos
            rect = (self.GRID_OFFSET_X + c * self.BLOCK_SIZE, self.GRID_OFFSET_Y + r * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=5)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Timer
        timer_color = self.COLOR_TEXT
        if self.timer < 10: timer_color = (255, 80, 80) # Red
        elif self.timer < 30: timer_color = (255, 255, 80) # Yellow
        timer_surf = self.font_ui.render(f"TIME: {self.timer:.1f}", True, timer_color)
        self.screen.blit(timer_surf, (630 - timer_surf.get_width(), 10))
        
        # Cleared Percentage
        cleared_pct = np.sum(self.grid == -1) / self.total_blocks if self.grid is not None else 0
        cleared_surf = self.font_ui.render(f"CLEARED: {cleared_pct:.0%}", True, self.COLOR_TEXT)
        self.screen.blit(cleared_surf, (640 // 2 - cleared_surf.get_width() // 2, 390 - cleared_surf.get_height()))

        # Game Over message
        if self.game_over and self.game_state == 'IDLE':
            msg = "YOU WIN!" if self.win_condition_met else "TIME'S UP!"
            msg_surf = self.font_msg.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(640 // 2, 400 // 2))
            
            # Draw a semi-transparent background for the message
            bg_rect = msg_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((*self.COLOR_BG, 200))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cleared_percentage": np.sum(self.grid == -1) / self.total_blocks if self.grid is not None else 0
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Color Grid Puzzle")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0)

    print(env.user_guide)

    while not done:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        mov = 0
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = np.array([mov, space, shift])

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds before closing

        clock.tick(env.FPS)

    env.close()