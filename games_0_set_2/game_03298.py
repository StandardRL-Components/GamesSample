
# Generated: 2025-08-27T22:56:54.571764
# Source Brief: brief_03298.md
# Brief Index: 3298

        
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
        "Controls: Arrow keys to move selector. Space to clear a group of matching tiles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the grid by selecting groups of two or more matching colored tiles. You have a limited number of moves. Plan your clicks to clear the whole board!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_ROWS = 8
    GRID_COLS = 12
    TILE_SIZE = 40
    GRID_LINE_WIDTH = 2
    MAX_MOVES = 20
    MAX_STEPS = 1000
    NUM_COLORS = 4  # Excluding empty

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    TILE_COLORS = [
        (0, 0, 0),  # 0: Empty
        (255, 80, 80),   # 1: Red
        (80, 200, 255),  # 2: Blue
        (80, 255, 80),   # 3: Green
        (255, 240, 80),  # 4: Yellow
    ]
    COLOR_TEXT = (240, 240, 240)
    
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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Grid positioning
        self.grid_width_px = self.GRID_COLS * self.TILE_SIZE
        self.grid_height_px = self.GRID_ROWS * self.TILE_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width_px) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height_px) // 2

        # Initialize state variables
        self.grid = None
        self.selector_pos = None
        self.moves_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self._np_random = None

        self.reset()
        
        # This check is for development and can be removed in production
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.particles = []
        self.selector_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        
        # Generate a valid starting board
        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_ROWS, self.GRID_COLS))
            if self._has_valid_move():
                break

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        
        reward = 0
        self.steps += 1

        # --- Handle movement ---
        if movement == 1: # Up
            self.selector_pos[0] = max(0, self.selector_pos[0] - 1)
        elif movement == 2: # Down
            self.selector_pos[0] = min(self.GRID_ROWS - 1, self.selector_pos[0] + 1)
        elif movement == 3: # Left
            self.selector_pos[1] = max(0, self.selector_pos[1] - 1)
        elif movement == 4: # Right
            self.selector_pos[1] = min(self.GRID_COLS - 1, self.selector_pos[1] + 1)

        # --- Handle tile clearing ---
        if space_pressed:
            reward = self._handle_click()

        # --- Check termination conditions ---
        terminated = self._check_termination()
        if terminated and not self.win and np.sum(self.grid > 0) == 0:
            self.win = True
            reward += 50 # Win bonus
            self.score += 50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_click(self):
        r, c = self.selector_pos
        tile_color = self.grid[r, c]

        if tile_color == 0: # Clicked an empty space
            return 0
        
        connected_tiles = self._find_connected_tiles(r, c)
        
        if len(connected_tiles) < 2: # Not a valid group
            # SFX: Negative beep
            return 0

        # Valid move
        self.moves_remaining -= 1
        num_cleared = len(connected_tiles)
        
        for tr, tc in connected_tiles:
            self.grid[tr, tc] = 0
            self._spawn_particles(tr, tc, tile_color)
        
        self._apply_gravity_and_refill()
        
        # SFX: Pop/clear sound, scaled by num_cleared
        self.score += num_cleared
        return num_cleared

    def _find_connected_tiles(self, start_r, start_c):
        target_color = self.grid[start_r, start_c]
        if target_color == 0:
            return []

        q = deque([(start_r, start_c)])
        visited = set([(start_r, start_c)])
        connected = []

        while q:
            r, c = q.popleft()
            connected.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                    if (nr, nc) not in visited and self.grid[nr, nc] == target_color:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return connected

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1
            
            # Refill top rows with new random tiles
            for r in range(empty_row, -1, -1):
                self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _has_valid_move(self):
        visited = np.zeros_like(self.grid, dtype=bool)
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if not visited[r,c] and self.grid[r,c] != 0:
                    connected = self._find_connected_tiles(r, c)
                    if len(connected) >= 2:
                        return True
                    for tr, tc in connected:
                        visited[tr, tc] = True
        return False

    def _check_termination(self):
        if self.game_over:
            return True
        
        board_cleared = np.sum(self.grid > 0) == 0
        if board_cleared:
            self.game_over = True
            self.win = True
            return True

        if self.moves_remaining <= 0:
            self.game_over = True
            return True

        if not self._has_valid_move(): # No more possible moves
            self.game_over = True
            return True

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.grid_offset_y + r * self.TILE_SIZE
            start = (self.grid_offset_x, y)
            end = (self.grid_offset_x + self.grid_width_px, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, self.GRID_LINE_WIDTH)
        for c in range(self.GRID_COLS + 1):
            x = self.grid_offset_x + c * self.TILE_SIZE
            start = (x, self.grid_offset_y)
            end = (x, self.grid_offset_y + self.grid_height_px)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, self.GRID_LINE_WIDTH)

        # Draw tiles
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_idx = self.grid[r, c]
                if color_idx > 0:
                    color = self.TILE_COLORS[color_idx]
                    rect = pygame.Rect(
                        self.grid_offset_x + c * self.TILE_SIZE,
                        self.grid_offset_y + r * self.TILE_SIZE,
                        self.TILE_SIZE, self.TILE_SIZE
                    )
                    # Draw a slight 3D effect
                    darker_color = tuple(max(0, val - 40) for val in color)
                    brighter_color = tuple(min(255, val + 40) for val in color)
                    
                    pygame.draw.rect(self.screen, darker_color, rect)
                    inner_rect = rect.inflate(-self.GRID_LINE_WIDTH*2, -self.GRID_LINE_WIDTH*2)
                    pygame.draw.rect(self.screen, color, inner_rect)

        # Draw selector
        sel_r, sel_c = self.selector_pos
        sel_rect = pygame.Rect(
            self.grid_offset_x + sel_c * self.TILE_SIZE,
            self.grid_offset_y + sel_r * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        # Flashing effect for selector
        alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.01)
        pygame.gfxdraw.rectangle(self.screen, sel_rect, (255, 255, 255, alpha))
        pygame.gfxdraw.rectangle(self.screen, sel_rect.inflate(-2,-2), (255, 255, 255, alpha))

    def _spawn_particles(self, grid_r, grid_c, color_idx):
        center_x = self.grid_offset_x + grid_c * self.TILE_SIZE + self.TILE_SIZE / 2
        center_y = self.grid_offset_y + grid_r * self.TILE_SIZE + self.TILE_SIZE / 2
        color = self.TILE_COLORS[color_idx]

        for _ in range(10): # Spawn 10 particles
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifespan = random.randint(15, 30) # frames
            self.particles.append([center_x, center_y, vx, vy, lifespan, color])

    def _update_and_render_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # lifespan--
            
            if p[4] <= 0:
                self.particles.pop(i)
            else:
                # Fade out effect
                alpha = max(0, min(255, int(255 * (p[4] / 20))))
                color = p[5]
                size = int(max(1, 5 * (p[4] / 20)))
                
                # Create a temporary surface for alpha blending
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color + (alpha,), (size, size), size)
                self.screen.blit(particle_surf, (int(p[0] - size), int(p[1] - size)))

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_main.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (150, 255, 150) if self.win else (255, 150, 150)
            
            end_text = self.font_main.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "win": self.win,
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Color Grid Puzzle")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Game loop
    while not terminated:
        # --- Human Controls ---
        movement = 0 # no-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                elif event.key == pygame.K_q:
                    terminated = True

        # Only step if an action is taken
        if movement != 0 or space != 0:
            action = [movement, space, 0] # shift is unused
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Score: {info['score']}, Moves: {info['moves_remaining']}, Terminated: {terminated}")

        # --- Rendering ---
        # The observation is the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()