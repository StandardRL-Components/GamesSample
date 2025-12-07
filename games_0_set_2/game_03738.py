
# Generated: 2025-08-28T00:16:18.707060
# Source Brief: brief_03738.md
# Brief Index: 3738

        
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
        "Use arrow keys to move the selector. Press space to select a block and clear groups of 3 or more."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Match colored blocks to clear the board before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 8
    MAX_MOVES = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_PANEL = (30, 35, 55, 180)
    COLOR_SELECTOR = (255, 255, 0)
    COLOR_GRID_LINE = (50, 60, 80)
    BLOCK_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 120, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
        (255, 140, 50),   # Orange
    ]

    # Isometric projection constants
    TILE_WIDTH = 60
    TILE_HEIGHT = TILE_WIDTH // 2
    TILE_WIDTH_HALF = TILE_WIDTH // 2
    TILE_HEIGHT_HALF = TILE_HEIGHT // 2
    BLOCK_Z_HEIGHT = 22
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 100

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Generate darker side colors for 3D effect
        self.BLOCK_SIDE_COLORS = [
            tuple(max(0, int(c * 0.7)) for c in color) for color in self.BLOCK_COLORS
        ]
        
        # Initialize state variables
        self.board = None
        self.selector_pos = None
        self.moves_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.particles = []
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.prev_space_held = False
        self.particles = []

        # Generate a board with at least one possible match
        while True:
            self._generate_board()
            if self._is_match_possible():
                break
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = 0
        
        # --- Handle Input ---
        self._handle_movement(movement)
        
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        if space_pressed:
            self.moves_remaining -= 1
            cleared_count = self._handle_selection()
            if cleared_count > 0:
                # sfx: positive match sound
                reward += cleared_count # +1 per block
            else:
                # sfx: negative buzz sound
                pass # No reward for failed match

        self._update_particles()
        self.steps += 1
        
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            is_win = np.all(self.board == 0)
            if is_win:
                # sfx: victory fanfare
                reward += 100
            else:
                # sfx: game over sound
                reward += -100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
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
            "moves_remaining": self.moves_remaining,
        }

    def _generate_board(self):
        self.board = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1, size=(self.GRID_SIZE, self.GRID_SIZE))

    def _is_match_possible(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color = self.board[r, c]
                if color == 0: continue
                # Check horizontal match
                if c < self.GRID_SIZE - 2 and self.board[r, c+1] == color and self.board[r, c+2] == color:
                    return True
                # Check vertical match
                if r < self.GRID_SIZE - 2 and self.board[r+1, c] == color and self.board[r+2, c] == color:
                    return True
        return False

    def _handle_movement(self, movement):
        r, c = self.selector_pos
        if movement == 1: # Up
            r = max(0, r - 1)
        elif movement == 2: # Down
            r = min(self.GRID_SIZE - 1, r + 1)
        elif movement == 3: # Left
            c = max(0, c - 1)
        elif movement == 4: # Right
            c = min(self.GRID_SIZE - 1, c + 1)
        
        if self.selector_pos != [r, c]:
            # sfx: cursor move tick
            self.selector_pos = [r, c]

    def _handle_selection(self):
        r, c = self.selector_pos
        if self.board[r, c] == 0:
            return 0

        connected_blocks = self._find_connected_blocks(r, c)
        
        if len(connected_blocks) >= 3:
            color_index = self.board[r, c] - 1
            for br, bc in connected_blocks:
                self._spawn_particles(br, bc, self.BLOCK_COLORS[color_index])
                self.board[br, bc] = 0
            
            self.score += len(connected_blocks)
            self._apply_gravity()
            return len(connected_blocks)
        
        return 0

    def _find_connected_blocks(self, r_start, c_start):
        target_color = self.board[r_start, c_start]
        if target_color == 0:
            return []

        q = [(r_start, c_start)]
        visited = set(q)
        connected = []

        while q:
            r, c = q.pop(0)
            connected.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and (nr, nc) not in visited:
                    if self.board[nr, nc] == target_color:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return connected
    
    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.board[r, c] != 0:
                    if r != empty_row:
                        self.board[empty_row, c] = self.board[r, c]
                        self.board[r, c] = 0
                    empty_row -= 1
    
    def _check_termination(self):
        is_win = np.all(self.board == 0)
        is_loss = self.moves_remaining <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS
        return is_win or is_loss or max_steps_reached
    
    def _spawn_particles(self, r, c, color):
        center_x, center_y = self._grid_to_screen(r, c)
        center_y -= self.BLOCK_Z_HEIGHT / 2
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(15, 30)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _grid_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_WIDTH_HALF
        y = self.ORIGIN_Y + (c + r) * self.TILE_HEIGHT_HALF
        return x, y

    def _draw_iso_block(self, surf, r, c, color_index):
        x, y = self._grid_to_screen(r, c)
        
        top_color = self.BLOCK_COLORS[color_index - 1]
        side_color = self.BLOCK_SIDE_COLORS[color_index - 1]
        
        # Points for the cube
        p_top_front = (int(x), int(y))
        p_top_left = (int(x - self.TILE_WIDTH_HALF), int(y + self.TILE_HEIGHT_HALF))
        p_top_right = (int(x + self.TILE_WIDTH_HALF), int(y + self.TILE_HEIGHT_HALF))
        p_top_back = (int(x), int(y + self.TILE_HEIGHT))
        
        p_bottom_front = (int(x), int(y + self.BLOCK_Z_HEIGHT))
        p_bottom_left = (int(x - self.TILE_WIDTH_HALF), int(y + self.TILE_HEIGHT_HALF + self.BLOCK_Z_HEIGHT))
        p_bottom_right = (int(x + self.TILE_WIDTH_HALF), int(y + self.TILE_HEIGHT_HALF + self.BLOCK_Z_HEIGHT))

        # Draw sides first (painter's algorithm)
        # Left face
        pygame.gfxdraw.aapolygon(surf, [p_top_front, p_top_left, p_bottom_left, p_bottom_front], side_color)
        pygame.gfxdraw.filled_polygon(surf, [p_top_front, p_top_left, p_bottom_left, p_bottom_front], side_color)
        
        # Right face
        pygame.gfxdraw.aapolygon(surf, [p_top_front, p_top_right, p_bottom_right, p_bottom_front], side_color)
        pygame.gfxdraw.filled_polygon(surf, [p_top_front, p_top_right, p_bottom_right, p_bottom_front], side_color)

        # Draw top face
        pygame.gfxdraw.aapolygon(surf, [p_top_front, p_top_right, p_top_back, p_top_left], top_color)
        pygame.gfxdraw.filled_polygon(surf, [p_top_front, p_top_right, p_top_back, p_top_left], top_color)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_SIZE + 1):
            start_x, start_y = self._grid_to_screen(r - 1, -1)
            end_x, end_y = self._grid_to_screen(r - 1, self.GRID_SIZE - 1)
            pygame.draw.aaline(self.screen, self.COLOR_GRID_LINE, (start_x + self.TILE_WIDTH_HALF, start_y + self.TILE_HEIGHT_HALF), (end_x + self.TILE_WIDTH_HALF, end_y + self.TILE_HEIGHT_HALF))
        for c in range(self.GRID_SIZE + 1):
            start_x, start_y = self._grid_to_screen(-1, c - 1)
            end_x, end_y = self._grid_to_screen(self.GRID_SIZE - 1, c - 1)
            pygame.draw.aaline(self.screen, self.COLOR_GRID_LINE, (start_x + self.TILE_WIDTH_HALF, start_y + self.TILE_HEIGHT_HALF), (end_x + self.TILE_WIDTH_HALF, end_y + self.TILE_HEIGHT_HALF))

        # Draw blocks
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_index = self.board[r, c]
                if color_index > 0:
                    self._draw_iso_block(self.screen, r, c, color_index)

        # Draw selector
        sel_r, sel_c = self.selector_pos
        sel_x, sel_y = self._grid_to_screen(sel_r, sel_c)
        selector_poly = [
            (int(sel_x), int(sel_y)),
            (int(sel_x + self.TILE_WIDTH_HALF), int(sel_y + self.TILE_HEIGHT_HALF)),
            (int(sel_x), int(sel_y + self.TILE_HEIGHT)),
            (int(sel_x - self.TILE_WIDTH_HALF), int(sel_y + self.TILE_HEIGHT_HALF)),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_SELECTOR, selector_poly, 3)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20.0))))
            color = p['color'] + (alpha,)
            size = max(1, int(4 * (p['lifespan'] / 20.0)))
            pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        # UI Panel
        panel_rect = pygame.Rect(10, 10, 180, 80)
        panel_surf = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_surf.fill(self.COLOR_UI_PANEL)
        self.screen.blit(panel_surf, panel_rect.topleft)
        
        # Moves
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (25, 20))
        
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (25, 55))

        # Game Over Text
        if self.game_over:
            is_win = np.all(self.board == 0)
            msg = "YOU WIN!" if is_win else "GAME OVER"
            color = (180, 255, 180) if is_win else (255, 180, 180)
            
            text = self.font_main.render(msg, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            # Draw a backdrop for the text
            backdrop_rect = text_rect.inflate(20, 20)
            backdrop_surf = pygame.Surface(backdrop_rect.size, pygame.SRCALPHA)
            backdrop_surf.fill((0, 0, 0, 150))
            self.screen.blit(backdrop_surf, backdrop_rect)
            
            self.screen.blit(text, text_rect)

    def close(self):
        pygame.font.quit()
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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Isometric Block Matcher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    # Game loop
    running = True
    while running:
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward}, Score: {info['score']}, Moves Left: {info['moves_remaining']}")

        if terminated:
            print("Game Over!")
            print(f"Final Score: {info['score']}")
            
        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we need to control the frame rate of the manual player
        env.clock.tick(15) # Limit to 15 actions per second

    env.close()