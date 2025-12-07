
# Generated: 2025-08-27T23:24:34.776018
# Source Brief: brief_03456.md
# Brief Index: 3456

        
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

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to swap the selected crystal "
        "with the one in the direction you last moved."
    )

    # User-facing game description
    game_description = (
        "An isometric match-3 puzzle game. Strategically swap crystals to create lines of three or "
        "more of the same color. Clear the entire board before the timer runs out to win!"
    )

    # Frames auto-advance for smooth animations and a real-time timer
    auto_advance = True

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_COLORS = 3
    MATCH_MIN_LENGTH = 3
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # Colors
    COLOR_BG = (25, 30, 45)
    COLOR_GRID = (45, 55, 75)
    CRYSTAL_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
    ]
    CRYSTAL_SHADOW_COLORS = [
        (160, 50, 50),
        (50, 160, 50),
        (50, 75, 160),
    ]
    CRYSTAL_HIGHLIGHT_COLORS = [
        (255, 180, 180),
        (180, 255, 180),
        (180, 200, 255),
    ]
    CURSOR_COLOR = (255, 255, 0)
    UI_TEXT_COLOR = (240, 240, 240)
    UI_BG_COLOR = (40, 45, 60, 180) # RGBA

    # Rendering parameters
    TILE_WIDTH = 48
    TILE_HEIGHT = 24
    TILE_WIDTH_HALF = TILE_WIDTH // 2
    TILE_HEIGHT_HALF = TILE_HEIGHT // 2
    CRYSTAL_HEIGHT = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 64)

        self.screen_center_x = self.screen.get_width() // 2
        self.screen_center_y = self.screen.get_height() // 2 - 60

        self.reset()
        
        # This can be commented out for performance but is good for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.last_reward = 0
        
        self.timer = self.MAX_STEPS
        
        self.board = self._generate_board()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_move_direction = None # (dx, dy)
        self.prev_space_held = False
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        if not self.game_over:
            self.steps += 1
            self.timer -= 1
            
            # Handle player actions
            reward += self._handle_actions(movement, space_held)
            
            # Update game systems
            self._update_particles()
            
            # Check for win/loss conditions
            terminated, term_reward = self._check_termination()
            if terminated:
                self.game_over = True
                reward += term_reward
        
        self.score += reward
        self.last_reward = reward
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, self.game_over, False, info

    def _handle_actions(self, movement, space_held):
        reward = 0
        
        # --- Cursor Movement ---
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
        if movement in move_map:
            dx, dy = move_map[movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_HEIGHT - 1)
            self.last_move_direction = (dx, dy)

        # --- Crystal Swap (on space press) ---
        if space_held and not self.prev_space_held and self.last_move_direction is not None:
            # Sfx: swap_attempt.wav
            x, y = self.cursor_pos
            dx, dy = self.last_move_direction
            nx, ny = x + dx, y + dy

            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                # Perform swap
                self.board[y][x], self.board[ny][nx] = self.board[ny][nx], self.board[y][x]
                
                # Check for matches
                matches = self._find_all_matches()
                
                if matches:
                    # Sfx: match_success.wav
                    cleared_count, chain_reward = self._process_matches(matches)
                    reward += cleared_count + chain_reward
                else:
                    # No match, swap back
                    self.board[y][x], self.board[ny][nx] = self.board[ny][nx], self.board[y][x]
                    reward -= 0.1 # Penalty for invalid move
        
        self.prev_space_held = space_held
        return reward

    def _process_matches(self, initial_matches):
        total_cleared = 0
        chain_level = 0
        chain_reward = 0
        matches_to_process = initial_matches

        while matches_to_process:
            chain_level += 1
            if chain_level > 1:
                # Sfx: chain_reaction.wav
                chain_reward += 2 * chain_level # Bonus for chains
            
            cleared_this_turn = set()
            for match in matches_to_process:
                cleared_this_turn.update(match)
            
            # Reward for clearing more than 3 at once
            for match in matches_to_process:
                if len(match) > self.MATCH_MIN_LENGTH:
                    reward_bonus = 5
                    chain_reward += reward_bonus

            for r, c in cleared_this_turn:
                if self.board[r][c] != 0:
                    self._create_particles(r, c, self.board[r][c])
                    self.board[r][c] = 0 # 0 represents an empty space
                    total_cleared += 1
            
            self._apply_gravity()
            matches_to_process = self._find_all_matches()
        
        return total_cleared, chain_reward

    def _check_termination(self):
        # Win condition: board is empty
        if all(self.board[r][c] == 0 for r in range(self.GRID_HEIGHT) for c in range(self.GRID_WIDTH)):
            self.win_state = True
            return True, 50

        # Loss condition: timer runs out
        if self.timer <= 0:
            return True, -10

        # Loss condition: step limit reached
        if self.steps >= self.MAX_STEPS:
            return True, -10

        return False, 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_details()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "crystals_left": sum(1 for r in range(self.GRID_HEIGHT) for c in range(self.GRID_WIDTH) if self.board[r][c] != 0),
        }

    # --- Rendering Methods ---

    def _iso_to_screen(self, r, c):
        x = self.screen_center_x + (c - r) * self.TILE_WIDTH_HALF
        y = self.screen_center_y + (c + r) * self.TILE_HEIGHT_HALF
        return int(x), int(y)

    def _render_background_details(self):
        # Draw some subtle cavern details
        for _ in range(10):
            x = self.np_random.integers(0, 640)
            y = self.np_random.integers(0, 400)
            size = self.np_random.integers(50, 150)
            color_val = self.np_random.integers(20, 25)
            pygame.draw.circle(self.screen, (color_val, color_val+5, color_val+15), (x, y), size)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            start_pos = self._iso_to_screen(r, -0.5)
            end_pos = self._iso_to_screen(r, self.GRID_WIDTH - 0.5)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for c in range(self.GRID_WIDTH + 1):
            start_pos = self._iso_to_screen(-0.5, c)
            end_pos = self._iso_to_screen(self.GRID_HEIGHT - 0.5, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        
        # Draw crystals
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color_idx = self.board[r][c]
                if color_idx > 0:
                    self._draw_crystal(r, c, color_idx - 1)
        
        # Draw cursor
        self._draw_cursor()
        
        # Draw particles
        self._draw_particles()

    def _draw_crystal(self, r, c, color_idx):
        x, y = self._iso_to_screen(r, c)
        
        base_color = self.CRYSTAL_COLORS[color_idx]
        shadow_color = self.CRYSTAL_SHADOW_COLORS[color_idx]
        highlight_color = self.CRYSTAL_HIGHLIGHT_COLORS[color_idx]
        
        h = self.CRYSTAL_HEIGHT
        w_half = self.TILE_WIDTH_HALF
        h_half = self.TILE_HEIGHT_HALF
        
        # Points for the isometric cube
        p_top = (x, y - h)
        p_bottom = (x, y)
        p_left = (x - w_half, y - h_half)
        p_right = (x + w_half, y - h_half)
        p_front_left = (x - w_half, y - h_half + h)
        p_front_right = (x + w_half, y - h_half + h)
        
        # Top face
        top_face_pts = [p_top, p_right, p_bottom, p_left]
        pygame.gfxdraw.filled_polygon(self.screen, top_face_pts, highlight_color)
        pygame.gfxdraw.aapolygon(self.screen, top_face_pts, highlight_color)
        
        # Left face
        left_face_pts = [p_left, p_bottom, p_front_left, (p_left[0], p_left[1] + h)]
        pygame.gfxdraw.filled_polygon(self.screen, left_face_pts, base_color)
        pygame.gfxdraw.aapolygon(self.screen, left_face_pts, base_color)
        
        # Right face
        right_face_pts = [p_right, p_bottom, p_front_right, (p_right[0], p_right[1] + h)]
        pygame.gfxdraw.filled_polygon(self.screen, right_face_pts, shadow_color)
        pygame.gfxdraw.aapolygon(self.screen, right_face_pts, shadow_color)

    def _draw_cursor(self):
        cx, cy = self.cursor_pos
        x, y = self._iso_to_screen(cx, cy)
        
        w_half = self.TILE_WIDTH_HALF
        h_half = self.TILE_HEIGHT_HALF
        
        points = [
            (x, y - h_half),
            (x + w_half, y),
            (x, y + h_half),
            (x - w_half, y),
        ]
        
        # Pulsating alpha for the cursor glow
        alpha = int(128 + 127 * math.sin(self.steps * 0.15))
        
        # Create a temporary surface for the transparent cursor
        cursor_surface = pygame.Surface((self.TILE_WIDTH, self.TILE_HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(cursor_surface, [(p[0] - x + w_half, p[1] - y + h_half) for p in points], (*self.CURSOR_COLOR, alpha))
        pygame.gfxdraw.aapolygon(cursor_surface, [(p[0] - x + w_half, p[1] - y + h_half) for p in points], self.CURSOR_COLOR)
        
        self.screen.blit(cursor_surface, (x - w_half, y - h_half))
    
    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"Score: {int(self.score)}", True, self.UI_TEXT_COLOR)
        score_rect = score_surf.get_rect(topleft=(10, 10))
        pygame.draw.rect(self.screen, self.UI_BG_COLOR, score_rect.inflate(10, 5))
        self.screen.blit(score_surf, score_rect)
        
        # Timer
        time_left = max(0, self.timer / 30)
        timer_color = (255, 100, 100) if time_left < 10 else self.UI_TEXT_COLOR
        timer_surf = self.font_ui.render(f"Time: {time_left:.1f}", True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.screen.get_width() - 10, 10))
        pygame.draw.rect(self.screen, self.UI_BG_COLOR, timer_rect.inflate(10, 5))
        self.screen.blit(timer_surf, timer_rect)

        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if self.win_state else "TIME'S UP!"
            color = (100, 255, 100) if self.win_state else (255, 100, 100)
            
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_surf = self.font_game_over.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
            self.screen.blit(end_surf, end_rect)

    # --- Particle System ---
    
    def _create_particles(self, r, c, color_idx):
        # Sfx: crystal_shatter.wav
        x, y = self._iso_to_screen(r, c)
        color = self.CRYSTAL_COLORS[color_idx - 1]
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(3 * life_ratio)
            if radius > 0:
                alpha = int(255 * life_ratio)
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)))

    # --- Game Logic Helpers ---

    def _generate_board(self):
        while True:
            board = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            # Clear any initial matches
            while True:
                matches = self._find_all_matches(board)
                if not matches:
                    break
                for match in matches:
                    for r, c in match:
                        board[r][c] = self.np_random.integers(1, self.NUM_COLORS + 1)
            
            if self._is_board_solvable(board):
                return board.tolist()

    def _is_board_solvable(self, board):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap with right neighbor
                if c < self.GRID_WIDTH - 1:
                    board[r][c], board[r][c+1] = board[r][c+1], board[r][c]
                    if self._find_all_matches(board):
                        board[r][c], board[r][c+1] = board[r][c+1], board[r][c]
                        return True
                    board[r][c], board[r][c+1] = board[r][c+1], board[r][c] # Swap back
                # Check swap with bottom neighbor
                if r < self.GRID_HEIGHT - 1:
                    board[r][c], board[r+1][c] = board[r+1][c], board[r][c]
                    if self._find_all_matches(board):
                        board[r][c], board[r+1][c] = board[r+1][c], board[r][c]
                        return True
                    board[r][c], board[r+1][c] = board[r+1][c], board[r][c] # Swap back
        return False

    def _find_all_matches(self, board=None):
        if board is None:
            board = self.board
        
        matches = set()
        
        # Horizontal matches
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - self.MATCH_MIN_LENGTH + 1):
                if board[r][c] == 0: continue
                color = board[r][c]
                if all(board[r][c+i] == color for i in range(self.MATCH_MIN_LENGTH)):
                    match = []
                    i = 0
                    while c + i < self.GRID_WIDTH and board[r][c+i] == color:
                        match.append((r, c+i))
                        i += 1
                    if len(match) >= self.MATCH_MIN_LENGTH:
                        matches.add(tuple(sorted(match)))
        
        # Vertical matches
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - self.MATCH_MIN_LENGTH + 1):
                if board[r][c] == 0: continue
                color = board[r][c]
                if all(board[r+i][c] == color for i in range(self.MATCH_MIN_LENGTH)):
                    match = []
                    i = 0
                    while r + i < self.GRID_HEIGHT and board[r+i][c] == color:
                        match.append((r+i, c))
                        i += 1
                    if len(match) >= self.MATCH_MIN_LENGTH:
                        matches.add(tuple(sorted(match)))
        
        return list(matches)

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[r][c] != 0:
                    if r != empty_row:
                        self.board[empty_row][c] = self.board[r][c]
                        self.board[r][c] = 0
                    empty_row -= 1
    
    def validate_implementation(self):
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # This is different from the agent's MultiDiscrete action space
    # It's just for human playability.
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = False
    
    print(env.user_guide)

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_SPACE:
                    space_held = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    space_held = False

        # --- Key State Reading for Movement ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # --- Construct Action and Step Environment ---
        action = [movement, 1 if space_held else 0, 0] # Shift is not used
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already the rendered screen
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    pygame.quit()