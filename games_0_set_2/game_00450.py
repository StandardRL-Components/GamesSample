
# Generated: 2025-08-27T13:41:42.628492
# Source Brief: brief_00450.md
# Brief Index: 450

        
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
        "Controls: Use arrows to move the cursor. Press space to select a tile, "
        "then move and press space on an adjacent tile to swap. Press shift to deselect."
    )

    game_description = (
        "Swap adjacent tiles to create matches of three or more. Plan your moves "
        "to create chain reactions and reach the target score before you run out of moves!"
    )

    auto_advance = False
    
    # --- Game Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_TILE_TYPES = 5
    WIN_SCORE = 1000
    MAX_STEPS = 1000
    INITIAL_MOVES = 30
    
    # --- Colors ---
    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (50, 50, 60)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_SELECTED = (255, 255, 255)
    
    TILE_COLORS = [
        (255, 80, 80),    # Red
        (80, 255, 80),    # Green
        (80, 120, 255),   # Blue
        (255, 255, 80),   # Yellow
        (200, 80, 255),   # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Font setup
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)

        # Rendering dimensions
        self.grid_pixel_size = 320
        self.tile_size = self.grid_pixel_size // self.GRID_WIDTH
        self.grid_offset_x = (self.screen_width - self.grid_pixel_size) // 2
        self.grid_offset_y = (self.screen_height - self.grid_pixel_size) // 2
        
        # State variables are initialized in reset()
        self.board = None
        self.score = 0
        self.moves_left = 0
        self.cursor_pos = [0, 0]
        self.selected_tile_pos = None
        self.last_space_state = False
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.shuffle_indicator = 0 # Countdown for showing "SHUFFLE" text
        
        # Initialize state
        self.reset()

        # Self-check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tile_pos = None
        self.last_space_state = False
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.shuffle_indicator = 0
        
        self._generate_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_state
        self.last_space_state = space_held

        reward = 0
        
        # --- Action Handling ---
        # 1. Cursor Movement
        if movement == 1: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
        elif movement == 2: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
        elif movement == 3: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
        elif movement == 4: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH

        # 2. Deselection
        if shift_held and self.selected_tile_pos:
            self.selected_tile_pos = None
            # sfx: deselect sound

        # 3. Selection / Swap
        if space_press:
            if self.selected_tile_pos is None:
                self.selected_tile_pos = list(self.cursor_pos)
                # sfx: select sound
            else:
                # Attempt swap if cursor is on an adjacent tile
                dx = abs(self.cursor_pos[0] - self.selected_tile_pos[0])
                dy = abs(self.cursor_pos[1] - self.selected_tile_pos[1])
                if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                    reward = self._handle_swap(self.selected_tile_pos, self.cursor_pos)
                    self.moves_left -= 1
                else:
                    # sfx: invalid action sound
                    pass # Not adjacent, do nothing
                self.selected_tile_pos = None

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            else:
                reward += -50 # Loss penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_swap(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Perform swap
        self.board[x1, y1], self.board[x2, y2] = self.board[x2, y2], self.board[x1, y1]
        
        matches = self._find_all_matches()
        
        # If no match, swap back and penalize
        if not matches:
            self.board[x1, y1], self.board[x2, y2] = self.board[x2, y2], self.board[x1, y1]
            # sfx: invalid swap sound
            return -0.1

        # --- Process Matches (Chain Reaction) ---
        # sfx: match sound
        total_reward = 0
        chain_level = 0
        while matches:
            # Calculate score and reward
            num_cleared = len(matches)
            self.score += num_cleared
            total_reward += num_cleared
            if chain_level > 0:
                total_reward += 5 # Chain reaction bonus
                # sfx: chain reaction sound
            
            # Create particles
            for x, y in matches:
                self._spawn_particles(x, y, self.board[x, y])

            # Clear matched tiles and let new ones fall
            self._clear_and_refill(matches)
            
            chain_level += 1
            matches = self._find_all_matches()

        # Anti-softlock: check for moves and reshuffle if none
        if not self._has_possible_moves():
            self._reshuffle_board()
            self.shuffle_indicator = 60 # Show "SHUFFLE" text for 60 frames/steps
        
        return total_reward

    def _check_termination(self):
        return (
            self.score >= self.WIN_SCORE or
            self.moves_left <= 0 or
            self.steps >= self.MAX_STEPS
        )

    def _generate_board(self):
        self.board = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        while self._find_all_matches() or not self._has_possible_moves():
            # Reroll until no initial matches and at least one move is possible
            matches = self._find_all_matches()
            if matches:
                for x, y in matches:
                    self.board[x, y] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)
            else: # No matches, but no possible moves
                self.board = self.np_random.integers(1, self.NUM_TILE_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))

    def _reshuffle_board(self):
        # sfx: shuffle sound
        flat_board = self.board.flatten()
        self.np_random.shuffle(flat_board)
        self.board = flat_board.reshape((self.GRID_WIDTH, self.GRID_HEIGHT))
        
        # Ensure new board is valid
        while self._find_all_matches() or not self._has_possible_moves():
            self.np_random.shuffle(flat_board)
            self.board = flat_board.reshape((self.GRID_WIDTH, self.GRID_HEIGHT))
            
    def _find_all_matches(self):
        matches = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Horizontal check
                if x < self.GRID_WIDTH - 2 and self.board[x, y] == self.board[x+1, y] == self.board[x+2, y]:
                    matches.update([(x, y), (x+1, y), (x+2, y)])
                # Vertical check
                if y < self.GRID_HEIGHT - 2 and self.board[x, y] == self.board[x, y+1] == self.board[x, y+2]:
                    matches.update([(x, y), (x, y+1), (x, y+2)])
        return matches

    def _has_possible_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Try swapping right
                if x < self.GRID_WIDTH - 1:
                    self.board[x, y], self.board[x+1, y] = self.board[x+1, y], self.board[x, y]
                    if self._find_all_matches():
                        self.board[x, y], self.board[x+1, y] = self.board[x+1, y], self.board[x, y]
                        return True
                    self.board[x, y], self.board[x+1, y] = self.board[x+1, y], self.board[x, y]
                # Try swapping down
                if y < self.GRID_HEIGHT - 1:
                    self.board[x, y], self.board[x, y+1] = self.board[x, y+1], self.board[x, y]
                    if self._find_all_matches():
                        self.board[x, y], self.board[x, y+1] = self.board[x, y+1], self.board[x, y]
                        return True
                    self.board[x, y], self.board[x, y+1] = self.board[x, y+1], self.board[x, y]
        return False

    def _clear_and_refill(self, matches):
        # Set matched tiles to 0 (empty)
        for x, y in matches:
            self.board[x, y] = 0
        
        # Gravity: drop tiles down
        for x in range(self.GRID_WIDTH):
            empty_slots = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.board[x, y] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.board[x, y + empty_slots] = self.board[x, y]
                    self.board[x, y] = 0
        
        # Fill top empty slots with new tiles
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.board[x, y] == 0:
                    self.board[x, y] = self.np_random.integers(1, self.NUM_TILE_TYPES + 1)

    def _spawn_particles(self, grid_x, grid_y, tile_type):
        px = self.grid_offset_x + grid_x * self.tile_size + self.tile_size // 2
        py = self.grid_offset_y + grid_y * self.tile_size + self.tile_size // 2
        color = self.TILE_COLORS[tile_type - 1]
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': [px, py], 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_lines()
        self._render_tiles()
        self._render_cursor_and_selection()
        self._render_particles()

    def _render_grid_lines(self):
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_offset_x + i * self.tile_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_pixel_size))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_offset_y + i * self.tile_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_pixel_size, y))

    def _render_tiles(self):
        shape_padding = self.tile_size // 5
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                tile_type = self.board[x, y]
                if tile_type > 0:
                    color = self.TILE_COLORS[tile_type - 1]
                    rect = pygame.Rect(
                        self.grid_offset_x + x * self.tile_size,
                        self.grid_offset_y + y * self.tile_size,
                        self.tile_size, self.tile_size
                    )
                    
                    # Draw a specific shape for each tile type for accessibility
                    shape_rect = rect.inflate(-shape_padding*2, -shape_padding*2)
                    center = shape_rect.center
                    radius = shape_rect.width // 2
                    
                    if tile_type == 1: # Circle
                        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, color)
                        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color)
                    elif tile_type == 2: # Square
                        pygame.draw.rect(self.screen, color, shape_rect)
                    elif tile_type == 3: # Triangle
                        points = [(center[0], shape_rect.top), (shape_rect.right, shape_rect.bottom), (shape_rect.left, shape_rect.bottom)]
                        pygame.gfxdraw.aapolygon(self.screen, points, color)
                        pygame.gfxdraw.filled_polygon(self.screen, points, color)
                    elif tile_type == 4: # Diamond
                        points = [(center[0], shape_rect.top), (shape_rect.right, center[1]), (center[0], shape_rect.bottom), (shape_rect.left, center[1])]
                        pygame.gfxdraw.aapolygon(self.screen, points, color)
                        pygame.gfxdraw.filled_polygon(self.screen, points, color)
                    elif tile_type == 5: # Cross
                        pygame.draw.rect(self.screen, color, shape_rect.inflate(-radius, 0))
                        pygame.draw.rect(self.screen, color, shape_rect.inflate(0, -radius))

    def _render_cursor_and_selection(self):
        # Draw selected tile highlight
        if self.selected_tile_pos:
            x, y = self.selected_tile_pos
            rect = pygame.Rect(
                self.grid_offset_x + x * self.tile_size,
                self.grid_offset_y + y * self.tile_size,
                self.tile_size, self.tile_size
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTED, rect, 3)

        # Draw cursor
        cx, cy = self.cursor_pos
        rect = pygame.Rect(
            self.grid_offset_x + cx * self.tile_size,
            self.grid_offset_y + cy * self.tile_size,
            self.tile_size, self.tile_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)

    def _render_particles(self):
        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                new_particles.append(p)
                alpha = max(0, min(255, int(255 * (p['lifespan'] / 40))))
                color = p['color'] + (alpha,)
                
                temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (2, 2), 2)
                self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))
        self.particles = new_particles

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 20))

        # Moves display
        moves_text = self.font_large.render(f"MOVES: {self.moves_left}", True, (255, 255, 255))
        self.screen.blit(moves_text, (self.screen_width - moves_text.get_width() - 20, 20))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)
        
        # Shuffle indicator
        if self.shuffle_indicator > 0:
            self.shuffle_indicator -= 1
            alpha = min(255, self.shuffle_indicator * 5)
            shuffle_text = self.font_large.render("SHUFFLE!", True, (255, 200, 0))
            shuffle_text.set_alpha(alpha)
            text_rect = shuffle_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(shuffle_text, text_rect)

    def close(self):
        pygame.quit()

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Setup ---
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Create a window to display the game
    pygame.display.set_caption("Match-3 Gym Environment")
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    running = True
    while running:
        # --- Action Calculation ---
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        # --- Step the Environment ---
        # Since auto_advance is False, we only step on an action
        # For a better human play experience, we'll step continuously
        # but the logic inside step() only processes changes on key presses.
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        if terminated:
            print("Game Over!")
            # Display final frame for a moment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(3000)
            
            # Reset for a new game
            obs, info = env.reset()

        # --- Render to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate for human play
        env.clock.tick(30)
        
    env.close()