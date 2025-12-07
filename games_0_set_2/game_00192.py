import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric puzzle game where the player collects crystals on a grid within a limited number of moves.

    **Gameplay:**
    - The player navigates a character on a 12x12 isometric grid.
    - The goal is to collect 20 crystals before running out of 100 moves.
    - Each movement action (up, down, left, right) consumes one move.
    - The game is turn-based; the state only updates when an action is taken.

    **Visuals:**
    - The game is rendered from a fixed isometric perspective.
    - The player and crystals are represented as 3D-style cubes with shading.
    - Anti-aliasing is used for smooth, high-quality graphics.
    - A UI displays the number of crystals collected and moves remaining.

    **Rewards:**
    - +1 for each crystal collected.
    - +100 for winning the game (collecting 20 crystals).
    - -10 for losing the game (running out of moves).
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys (Up, Down, Left, Right) to move your character one tile at a time."
    )

    # User-facing description of the game
    game_description = (
        "A strategic puzzle game. Collect all 20 crystals in under 100 moves by planning your path on the isometric grid."
    )

    # Frames wait for user input
    auto_advance = False
    
    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 12
    GRID_HEIGHT = 12
    TILE_WIDTH_HALF = 24
    TILE_HEIGHT_HALF = 12
    
    MOVE_LIMIT = 100
    CRYSTAL_TARGET = 20
    NUM_CRYSTALS = 20

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (50, 60, 70)
    COLOR_PLAYER = (255, 255, 255)
    CRYSTAL_COLORS = [
        (255, 80, 80),  # Red
        (80, 255, 80),  # Green
        (80, 150, 255), # Blue
        (255, 255, 80)  # Yellow
    ]
    COLOR_TEXT = (220, 220, 220)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)

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
        
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 72)

        # Isometric projection origin
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - self.GRID_HEIGHT * self.TILE_HEIGHT_HALF
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.crystals = None
        self.crystal_colors = None
        self.moves_remaining = 0
        self.crystals_collected = 0
        self.game_over = False
        self.win = False
        self.np_random = None

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_remaining = self.MOVE_LIMIT
        self.crystals_collected = 0
        self.game_over = False
        self.win = False
        
        # Initialize RNG
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Place player in the center
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        
        # Generate crystal locations
        self.crystals = set()
        all_possible_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        all_possible_coords.remove(self.player_pos)
        
        chosen_indices = self.np_random.choice(len(all_possible_coords), self.NUM_CRYSTALS, replace=False)
        for i in chosen_indices:
            self.crystals.add(all_possible_coords[i])
        
        self.crystal_colors = {pos: self.np_random.integers(0, len(self.CRYSTAL_COLORS)) for pos in self.crystals}

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1  # Unused
        # shift_held = action[2] == 1 # Unused
        
        reward = 0
        moved = False

        if movement > 0: # 0 is no-op
            px, py = self.player_pos
            if movement == 1: # Up (grid-wise)
                py -= 1
            elif movement == 2: # Down
                py += 1
            elif movement == 3: # Left
                px -= 1
            elif movement == 4: # Right
                px += 1

            # Check boundaries
            if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
                self.player_pos = (px, py)
            
            self.moves_remaining -= 1
            moved = True

        # Check for crystal collection
        if self.player_pos in self.crystals:
            self.crystals.remove(self.player_pos)
            self.crystals_collected += 1
            reward += 1
            # sfx: crystal collect sound

        # Check for termination
        terminated = False
        if self.crystals_collected >= self.CRYSTAL_TARGET:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100 # Win bonus
            # sfx: win fanfare
        elif self.moves_remaining <= 0:
            terminated = True
            self.game_over = True
            self.win = False
            reward -= 10 # Lose penalty
            # sfx: lose sound

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _iso_to_screen(self, grid_x, grid_y):
        """Converts grid coordinates to screen coordinates for isometric projection."""
        screen_x = self.origin_x + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.origin_y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, grid_pos, base_color):
        """Draws a 3D-looking isometric cube."""
        x, y = self._iso_to_screen(grid_pos[0], grid_pos[1])
        w, h = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        
        # Shaded colors for 3D effect
        top_color = base_color
        left_color = tuple(max(0, c - 40) for c in base_color)
        right_color = tuple(max(0, c - 70) for c in base_color)
        
        # Define cube vertices
        top_point = (x, y - h)
        bottom_point = (x, y + h)
        left_point = (x - w, y)
        right_point = (x + w, y)
        
        # Top face
        top_face_pts = [top_point, (x, y), left_point, (x - w, y - h)]
        pygame.gfxdraw.filled_polygon(surface, top_face_pts, top_color)
        pygame.gfxdraw.aapolygon(surface, top_face_pts, top_color)
        
        # Left face
        left_face_pts = [left_point, (x - w, y - h), (x - w, y + h), bottom_point]
        pygame.gfxdraw.filled_polygon(surface, left_face_pts, left_color)
        pygame.gfxdraw.aapolygon(surface, left_face_pts, left_color)
        
        # Right face
        right_face_pts = [right_point, (x + w, y - h), (x + w, y + h), bottom_point]
        pygame.gfxdraw.filled_polygon(surface, right_face_pts, right_color)
        pygame.gfxdraw.aapolygon(surface, right_face_pts, right_color)


    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._iso_to_screen(x, y)
                points = [
                    (sx, sy - self.TILE_HEIGHT_HALF),
                    (sx + self.TILE_WIDTH_HALF, sy),
                    (sx, sy + self.TILE_HEIGHT_HALF),
                    (sx - self.TILE_WIDTH_HALF, sy)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Create a render queue and sort by depth for correct isometric drawing
        render_queue = []
        # Add crystals
        if self.crystals is not None:
            for pos in self.crystals:
                color_index = self.crystal_colors[pos]
                render_queue.append(('crystal', pos, self.CRYSTAL_COLORS[color_index]))
        # Add player
        if self.player_pos is not None:
            render_queue.append(('player', self.player_pos, self.COLOR_PLAYER))

        # Sort by y+x for proper Z-ordering
        render_queue.sort(key=lambda item: (item[1][0] + item[1][1]))

        # Draw items from queue
        for item_type, pos, color in render_queue:
            self._draw_iso_cube(self.screen, pos, color)

    def _render_ui(self):
        # Draw UI text
        crystal_text = self.font_ui.render(f"Crystals: {self.crystals_collected}/{self.CRYSTAL_TARGET}", True, self.COLOR_TEXT)
        self.screen.blit(crystal_text, (10, 10))
        
        moves_text = self.font_ui.render(f"Moves Left: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))

            if self.win:
                msg_text = self.font_game_over.render("YOU WIN!", True, self.COLOR_WIN)
            else:
                msg_text = self.font_game_over.render("GAME OVER", True, self.COLOR_LOSE)
            
            text_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.crystals_collected,
            "steps": self.MOVE_LIMIT - self.moves_remaining,
            "moves_remaining": self.moves_remaining,
            "crystals_collected": self.crystals_collected,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset first to initialize the state
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)

        # Test observation space (now that state is initialized)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
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
    # This block allows you to play the game manually
    # The environment is created in headless mode, so we need a separate display
    pygame.display.init()
    pygame.font.init()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Isometric Crystal Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                if not terminated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        if not terminated and action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
        
        # Draw the observation to the screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
    env.close()
    pygame.quit()