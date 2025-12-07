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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character on the isometric grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate an isometric cavern, strategically collecting crystals to escape before your light source runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 15
    GRID_HEIGHT = 15
    TOTAL_CRYSTALS = 25
    MAX_MOVES = 50

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    CRYSTAL_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Lime
        (255, 128, 0),  # Orange
    ]

    # Isometric projection constants
    TILE_WIDTH_HALF = 18
    TILE_HEIGHT_HALF = 9
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 80

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
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_small = pygame.font.SysFont("monospace", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 22)

        # Game state variables (initialized in reset)
        self.player_pos = None
        self.crystals = None
        self.moves_left = None
        self.crystals_collected = None
        self.score = None
        self.steps = None
        self.game_over_message = ""

        # Initialize state variables
        # self.reset() called by validate_implementation

        # Validate implementation after initialization
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.crystals_collected = 0
        self.game_over_message = ""

        # Place player and crystals
        self._place_entities()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are unused in this game

        reward = 0
        terminated = False

        if self.moves_left > 0 and not self.game_over_message:
            # Get distance to nearest crystal before moving
            dist_before = self._get_dist_to_nearest_crystal()

            # Apply movement
            moved = self._move_player(movement)

            # Decrease moves only if a move was attempted (not a no-op)
            if movement != 0:
                self.moves_left -= 1
                self.steps += 1

            # Get distance to nearest crystal after moving
            dist_after = self._get_dist_to_nearest_crystal()

            # --- Reward Calculation ---

            # 1. Crystal collection reward
            if self._check_crystal_collection():
                reward += 10.0
                # sfx: crystal collect sound

            # 2. Distance-based reward (only if no crystal was collected and player moved)
            elif moved and dist_before is not None and dist_after is not None:
                if dist_after < dist_before:
                    reward += 0.1  # Closer to a crystal
                else:
                    reward += -0.2 # Farther from a crystal

        # --- Termination Check ---
        if self.crystals_collected == self.TOTAL_CRYSTALS:
            if not self.game_over_message: # Apply reward only on first frame of termination
                reward += 100.0  # Win bonus
            terminated = True
            self.game_over_message = "SUCCESS! ALL CRYSTALS COLLECTED."
        elif self.moves_left <= 0:
            if not self.game_over_message: # Apply reward only on first frame of termination
                reward += -50.0  # Loss penalty
            terminated = True
            self.game_over_message = "FAILURE! OUT OF MOVES."

        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _place_entities(self):
        """Places the player and crystals on the grid, avoiding overlaps."""
        occupied_positions = set()

        # Place player in the center
        self.player_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        occupied_positions.add(tuple(self.player_pos))

        # Place crystals
        self.crystals = []
        while len(self.crystals) < self.TOTAL_CRYSTALS:
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            if pos not in occupied_positions:
                self.crystals.append({"pos": np.array(pos), "color": random.choice(self.CRYSTAL_COLORS)})
                occupied_positions.add(pos)

    def _move_player(self, movement):
        """Updates player position based on action, checking boundaries."""
        delta = np.array([0, 0])
        if movement == 1:  # Up (iso up-left)
            delta = np.array([0, -1])
        elif movement == 2:  # Down (iso down-right)
            delta = np.array([0, 1])
        elif movement == 3:  # Left (iso down-left)
            delta = np.array([-1, 0])
        elif movement == 4:  # Right (iso up-right)
            delta = np.array([1, 0])

        if np.any(delta != 0):
            new_pos = self.player_pos + delta
            if (0 <= new_pos[0] < self.GRID_WIDTH) and (0 <= new_pos[1] < self.GRID_HEIGHT):
                self.player_pos = new_pos
                # sfx: player move step
                return True
        return False

    def _check_crystal_collection(self):
        """Checks if player is on a crystal tile and handles collection."""
        for i, crystal in enumerate(self.crystals):
            if np.array_equal(self.player_pos, crystal["pos"]):
                self.crystals.pop(i)
                self.crystals_collected += 1
                return True
        return False

    def _get_dist_to_nearest_crystal(self):
        """Calculates Manhattan distance to the nearest crystal."""
        if not self.crystals:
            return 0

        min_dist = float('inf')
        for crystal in self.crystals:
            dist = np.sum(np.abs(self.player_pos - crystal["pos"])) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "crystals_collected": self.crystals_collected,
        }

    def _grid_to_screen(self, grid_pos):
        """Converts grid coordinates to screen pixel coordinates."""
        screen_x = self.ORIGIN_X + (grid_pos[0] - grid_pos[1]) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (grid_pos[0] + grid_pos[1]) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _render_game(self):
        """Renders the main game elements: grid, crystals, player."""
        # Render grid lines
        for i in range(self.GRID_WIDTH + 1):
            start = self._grid_to_screen(np.array([i, 0]))
            end = self._grid_to_screen(np.array([i, self.GRID_HEIGHT]))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for i in range(self.GRID_HEIGHT + 1):
            start = self._grid_to_screen(np.array([0, i]))
            end = self._grid_to_screen(np.array([self.GRID_WIDTH, i]))
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Render crystals
        for crystal in self.crystals:
            self._render_crystal(crystal["pos"], crystal["color"])

        # Render player
        self._render_player()

    def _render_crystal(self, pos, color):
        """Renders a single glowing crystal."""
        screen_pos = self._grid_to_screen(pos)

        # Glow effect
        glow_radius = self.TILE_WIDTH_HALF * 1.2
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (screen_pos[0] - glow_radius, screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Crystal shape (isometric cube)
        p = screen_pos
        w, h = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF

        top_face = [(p[0], p[1]-h), (p[0]+w, p[1]), (p[0], p[1]+h), (p[0]-w, p[1])]
        left_face = [(p[0]-w, p[1]), (p[0], p[1]+h), (p[0], p[1]+h*2), (p[0]-w, p[1]+h)]
        right_face = [(p[0]+w, p[1]), (p[0], p[1]+h), (p[0], p[1]+h*2), (p[0]+w, p[1]+h)]

        # Darken side colors
        darken_factor = 0.7
        left_color = tuple(int(c * darken_factor) for c in color)
        right_color = tuple(int(c * darken_factor * darken_factor) for c in color)

        pygame.gfxdraw.filled_polygon(self.screen, left_face, left_color)
        pygame.gfxdraw.aapolygon(self.screen, left_face, left_color)
        pygame.gfxdraw.filled_polygon(self.screen, right_face, right_color)
        pygame.gfxdraw.aapolygon(self.screen, right_face, right_color)
        pygame.gfxdraw.filled_polygon(self.screen, top_face, color)
        pygame.gfxdraw.aapolygon(self.screen, top_face, color)

    def _render_player(self):
        """Renders the player character."""
        screen_pos = self._grid_to_screen(self.player_pos)

        # Glow effect
        glow_radius = self.TILE_WIDTH_HALF * 1.5
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 40), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (screen_pos[0] - glow_radius, screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player shape (diamond)
        w, h = self.TILE_WIDTH_HALF * 0.8, self.TILE_HEIGHT_HALF * 1.6
        points = [
            (screen_pos[0], screen_pos[1] - h),
            (screen_pos[0] + w, screen_pos[1]),
            (screen_pos[0], screen_pos[1] + h),
            (screen_pos[0] - w, screen_pos[1]),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        """Renders the UI elements like score and moves left."""
        # Crystals collected
        crystal_text = f"Crystals: {self.crystals_collected} / {self.TOTAL_CRYSTALS}"
        text_surf = self.font_small.render(crystal_text, True, self.COLOR_PLAYER)
        self.screen.blit(text_surf, (10, 10))

        # Moves left
        moves_text = f"Moves Left: {self.moves_left}"
        text_surf = self.font_small.render(moves_text, True, self.COLOR_PLAYER)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

        # Game over message
        if self.game_over_message:
            color = self.CRYSTAL_COLORS[3] if "SUCCESS" in self.game_over_message else self.CRYSTAL_COLORS[4]

            # Shadow
            shadow_surf = self.font_large.render(self.game_over_message, True, (0,0,0))
            shadow_rect = shadow_surf.get_rect(center=(self.SCREEN_WIDTH / 2 + 2, self.SCREEN_HEIGHT / 2 + 2))
            self.screen.blit(shadow_surf, shadow_rect)

            # Main text
            text_surf = self.font_large.render(self.game_over_message, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Beginning implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset to get an observation
        obs, info = self.reset()

        # Test observation space
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert self.observation_space.contains(obs)

        # Test info from reset
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert self.observation_space.contains(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()

    # Re-initialize pygame for display
    pygame.quit()
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    pygame.init()

    screen_width, screen_height = 640, 400
    pygame.display.set_caption("Isometric Crystal Cavern")
    screen = pygame.display.set_mode((screen_width, screen_height))

    running = True
    terminated = False

    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("Press 'R' to reset.")
    print("="*30 + "\n")

    while running:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                move_action = 0
                if event.key == pygame.K_UP:
                    move_action = 1
                elif event.key == pygame.K_DOWN:
                    move_action = 2
                elif event.key == pygame.K_LEFT:
                    move_action = 3
                elif event.key == pygame.K_RIGHT:
                    move_action = 4

                if event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
                    terminated = False
                    print("--- Game Reset ---")

                # If a move key was pressed, step the environment
                if move_action != 0 and not terminated:
                    action[0] = move_action
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Render the observation to the display
        # The environment's rendering is headless, so we get the frame from obs
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()