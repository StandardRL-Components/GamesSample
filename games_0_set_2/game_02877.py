import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


# Set headless mode for pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys (↑↓←→) to move your character one tile at a time."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated isometric cavern, collecting crystals while avoiding deadly traps to amass a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_FLOOR = (45, 45, 65)
    COLOR_FLOOR_GRID = (55, 55, 80)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_OUTLINE = (200, 255, 255)
    COLOR_TRAP = (200, 30, 30)
    COLOR_TRAP_OUTLINE = (255, 100, 100)
    CRYSTAL_COLORS = [
        (100, 255, 100), (100, 100, 255), (255, 100, 255), (255, 255, 100)
    ]
    COLOR_UI_TEXT = (220, 220, 220)

    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Grid and Tile Dimensions
    GRID_WIDTH = 18
    GRID_HEIGHT = 18
    TILE_WIDTH_HALF = 18
    TILE_HEIGHT_HALF = 9
    
    # Game Parameters
    TOTAL_CRYSTALS = 20
    NUM_TRAPS = 25
    MAX_STEPS = 1000

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Center the grid
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF) // 2
        
        # State variables are initialized here to allow rendering before the first reset() call.
        # This is necessary for the validation check.
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.crystals = []
        self.traps = []
        self.score = 0.0
        self.steps = 0
        self.game_over = False
        self.crystals_collected = 0
        self.last_action_was_move = False

        # This will be set by the super().reset() call
        self.np_random = None

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.crystals_collected = 0
        self.last_action_was_move = False
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean (unused)
        shift_held = action[2] == 1  # Boolean (unused)
        
        self.steps += 1
        reward = 0.0
        self.last_action_was_move = False

        if not self.game_over and movement != 0:
            self.last_action_was_move = True
            
            # --- Continuous Reward Calculation (before move) ---
            dist_crystal_before = self._find_closest_distance(self.player_pos, self.crystals)
            dist_trap_before = self._find_closest_distance(self.player_pos, self.traps)

            # --- Move Logic ---
            px, py = self.player_pos
            next_pos = self.player_pos
            if movement == 1:  # Up
                next_pos = (px - 1, py - 1)
            elif movement == 2:  # Down
                next_pos = (px + 1, py + 1)
            elif movement == 3:  # Left
                next_pos = (px - 1, py + 1)
            elif movement == 4:  # Right
                next_pos = (px + 1, py - 1)
            
            # Check boundaries
            nx, ny = next_pos
            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                self.player_pos = next_pos
                
                # --- Continuous Reward Calculation (after move) ---
                dist_crystal_after = self._find_closest_distance(self.player_pos, self.crystals)
                dist_trap_after = self._find_closest_distance(self.player_pos, self.traps)

                if dist_crystal_after < dist_crystal_before:
                    reward += 0.1
                
                if dist_trap_after < dist_trap_before:
                    reward += -0.2

        # --- Event-based Reward and State Change ---
        if self.player_pos in self.crystals:
            # Sound: Crystal collect
            self.crystals.remove(self.player_pos)
            self.crystals_collected += 1
            self.score += 1.0
            reward += 1.0
            if self.crystals_collected >= self.TOTAL_CRYSTALS:
                self.game_over = True
                self.score += 100.0
                reward += 100.0
        
        if self.player_pos in self.traps:
            # Sound: Player falls into trap
            self.game_over = True
            self.score -= 10.0
            reward -= 10.0

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        # Clamp reward to specified range for non-terminal
        if not terminated:
            reward = np.clip(reward, -10.0, 10.0)
        
        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,  # truncated always False
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
            "crystals_collected": self.crystals_collected,
        }

    def _generate_level(self):
        # Place player in the center
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)

        # Generate all possible tile positions
        all_positions = set()
        for r in range(self.GRID_WIDTH):
            for c in range(self.GRID_HEIGHT):
                all_positions.add((r, c))

        # Ensure player start is not used for objects
        if self.player_pos in all_positions:
            all_positions.remove(self.player_pos)
        
        # Use BFS to find all reachable tiles (in this open grid, it's all of them)
        # This structure is here to support future complex wall generation
        q = deque([self.player_pos])
        reachable = {self.player_pos}
        while q:
            r, c = q.popleft()
            # Isometric neighbors
            for dr, dc in [(-1, -1), (1, 1), (-1, 1), (1, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_WIDTH and 0 <= nc < self.GRID_HEIGHT and (nr, nc) not in reachable:
                    reachable.add((nr, nc))
                    q.append((nr, nc))
        
        # Place crystals on reachable tiles
        self.crystals = []
        possible_crystal_locs = list(reachable - {self.player_pos})
        if len(possible_crystal_locs) >= self.TOTAL_CRYSTALS:
            crystal_indices = self.np_random.choice(len(possible_crystal_locs), self.TOTAL_CRYSTALS, replace=False)
            self.crystals = [possible_crystal_locs[i] for i in crystal_indices]

        # Place traps, avoiding player start and crystal locations
        possible_trap_locs = list(all_positions - set(self.crystals))
        if len(possible_trap_locs) >= self.NUM_TRAPS:
            trap_indices = self.np_random.choice(len(possible_trap_locs), self.NUM_TRAPS, replace=False)
            self.traps = [possible_trap_locs[i] for i in trap_indices]

    def _grid_to_screen(self, grid_x, grid_y):
        screen_x = self.origin_x + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.origin_y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Render floor tiles
        for r in range(self.GRID_WIDTH):
            for c in range(self.GRID_HEIGHT):
                sx, sy = self._grid_to_screen(r, c)
                points = [
                    (sx, sy - self.TILE_HEIGHT_HALF),
                    (sx + self.TILE_WIDTH_HALF, sy),
                    (sx, sy + self.TILE_HEIGHT_HALF),
                    (sx - self.TILE_WIDTH_HALF, sy),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_FLOOR)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_FLOOR_GRID)

        # Render traps
        pulse = abs(math.sin(self.steps * 0.1)) * 0.2 + 0.8
        for r, c in self.traps:
            sx, sy = self._grid_to_screen(r, c)
            size = int(self.TILE_WIDTH_HALF * 0.6 * pulse)
            points = [
                (sx, sy - size // 2),
                (sx + size, sy),
                (sx, sy + size // 2),
                (sx - size, sy),
            ]
            color = tuple(int(val * pulse) for val in self.COLOR_TRAP)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TRAP_OUTLINE)
        
        # Render crystals
        sparkle = math.sin(self.steps * 0.2)
        for i, (r, c) in enumerate(self.crystals):
            sx, sy = self._grid_to_screen(r, c)
            size = self.TILE_WIDTH_HALF * 0.5
            color = self.CRYSTAL_COLORS[i % len(self.CRYSTAL_COLORS)]
            
            # Animate with a sparkle
            final_color = [min(255, int(val + (val * sparkle * 0.3))) for val in color]

            points = []
            for j in range(6):
                angle = 2 * math.pi / 6 * j + (self.steps * 0.05)
                px = sx + size * math.cos(angle)
                py = sy + size * math.sin(angle) * 0.5 # squash for isometric view
                points.append((int(px), int(py)))
            pygame.gfxdraw.filled_polygon(self.screen, points, final_color)
            pygame.gfxdraw.aapolygon(self.screen, points, (255, 255, 255))

        # Render player
        if self.player_pos and not (self.player_pos in self.traps):
            px, py = self._grid_to_screen(*self.player_pos)
            
            # Bobbing animation
            bob = math.sin(self.steps * 0.15) * 2
            py += int(bob)

            # Movement "pop" effect
            size_mod = 1.3 if self.last_action_was_move else 1.0

            player_size = int(self.TILE_WIDTH_HALF * 0.7 * size_mod)
            points = [
                (px, py - player_size // 2),
                (px + player_size, py),
                (px, py + player_size // 2),
                (px - player_size, py),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_OUTLINE)

    def _render_ui(self):
        # Score and Crystal Count
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))
        
        crystal_text = self.font_ui.render(f"Crystals: {self.crystals_collected} / {self.TOTAL_CRYSTALS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (15, 45))

        # Game Over Message
        if self.game_over:
            if self.crystals_collected >= self.TOTAL_CRYSTALS:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = self.COLOR_TRAP

            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _find_closest_distance(self, pos, object_list):
        if not object_list:
            return float('inf')
        
        px, py = pos
        min_dist = float('inf')
        for ox, oy in object_list:
            # Manhattan distance on the grid
            dist = abs(px - ox) + abs(py - oy)
            if dist < min_dist:
                min_dist = dist
        return min_dist

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
        
        # print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    # It will create a window and render the game
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS"
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Isometric Crystal Caverns")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    action = np.array([0, 0, 0]) # No-op
    
    # Game loop
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    done = False
                    action = np.array([0, 0, 0])
                    continue

                # Default to no-op
                movement = 0
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                
                if movement != 0:
                    action = np.array([movement, 0, 0])
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")
                    done = terminated or truncated

        # Rendering
        frame = env._get_observation()
        # The observation is (H, W, C), but pygame wants (W, H, C) for surfarray
        # So we need to transpose it back for display
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        if done:
            # Wait for 'R' to reset
            pass

    env.close()