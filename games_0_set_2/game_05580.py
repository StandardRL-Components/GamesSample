
# Generated: 2025-08-28T05:27:35.229224
# Source Brief: brief_05580.md
# Brief Index: 5580

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your avatar. "
        "Collect 50 crystals before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down puzzle game. Navigate a shifting crystal maze to collect 50 crystals "
        "within 200 moves. The maze reconfigures every 5 moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20  # SCREEN_WIDTH / GRID_WIDTH

    COLOR_BG = (10, 5, 15)
    COLOR_WALL = (40, 30, 60)
    COLOR_WALL_ACCENT = (80, 60, 120)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    CRYSTAL_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Lime
        (255, 128, 0),  # Orange
    ]

    MAX_MOVES = 200
    CRYSTALS_TO_WIN = 50
    MAZE_SHIFT_INTERVAL = 5
    NUM_WALLS = 60
    NUM_CRYSTALS_ON_SCREEN = 20
    
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
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 32)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.wall_grid = None
        self.crystals = None
        self.crystal_locations_set = None
        self.moves_remaining = 0
        self.score = 0
        self.moves_since_shift = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        
        # This will be properly initialized in reset()
        self.np_random = None

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.moves_since_shift = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        
        self._generate_maze()
        
        return self._get_observation(), self._get_info()

    def _generate_maze(self):
        # Initialize an empty grid
        self.wall_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int8)
        self.crystals = []

        # Get all possible grid cells
        possible_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        
        # Player position is not available for walls/crystals
        player_coord_tuple = tuple(self.player_pos)
        if player_coord_tuple in possible_coords:
            possible_coords.remove(player_coord_tuple)
            
        # Shuffle coordinates for random placement
        self.np_random.shuffle(possible_coords)

        # Place walls
        num_walls_to_place = min(self.NUM_WALLS, len(possible_coords))
        for _ in range(num_walls_to_place):
            x, y = possible_coords.pop()
            self.wall_grid[x, y] = 1

        # Place crystals
        num_crystals_to_place = min(self.NUM_CRYSTALS_ON_SCREEN, len(possible_coords))
        for _ in range(num_crystals_to_place):
            self.crystals.append(np.array(possible_coords.pop()))
        
        self.crystal_locations_set = {tuple(c) for c in self.crystals}

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # Only movement actions consume a turn
        if movement > 0:
            self.moves_remaining -= 1
            self.moves_since_shift += 1
            reward = -0.1 # Cost for making a move

            # --- Player Movement ---
            delta = np.array([0, 0])
            if movement == 1: delta[1] = -1 # Up
            elif movement == 2: delta[1] = 1 # Down
            elif movement == 3: delta[0] = -1 # Left
            elif movement == 4: delta[0] = 1 # Right
            
            target_pos = self.player_pos + delta
            tx, ty = target_pos

            # --- Collision Detection ---
            is_valid_move = True
            if not (0 <= tx < self.GRID_WIDTH and 0 <= ty < self.GRID_HEIGHT):
                is_valid_move = False # Wall boundary
            elif self.wall_grid[tx, ty] == 1:
                is_valid_move = False # Wall collision
            
            if is_valid_move:
                self.player_pos = target_pos
                # SFX: player_move.wav

        # --- Crystal Collection ---
        player_pos_tuple = tuple(self.player_pos)
        if player_pos_tuple in self.crystal_locations_set:
            self.score += 1
            reward += 1.0 # Reward for collecting a crystal
            self.crystal_locations_set.remove(player_pos_tuple)
            self.crystals = [c for c in self.crystals if tuple(c) != player_pos_tuple]
            # SFX: crystal_collect.wav

        # --- Maze Shift ---
        if self.moves_since_shift >= self.MAZE_SHIFT_INTERVAL and movement > 0:
            self._generate_maze()
            self.moves_since_shift = 0
            # SFX: maze_shift.wav
        
        self.steps += 1

        # --- Termination Check ---
        terminated = False
        if self.score >= self.CRYSTALS_TO_WIN:
            self.win = True
            terminated = True
            reward += 100.0 # Victory bonus
        elif self.moves_remaining <= 0:
            terminated = True
            reward += -10.0 # Defeat penalty
        
        if terminated:
            self.game_over = True

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
            "crystals_to_win": self.CRYSTALS_TO_WIN,
        }

    def _render_game(self):
        # Draw walls
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.wall_grid[x, y] == 1:
                    rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                    pygame.draw.rect(self.screen, self.COLOR_WALL_ACCENT, rect, 1)

        # Draw crystals
        for i, crystal_pos in enumerate(self.crystals):
            cx, cy = crystal_pos
            color = self.CRYSTAL_COLORS[i % len(self.CRYSTAL_COLORS)]
            
            outer_rect = pygame.Rect(
                int(cx * self.CELL_SIZE + self.CELL_SIZE * 0.15),
                int(cy * self.CELL_SIZE + self.CELL_SIZE * 0.15),
                int(self.CELL_SIZE * 0.7),
                int(self.CELL_SIZE * 0.7)
            )
            inner_rect = pygame.Rect(
                int(cx * self.CELL_SIZE + self.CELL_SIZE * 0.25),
                int(cy * self.CELL_SIZE + self.CELL_SIZE * 0.25),
                int(self.CELL_SIZE * 0.5),
                int(self.CELL_SIZE * 0.5)
            )

            pygame.draw.rect(self.screen, color, outer_rect, border_radius=3)
            brighter_color = tuple(min(255, c + 60) for c in color)
            pygame.draw.rect(self.screen, brighter_color, inner_rect, border_radius=2)

        # Draw player
        px, py = self.player_pos
        center_x = int(px * self.CELL_SIZE + self.CELL_SIZE / 2)
        center_y = int(py * self.CELL_SIZE + self.CELL_SIZE / 2)
        radius = int(self.CELL_SIZE * 0.4)

        # Glow effect
        glow_radius = int(radius * 1.5)
        glow_color = (150, 150, 255, 100) # Semi-transparent blueish white
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (center_x - glow_radius, center_y - glow_radius))
        
        # Player circle
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Moves remaining
        moves_text = f"Moves: {self.moves_remaining}"
        text_surface = self.font_medium.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Crystal count
        crystal_text = f"Crystals: {self.score} / {self.CRYSTALS_TO_WIN}"
        text_surface = self.font_medium.render(crystal_text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(text_surface, text_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            text_surface = self.font_large.render(message, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Reset to get a valid state before testing
        self.reset(seed=123)
        
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=456)
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

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Pygame setup for human play
    pygame.display.set_caption("Crystal Maze")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    
    print(env.user_guide)
    print(f"Game: {env.game_description}")

    while running:
        action = np.array([0, 0, 0]) # Default action is no-op
        
        made_move = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset(seed=random.randint(0, 10000))
                    terminated = False
                    made_move = True

                # Map keys to actions for human play
                if not terminated:
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        action[0] = 4
                    
                    if action[0] != 0:
                        made_move = True

        if made_move and not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_remaining']}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Use a clock to limit FPS for human play
        env.clock.tick(30)
        
    env.close()