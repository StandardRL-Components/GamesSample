
# Generated: 2025-08-27T21:14:15.458497
# Source Brief: brief_02720.md
# Brief Index: 2720

        
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
        "Controls: ↑↓←→ to move. Collect all gems and reach the green exit before the timer runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze, collecting gems and racing against the clock to reach the exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_WALL = (40, 50, 70)
        self.COLOR_PATH = (15, 25, 40) # Same as BG
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0, 50)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_GEM = (0, 180, 255)
        self.COLOR_GEM_SPARKLE = (200, 240, 255)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # Game state variables
        self.successful_episodes_count = 0
        self.maze_width = 5
        self.maze_height = 5
        self.maze = None
        self.player_pos = [0, 0]
        self.exit_pos = [0, 0]
        self.gem_positions = set()
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.win = False
        self.last_dist_to_exit = 0

        # Initialize state
        self.reset()

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Difficulty scaling
        base_size = 5
        size_increase = self.successful_episodes_count // 5
        self.maze_width = base_size + size_increase
        self.maze_height = base_size + size_increase
        # Ensure odd dimensions for maze generator
        if self.maze_width % 2 == 0: self.maze_width += 1
        if self.maze_height % 2 == 0: self.maze_height += 1
        
        # Generate maze and place items
        self.maze = self._generate_maze(self.maze_width, self.maze_height)
        self.player_pos = [1, 1]
        self.exit_pos = [self.maze_width - 2, self.maze_height - 2]
        self._place_gems(num_gems=5)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.time_remaining = 30
        self.game_over = False
        self.win = False
        
        self.last_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self.steps += 1
        self.time_remaining -= 1
        reward = 0

        # --- Player Movement ---
        prev_pos = list(self.player_pos)
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if movement != 0:
            new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
            if 0 <= new_y < self.maze_height and 0 <= new_x < self.maze_width and self.maze[new_y, new_x] == 0:
                self.player_pos = [new_x, new_y]
                # sfx: player move
        
        # --- Reward Calculation ---
        # Distance-based reward
        new_dist = self._manhattan_distance(self.player_pos, self.exit_pos)
        if new_dist < self.last_dist_to_exit:
            reward += 0.1
        elif new_dist > self.last_dist_to_exit:
            reward -= 0.2
        self.last_dist_to_exit = new_dist

        # Gem collection
        if tuple(self.player_pos) in self.gem_positions:
            self.gem_positions.remove(tuple(self.player_pos))
            self.score += 5
            reward += 5
            # sfx: gem collect

        # --- Termination Check ---
        terminated = False
        if self.time_remaining <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 50
            # sfx: lose
        
        if self.player_pos == self.exit_pos:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 50
            self.successful_episodes_count += 1
            # sfx: win

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
            "gems_remaining": len(self.gem_positions),
        }

    def _generate_maze(self, width, height):
        maze = np.ones((height, width), dtype=np.uint8)
        
        def carve(x, y):
            maze[y, x] = 0
            directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
            self.np_random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 < nx < width and 0 < ny < height and maze[ny, nx] == 1:
                    maze[ny - dy // 2, nx - dx // 2] = 0
                    carve(nx, ny)

        carve(1, 1)

        # Add some loops to make it non-perfect
        num_loops = (width * height) // 20
        for _ in range(num_loops):
            while True:
                rx = self.np_random.integers(1, width - 1)
                ry = self.np_random.integers(1, height - 1)
                if maze[ry, rx] == 1:
                    # Check if it's an internal wall
                    is_horizontal = maze[ry-1, rx] == 0 and maze[ry+1, rx] == 0
                    is_vertical = maze[ry, rx-1] == 0 and maze[ry, rx+1] == 0
                    if is_horizontal or is_vertical:
                        maze[ry, rx] = 0
                        break
        return maze

    def _place_gems(self, num_gems):
        self.gem_positions.clear()
        possible_cells = []
        for y in range(1, self.maze_height - 1):
            for x in range(1, self.maze_width - 1):
                if self.maze[y, x] == 0 and [x, y] != self.player_pos and [x, y] != self.exit_pos:
                    possible_cells.append((x, y))
        
        if len(possible_cells) < num_gems:
            num_gems = len(possible_cells)
            
        selected_indices = self.np_random.choice(len(possible_cells), num_gems, replace=False)
        for i in selected_indices:
            self.gem_positions.add(possible_cells[i])

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _render_game(self):
        # Calculate tile size and offsets to center the maze
        tile_w = self.SCREEN_WIDTH / self.maze_width
        tile_h = self.SCREEN_HEIGHT / self.maze_height
        
        # Draw maze
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                rect = pygame.Rect(int(x * tile_w), int(y * tile_h), math.ceil(tile_w), math.ceil(tile_h))
                if self.maze[y, x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw exit
        exit_rect = pygame.Rect(int(self.exit_pos[0] * tile_w), int(self.exit_pos[1] * tile_h), math.ceil(tile_w), math.ceil(tile_h))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw gems
        gem_radius = int(min(tile_w, tile_h) * 0.3)
        sparkle_phase = (self.steps % 30) / 30.0
        sparkle_size = int(gem_radius * 0.5 * (1 + math.sin(sparkle_phase * 2 * math.pi)))
        for x, y in self.gem_positions:
            center_x = int((x + 0.5) * tile_w)
            center_y = int((y + 0.5) * tile_h)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, gem_radius, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, gem_radius, self.COLOR_GEM)
            if sparkle_size > 0:
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, sparkle_size, self.COLOR_GEM_SPARKLE)

        # Draw player
        player_radius = int(min(tile_w, tile_h) * 0.35)
        player_center_x = int((self.player_pos[0] + 0.5) * tile_w)
        player_center_y = int((self.player_pos[1] + 0.5) * tile_h)
        
        # Glow effect
        glow_radius = int(player_radius * (1.5 + 0.2 * math.sin(self.steps * 0.3)))
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Player circle
        pygame.gfxdraw.filled_circle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center_x, player_center_y, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # UI Background panels
        score_panel = pygame.Surface((150, 40), pygame.SRCALPHA)
        score_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(score_panel, (10, 10))

        timer_panel = pygame.Surface((150, 40), pygame.SRCALPHA)
        timer_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(timer_panel, (self.SCREEN_WIDTH - 160, 10))

        # Score Text
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 18))

        # Timer Text
        timer_text = self.font_ui.render(f"Time: {self.time_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - 150, 18))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "SUCCESS!" if self.win else "TIME UP!"
            color = self.COLOR_EXIT if self.win else (255, 50, 50)
            
            end_text = self.font_game_over.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To render, we need a display
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    # Game loop for human play
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Only step if a key was pressed, as auto_advance is False
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # A small delay to make it playable for humans
        pygame.time.wait(100)

    print(f"Game Over. Final Score: {info['score']}")
    pygame.quit()