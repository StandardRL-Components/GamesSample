
# Generated: 2025-08-27T21:51:59.082040
# Source Brief: brief_02934.md
# Brief Index: 2934

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Collect all yellow keys and reach the green exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze, collect keys, and reach the exit within the time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Game Constants ---
    FPS = 30
    MAX_STEPS = 1000
    TIME_LIMIT_SECONDS = 120

    # --- Maze Difficulty Scaling ---
    MAZE_MIN_WIDTH, MAZE_MIN_HEIGHT = 5, 5
    MAZE_MAX_WIDTH, MAZE_MAX_HEIGHT = 25, 25
    KEYS_MIN = 1
    KEYS_MAX = 5
    
    # --- Rewards ---
    REWARD_PER_STEP = -0.01  # Small penalty for time passing
    REWARD_KEY = 5.0
    REWARD_WIN = 50.0
    REWARD_LOSE = -50.0

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_WALL = (40, 50, 100)
    COLOR_PATH = (30, 30, 45)
    COLOR_PLAYER = (255, 50, 50)
    COLOR_KEY = (255, 220, 0)
    COLOR_EXIT_LOCKED = (30, 150, 30)
    COLOR_EXIT_UNLOCKED = (50, 255, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_PARTICLE = (255, 255, 255)

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)
        
        # State variables that persist across episodes for difficulty scaling
        self.consecutive_wins = 0
        self.current_maze_w = self.MAZE_MIN_WIDTH
        self.current_maze_h = self.MAZE_MIN_HEIGHT
        self.current_num_keys = self.KEYS_MIN
        
        # Initialize state variables
        self.maze = None
        self.player_pos = None
        self.key_locations = []
        self.exit_pos = None
        self.keys_collected = 0
        self.time_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.key_collect_effects = []

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.key_collect_effects = []
        
        # Update difficulty based on past performance
        self._update_difficulty()

        # Generate maze and place objects
        self.maze = self._generate_maze(self.current_maze_w, self.current_maze_h)
        self._place_objects()
        
        return self._get_observation(), self._get_info()

    def _update_difficulty(self):
        # Increase maze size with consecutive wins
        size_increase = self.consecutive_wins
        self.current_maze_w = min(self.MAZE_MAX_WIDTH, self.MAZE_MIN_WIDTH + size_increase)
        self.current_maze_h = min(self.MAZE_MAX_HEIGHT, self.MAZE_MIN_HEIGHT + size_increase)
        
        # Increase keys every 3 wins
        key_increase = self.consecutive_wins // 3
        self.current_num_keys = min(self.KEYS_MAX, self.KEYS_MIN + key_increase)

    def _generate_maze(self, width, height):
        # Maze grid is (width*2+1) x (height*2+1) to include walls
        grid_w, grid_h = width * 2 + 1, height * 2 + 1
        maze = np.ones((grid_h, grid_w), dtype=np.uint8) # 1 = wall
        
        # Start carving from a random cell
        start_x = self.np_random.integers(0, width) * 2 + 1
        start_y = self.np_random.integers(0, height) * 2 + 1
        
        stack = [(start_x, start_y)]
        maze[start_y, start_x] = 0 # 0 = path

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < grid_w and 0 < ny < grid_h and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice(neighbors, axis=0)
                # Carve path
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return maze

    def _place_objects(self):
        path_cells = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(path_cells)
        
        # Place player, keys, and exit, ensuring they don't overlap
        # Path cells are [y, x], so we swap for (x, y) convention
        self.player_pos = tuple(path_cells.pop()[::-1])
        self.exit_pos = tuple(path_cells.pop()[::-1])
        
        self.key_locations = []
        for _ in range(self.current_num_keys):
            if not path_cells: break
            self.key_locations.append(tuple(path_cells.pop()[::-1]))
        
        self.keys_collected = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = self.REWARD_PER_STEP
        
        # --- Game Logic ---
        self._handle_movement(movement)
        
        # Check for key collection
        key_idx = self.player_pos in self.key_locations
        if key_idx:
            self.key_locations.remove(self.player_pos)
            self.keys_collected += 1
            reward += self.REWARD_KEY
            self._add_key_collect_effect(self.player_pos)
            # sfx: key_pickup.wav

        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS

        # --- Termination Check ---
        terminated = False
        # Win condition
        if self.player_pos == self.exit_pos and self.keys_collected == self.current_num_keys:
            reward += self.REWARD_WIN
            self.game_over = True
            terminated = True
            self.win_message = "YOU WIN!"
            self.consecutive_wins += 1
            # sfx: win_jingle.wav
        # Loss conditions
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            reward += self.REWARD_LOSE
            self.game_over = True
            terminated = True
            self.win_message = "TIME UP!"
            self.consecutive_wins = 0 # Reset win streak on loss
            # sfx: lose_sound.wav
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        px, py = self.player_pos
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        # Check if new position is valid (not a wall)
        if self.maze[py][px] == 0:
            self.player_pos = (px, py)

    def _add_key_collect_effect(self, pos):
        # Create a particle burst effect
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.key_collect_effects.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 20)
            })

    def _get_observation(self):
        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        
        grid_h, grid_w = self.maze.shape
        cell_w = 640 / grid_w
        cell_h = 400 / grid_h

        # Render maze paths and walls
        for y in range(grid_h):
            for x in range(grid_w):
                rect = pygame.Rect(x * cell_w, y * cell_h, math.ceil(cell_w), math.ceil(cell_h))
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)

        # Render keys with pulsing effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 0.8 + 0.2 # Varies between 0.2 and 1.0
        key_size = min(cell_w, cell_h) * 0.6 * pulse
        for x, y in self.key_locations:
            rect = pygame.Rect(
                x * cell_w + (cell_w - key_size) / 2,
                y * cell_h + (cell_h - key_size) / 2,
                key_size, key_size
            )
            pygame.draw.rect(self.screen, self.COLOR_KEY, rect, border_radius=int(key_size/4))

        # Render exit
        exit_unlocked = self.keys_collected == self.current_num_keys
        exit_color = self.COLOR_EXIT_UNLOCKED if exit_unlocked else self.COLOR_EXIT_LOCKED
        exit_rect = pygame.Rect(self.exit_pos[0] * cell_w, self.exit_pos[1] * cell_h, cell_w, cell_h)
        pygame.draw.rect(self.screen, exit_color, exit_rect.inflate(-cell_w*0.2, -cell_h*0.2))

        # Render player with a glow
        player_size = min(cell_w, cell_h) * 0.8
        player_center_x = self.player_pos[0] * cell_w + cell_w / 2
        player_center_y = self.player_pos[1] * cell_h + cell_h / 2
        
        # Glow effect
        glow_size = int(player_size * 1.5)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER, 80), (glow_size//2, glow_size//2), glow_size//2)
        self.screen.blit(glow_surf, (player_center_x - glow_size//2, player_center_y - glow_size//2))
        
        # Player square
        player_rect = pygame.Rect(
            player_center_x - player_size / 2,
            player_center_y - player_size / 2,
            player_size, player_size
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=int(player_size/5))
        
        # Render key collection particles
        for effect in self.key_collect_effects[:]:
            effect["pos"][0] += effect["vel"][0] * (cell_w / 20)
            effect["pos"][1] += effect["vel"][1] * (cell_h / 20)
            effect["life"] -= 1
            if effect["life"] <= 0:
                self.key_collect_effects.remove(effect)
            else:
                alpha = max(0, min(255, int(255 * (effect["life"] / 20.0))))
                px = effect["pos"][0] * cell_w + cell_w / 2
                py = effect["pos"][1] * cell_h + cell_h / 2
                pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), 2, (*self.COLOR_PARTICLE, alpha))

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Timer
        time_text = f"Time: {max(0, self.time_remaining):.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (640 - time_surf.get_width() - 10, 10))
        
        # Keys
        key_text = f"Keys: {self.keys_collected} / {self.current_num_keys}"
        key_surf = self.font_ui.render(key_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(key_surf, (640 - key_surf.get_width() - 10, 35))

        # Game Over Message
        if self.game_over:
            msg_surf = self.font_msg.render(self.win_message, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(640 / 2, 400 / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "keys_collected": self.keys_collected,
            "maze_size": f"{self.current_maze_w}x{self.current_maze_h}",
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space after reset
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a simple example of how to use the environment
    
    # To prevent the Pygame window from opening, we can set the video driver
    # This is useful for running in environments without a display
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to reset.")
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()