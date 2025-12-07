import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a top-down maze puzzle game.

    The agent controls a player avatar and must navigate through three procedurally
    generated mazes to reach an exit. Each move costs a small amount of reward,
    and reaching the exit provides a large positive reward. The game is turn-based,
    with a limited number of moves per level. Optional "risk zones" offer a
    reward bonus upon entry but penalize movement within them.

    The visual design is minimalist and clean, prioritizing clarity for both
    human players and RL agents.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Reach the green exit tile to advance."
    )
    game_description = (
        "Navigate increasingly complex mazes to reach the exit. You have a limited "
        "number of moves per level. Plan your path carefully!"
    )

    # Frame advance behavior
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    UI_HEIGHT = 40
    GAME_AREA_HEIGHT = SCREEN_HEIGHT - UI_HEIGHT
    MAX_MOVES_PER_LEVEL = 100
    TOTAL_LEVELS = 3

    # Colors
    COLOR_BG = (15, 23, 42)         # Slate 900
    COLOR_WALL = (100, 116, 139)    # Slate 500
    COLOR_PATH = (30, 41, 59)       # Slate 800
    COLOR_PLAYER = (59, 130, 246)   # Blue 500
    COLOR_PLAYER_GLOW = (37, 99, 235) # Blue 600
    COLOR_EXIT = (34, 197, 94)      # Green 500
    COLOR_RISK = (239, 68, 68)      # Red 500
    COLOR_TEXT = (226, 232, 240)    # Slate 200
    COLOR_UI_BG = (30, 41, 59)      # Slate 800

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("sans-serif", 20, bold=True)
        self.font_large = pygame.font.SysFont("sans-serif", 48, bold=True)

        # Initialize state variables
        self.maze_grid = []
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.risk_zones = set()
        self.was_in_risk_zone = False
        self.maze_dims = (0, 0)
        self.cell_size = 0
        self.maze_offset = (0, 0)
        self.current_level = 0
        self.moves_remaining = 0
        self.score = 0.0
        self.steps = 0
        self.game_over = False
        self.victory = False
        self.np_random = None

    def _generate_maze(self, width, height):
        """Generates a maze using randomized Depth-First Search."""
        grid = [[{'N': True, 'S': True, 'E': True, 'W': True} for _ in range(width)] for _ in range(height)]
        stack = [(0, 0)]
        visited = set([(0, 0)])

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            if cx > 0 and (cx - 1, cy) not in visited: neighbors.append(('W', (cx - 1, cy)))
            if cx < width - 1 and (cx + 1, cy) not in visited: neighbors.append(('E', (cx + 1, cy)))
            if cy > 0 and (cx, cy - 1) not in visited: neighbors.append(('N', (cx, cy - 1)))
            if cy < height - 1 and (cx, cy + 1) not in visited: neighbors.append(('S', (cx, cy + 1)))

            if neighbors:
                # FIX: np.random.choice cannot handle a list of tuples with mixed types.
                # Instead, we pick a random index.
                choice_index = self.np_random.integers(len(neighbors))
                direction, (nx, ny) = neighbors[choice_index]

                if direction == 'N':
                    grid[cy][cx]['N'] = False
                    grid[ny][nx]['S'] = False
                elif direction == 'S':
                    grid[cy][cx]['S'] = False
                    grid[ny][nx]['N'] = False
                elif direction == 'W':
                    grid[cy][cx]['W'] = False
                    # FIX: Corrected index from grid[nx][nx] to grid[ny][nx]
                    grid[ny][nx]['E'] = False
                elif direction == 'E':
                    grid[cy][cx]['E'] = False
                    # FIX: Corrected index from grid[nx][nx] to grid[ny][nx]
                    grid[ny][nx]['W'] = False
                
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        return grid

    def _setup_level(self):
        """Initializes the maze and game state for the current level."""
        self.moves_remaining = self.MAX_MOVES_PER_LEVEL
        self.was_in_risk_zone = False

        # Increase maze size with level
        level_configs = {1: (15, 10), 2: (20, 15), 3: (25, 20)}
        self.maze_dims = level_configs[self.current_level]
        cols, rows = self.maze_dims

        self.maze_grid = self._generate_maze(cols, rows)
        
        self.player_pos = (0, 0)
        self.exit_pos = (cols - 1, rows - 1)

        # Calculate rendering geometry
        self.cell_size = min(
            (self.SCREEN_WIDTH - 20) // cols, 
            (self.GAME_AREA_HEIGHT - 20) // rows
        )
        maze_render_width = cols * self.cell_size
        maze_render_height = rows * self.cell_size
        self.maze_offset = (
            (self.SCREEN_WIDTH - maze_render_width) // 2,
            self.UI_HEIGHT + (self.GAME_AREA_HEIGHT - maze_render_height) // 2
        )

        # Place risk zones
        self.risk_zones.clear()
        num_risk_zones = self.current_level * 2
        for _ in range(num_risk_zones):
            while True:
                rx, ry = self.np_random.integers(0, cols), self.np_random.integers(0, rows)
                if (rx, ry) != self.player_pos and (rx, ry) != self.exit_pos:
                    self.risk_zones.add((rx, ry))
                    break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.victory = False
        self.current_level = 1
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = 0.0
        
        if movement != 0: # A move is attempted
            self.moves_remaining -= 1
            reward -= 0.1 # Cost for any move attempt

            px, py = self.player_pos
            nx, ny = px, py
            
            if movement == 1 and not self.maze_grid[py][px]['N']: # Up
                ny -= 1
            elif movement == 2 and not self.maze_grid[py][px]['S']: # Down
                ny += 1
            elif movement == 3 and not self.maze_grid[py][px]['W']: # Left
                nx -= 1
            elif movement == 4 and not self.maze_grid[py][px]['E']: # Right
                nx += 1

            self.player_pos = (nx, ny)

        # Handle risk zones
        is_in_risk_zone = self.player_pos in self.risk_zones
        if is_in_risk_zone:
            reward -= 2.0 # Penalty for being in a risk zone
            if not self.was_in_risk_zone:
                reward += 5.0 # Bonus for entering a risk zone
        self.was_in_risk_zone = is_in_risk_zone

        # Check for level completion
        if self.player_pos == self.exit_pos:
            reward += 10.0 # Reward for completing a level
            self.current_level += 1
            if self.current_level > self.TOTAL_LEVELS:
                self.victory = True
                self.game_over = True
                reward += 50.0 # Huge bonus for winning the game
            else:
                self._setup_level()
        
        # Check for termination by running out of moves
        if self.moves_remaining <= 0 and not self.game_over:
            self.game_over = True
        
        self.steps += 1
        self.score += reward
        terminated = self.game_over
        
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

    def _render_game(self):
        cols, rows = self.maze_dims
        ox, oy = self.maze_offset
        cs = self.cell_size

        # Draw path background, risk zones, and exit
        for y in range(rows):
            for x in range(cols):
                rect = pygame.Rect(ox + x * cs, oy + y * cs, cs, cs)
                color = self.COLOR_PATH
                if (x, y) == self.exit_pos:
                    color = self.COLOR_EXIT
                elif (x, y) in self.risk_zones:
                    color = self.COLOR_PATH # Draw path underneath
                pygame.draw.rect(self.screen, color, rect)
                
                if (x,y) in self.risk_zones:
                    # Flashing effect for risk zones
                    alpha = 128 + 127 * math.sin(self.steps * 0.2)
                    risk_surface = pygame.Surface((cs, cs), pygame.SRCALPHA)
                    risk_surface.fill((*self.COLOR_RISK, alpha))
                    self.screen.blit(risk_surface, rect.topleft)


        # Draw maze walls
        for y in range(rows):
            for x in range(cols):
                px, py = ox + x * cs, oy + y * cs
                if self.maze_grid[y][x]['N']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + cs, py), 2)
                if self.maze_grid[y][x]['S']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + cs), (px + cs, py + cs), 2)
                if self.maze_grid[y][x]['W']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + cs), 2)
                if self.maze_grid[y][x]['E']:
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + cs, py), (px + cs, py + cs), 2)

        # Draw player
        player_x = ox + int((self.player_pos[0] + 0.5) * cs)
        player_y = oy + int((self.player_pos[1] + 0.5) * cs)
        player_radius = int(cs * 0.35)
        
        # Glow effect
        glow_radius = int(player_radius * 1.5)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (player_x - glow_radius, player_y - glow_radius))

        # Player circle
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Draw UI background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, self.UI_HEIGHT-1), (self.SCREEN_WIDTH, self.UI_HEIGHT-1), 1)

        # Moves remaining
        moves_text = self.font_main.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        # Score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(centerx=self.SCREEN_WIDTH / 2, y=10)
        self.screen.blit(score_text, score_rect)

        # Level
        level_text_str = f"Level: {self.current_level}/{self.TOTAL_LEVELS}" if not self.victory else "Level: Complete!"
        level_text = self.font_main.render(level_text_str, True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(right=self.SCREEN_WIDTH - 10, y=10)
        self.screen.blit(level_text, level_rect)

        # Game Over / Victory message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            message = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_EXIT if self.victory else self.COLOR_RISK
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
            "moves_remaining": self.moves_remaining,
            "player_pos": self.player_pos,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Need to reset first to initialize everything
        self.reset()
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


# --- Example Usage ---
if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    env.validate_implementation() # Run validation on a fresh instance

    # Re-initialize for playable demo
    env.close()
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    # Un-dummy the video driver to see the window
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    print("\n" + "="*30)
    print("      MANUAL PLAY INSTRUCTIONS")
    print("="*30)
    print(env.user_guide)
    print("Press 'R' to reset the environment.")
    print("Press 'Q' or close the window to quit.")
    print("="*30 + "\n")

    while running:
        action = np.array([0, 0, 0]) # Default action is no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    print("Resetting environment...")
                    obs, info = env.reset()
                
                # Map keys to actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4

        # If a move key was pressed, step the environment
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Press 'R' to play again.")

        # Render the observation to the display window
        # Pygame uses (width, height), gym uses (height, width)
        # We need to transpose the observation back for display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate

    env.close()