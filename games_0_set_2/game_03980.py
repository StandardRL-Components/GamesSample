
# Generated: 2025-08-28T01:02:31.547348
# Source Brief: brief_03980.md
# Brief Index: 3980

        
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
        "Controls: Arrow keys to move. Stand on a green square and press Space to solve a puzzle. Avoid red traps."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a dark, procedurally generated dungeon. Solve puzzles and avoid deadly traps to find the glowing exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.CELL_SIZE = self.WIDTH // self.GRID_W
        
        self.MAX_STEPS = 1000
        self.NUM_PUZZLES = 5
        self.NUM_TRAPS = 30
        
        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (50, 50, 60)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255, 60)
        self.COLOR_TRAP = (255, 50, 50)
        self.COLOR_TRAP_FLICKER = (255, 150, 150)
        self.COLOR_PUZZLE = (50, 255, 50)
        self.COLOR_PUZZLE_SOLVED = (30, 100, 30)
        self.COLOR_EXIT = (150, 200, 255)
        self.COLOR_EXIT_GLOW = (150, 200, 255, 40)
        self.COLOR_UI_TEXT = (220, 220, 220)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # --- State Variables ---
        # These are initialized properly in _initialize_game_state()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.grid = np.zeros((self.GRID_W, self.GRID_H), dtype=np.int8)
        self.puzzles = []
        self.traps = []
        self.puzzles_solved_count = 0

        # Initialize state for validation
        self._initialize_game_state()
        self.validate_implementation()
    
    def _generate_dungeon(self):
        # 1 = wall, 0 = path
        grid = np.ones((self.GRID_W, self.GRID_H), dtype=np.int8)
        stack = []
        
        # Start carving from a random odd-numbered cell for a better maze structure
        start_x = self.np_random.integers(0, self.GRID_W // 2) * 2 + 1
        start_y = self.np_random.integers(0, self.GRID_H // 2) * 2 + 1
        
        grid[start_x, start_y] = 0
        stack.append((start_x, start_y))
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.GRID_W - 1 and 0 < ny < self.GRID_H - 1 and grid[nx, ny] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                grid[nx, ny] = 0
                grid[cx + (nx - cx) // 2, cy + (ny - cy) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return grid

    def _initialize_game_state(self):
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.grid = self._generate_dungeon()
        
        empty_cells = np.argwhere(self.grid == 0).tolist()
        self.np_random.shuffle(empty_cells)
        
        # Convert to list of tuples
        empty_cells = [tuple(cell) for cell in empty_cells]

        # Place player
        self.player_pos = empty_cells.pop()
        
        # Place exit (ensure it's far from player)
        self.exit_pos = None
        for pos in reversed(empty_cells):
            dist = abs(pos[0] - self.player_pos[0]) + abs(pos[1] - self.player_pos[1])
            if dist > (self.GRID_W + self.GRID_H) / 3:
                self.exit_pos = pos
                empty_cells.remove(pos)
                break
        if self.exit_pos is None and empty_cells: # Fallback
            self.exit_pos = empty_cells.pop()

        # Place puzzles
        self.puzzles = []
        for _ in range(self.NUM_PUZZLES):
            if not empty_cells: break
            pos = empty_cells.pop()
            self.puzzles.append([pos[0], pos[1], False]) # [x, y, solved]
        self.puzzles_solved_count = 0

        # Place traps
        self.traps = []
        for _ in range(self.NUM_TRAPS):
            if not empty_cells: break
            self.traps.append(empty_cells.pop())
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_game_state()
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.1  # Cost of taking a step
        terminated = False
        
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Handle Movement ---
        px, py = self.player_pos
        nx, ny = px, py
        
        if movement == 1: ny -= 1 # Up
        elif movement == 2: ny += 1 # Down
        elif movement == 3: nx -= 1 # Left
        elif movement == 4: nx += 1 # Right
        
        if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.grid[nx, ny] == 0:
            self.player_pos = (nx, ny)
        
        # --- Handle Interaction ---
        if space_held:
            for i, p in enumerate(self.puzzles):
                if p[0] == self.player_pos[0] and p[1] == self.player_pos[1] and not p[2]:
                    self.puzzles[i][2] = True # Mark as solved
                    reward += 10.0
                    self.puzzles_solved_count += 1
                    # sfx: puzzle solved chime
                    break
        
        # --- Check for Terminal Conditions ---
        if self.player_pos in self.traps:
            reward = -100.0
            terminated = True
            self.game_over = True
            # sfx: trap sprung sound
        
        if self.player_pos == self.exit_pos:
            reward = 100.0
            terminated = True
            self.game_over = True
            # sfx: level complete fanfare
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_game(self):
        # Draw grid and walls
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                rect = (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.grid[c, r] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

        # Draw traps
        flicker_alpha = 128 + int(127 * math.sin(self.steps * 0.8))
        for tx, ty in self.traps:
            rect = (tx * self.CELL_SIZE, ty * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.gfxdraw.box(self.screen, rect, self.COLOR_TRAP)
            if self.steps % 4 < 2: # Flicker effect
                 pygame.gfxdraw.box(self.screen, rect, (*self.COLOR_TRAP_FLICKER, flicker_alpha))

        # Draw puzzles
        pulse = 0.5 + 0.5 * math.sin(self.steps * 0.2)
        for px, py, solved in self.puzzles:
            rect = pygame.Rect(px * self.CELL_SIZE, py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            if solved:
                pygame.draw.rect(self.screen, self.COLOR_PUZZLE_SOLVED, rect)
            else:
                pygame.draw.rect(self.screen, self.COLOR_PUZZLE, rect)
                pulse_rect = rect.inflate(-self.CELL_SIZE * (0.5 * pulse), -self.CELL_SIZE * (0.5 * pulse))
                pygame.draw.rect(self.screen, (200, 255, 200), pulse_rect, border_radius=3)

        # Draw exit
        if self.exit_pos:
            ex, ey = self.exit_pos
            rect = (ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_EXIT, rect)
            glow_radius = int(self.CELL_SIZE * (1.2 + 0.2 * math.sin(self.steps * 0.1)))
            pygame.gfxdraw.filled_circle(
                self.screen,
                int((ex + 0.5) * self.CELL_SIZE),
                int((ey + 0.5) * self.CELL_SIZE),
                glow_radius,
                self.COLOR_EXIT_GLOW,
            )

        # Draw player
        px, py = self.player_pos
        player_center_x = int((px + 0.5) * self.CELL_SIZE)
        player_center_y = int((py + 0.5) * self.CELL_SIZE)
        
        pygame.gfxdraw.filled_circle(
            self.screen, player_center_x, player_center_y, int(self.CELL_SIZE * 0.8), self.COLOR_PLAYER_GLOW
        )
        
        player_rect = pygame.Rect(0, 0, self.CELL_SIZE * 0.7, self.CELL_SIZE * 0.7)
        player_rect.center = (player_center_x, player_center_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

    def _render_ui(self):
        puzzles_text = f"Puzzles: {self.puzzles_solved_count}/{len(self.puzzles)}"
        score_text = f"Score: {self.score:.1f}"
        
        puzzles_surf = self.font.render(puzzles_text, True, self.COLOR_UI_TEXT)
        score_surf = self.font.render(score_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(puzzles_surf, (10, 10))
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

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
            "puzzles_solved": self.puzzles_solved_count,
            "player_pos": self.player_pos,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block demonstrates how to run the environment and play it as a human.
    # It requires a display and is not part of the core headless environment.
    
    class HumanPlayableGameEnv(GameEnv):
        """A wrapper class to enable human-playable rendering."""
        def __init__(self, render_mode="human"):
            # Temporarily remove validation for this wrapper class
            # to avoid re-running it.
            validate_func = self.validate_implementation
            self.validate_implementation = lambda: None
            
            super().__init__(render_mode)
            self.validate_implementation = validate_func

            self.render_mode = render_mode
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                pygame.display.set_caption(self.game_description)
        
        def _get_observation(self):
            obs = super()._get_observation()
            if self.render_mode == "human":
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                self.screen.blit(surf, (0, 0))
                pygame.display.flip()
            return obs

    print("--- Human Playable Demo ---")
    print(GameEnv.user_guide)
    
    try:
        env = HumanPlayableGameEnv(render_mode="human")
        obs, info = env.reset()
        env._get_observation()
        
        running = True
        while running:
            movement = 0 # no-op
            space = 0
            
            action_taken = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    action_taken = True
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    elif event.key == pygame.K_SPACE: space = 1
                    elif event.key == pygame.K_ESCAPE: running = False
            
            if action_taken:
                action = [movement, space, 0]
                obs, reward, terminated, truncated, info = env.step(action)
                env._get_observation()
                
                if terminated:
                    print(f"Game Over! Final Score: {info['score']:.1f}. Resetting in 2 seconds.")
                    pygame.time.wait(2000)
                    obs, info = env.reset()
                    env._get_observation()
            
            env.clock.tick(30)
            
        env.close()
    except Exception as e:
        print("\nCould not run human-playable demo.")
        print("This is expected in a headless environment (e.g., a server).")
        print(f"Error: {e}")
        print("\nRunning a short headless test instead...")
        
        # Fallback to headless test
        env = GameEnv()
        obs, info = env.reset()
        terminated = False
        for _ in range(100):
            if terminated:
                break
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        print("Headless test finished.")
        print(f"Final Info: {info}")
        env.close()