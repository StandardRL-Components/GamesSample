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


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ↑↓←→ to navigate the maze and reach the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate procedurally generated mazes to reach the exit in the shortest time possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

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

        # Visuals
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)

        self.COLOR_BG = (15, 15, 25)  # Dark Navy
        self.COLOR_WALL = (40, 50, 120)  # Muted Blue
        self.COLOR_PLAYER = (255, 215, 0)  # Gold
        self.COLOR_PLAYER_GLOW = (255, 215, 0, 50)
        self.COLOR_EXIT = (0, 255, 127)  # Spring Green
        self.COLOR_BREADCRUMB = (220, 50, 80)  # Muted Red
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

        # Game configuration
        self.max_steps = 500
        self.initial_maze_dim = 10
        self.max_maze_dim = 30
        self.current_maze_dim = self.initial_maze_dim

        # Game state variables (initialized in reset)
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.start_time = 0
        self.breadcrumbs = []

        # Initialize state variables - this is called to set up the initial maze
        # A seed will be provided by the environment wrapper if needed
        # self.reset() # This is now called by the wrapper, not in __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.start_time = pygame.time.get_ticks()

        # Generate new maze
        self.maze = self._generate_maze(self.current_maze_dim, self.current_maze_dim)
        self.player_pos = [1, 1]
        self.exit_pos = [self.maze.shape[0] - 2, self.maze.shape[1] - 2]

        self.breadcrumbs = [self.player_pos.copy()]

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right

        reward = -0.1  # Cost for taking a step

        if not self.game_over:
            prev_pos = self.player_pos.copy()

            target_pos = self.player_pos.copy()
            if movement == 1:  # Up
                target_pos[0] -= 1
            elif movement == 2:  # Down
                target_pos[0] += 1
            elif movement == 3:  # Left
                target_pos[1] -= 1
            elif movement == 4:  # Right
                target_pos[1] += 1

            # Check for valid move
            if self.maze[target_pos[0], target_pos[1]] == 0:
                self.player_pos = target_pos
                # sound_effect: "player_move.wav"

                # Update breadcrumbs
                if self.player_pos != prev_pos:
                    self.breadcrumbs.append(self.player_pos.copy())
                    if len(self.breadcrumbs) > 20:  # Keep trail length manageable
                        self.breadcrumbs.pop(0)

        self.steps += 1

        # Check for win/loss conditions
        terminated = self._check_termination()

        if self.game_won:
            reward += 10.0
            self.score += 10
            # Increase difficulty for the next round
            self.current_maze_dim = min(self.max_maze_dim, self.current_maze_dim + 1)
        elif self.game_over and not self.game_won:  # Lost by timeout
            reward -= 10.0
            self.score -= 10

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _check_termination(self):
        if self.game_over:  # Already terminated
            return True

        if self.player_pos == self.exit_pos:
            self.game_won = True
            self.game_over = True
            # sound_effect: "win_level.wav"
            return True

        if self.steps >= self.max_steps:
            self.game_over = True
            # sound_effect: "lose_timeout.wav"
            return True

        return False

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
            "maze_dim": self.current_maze_dim,
            "player_pos": self.player_pos,
        }

    def _generate_maze(self, width, height):
        # Ensure odd dimensions for a proper maze with walls
        width = width if width % 2 != 0 else width + 1
        height = height if height % 2 != 0 else height + 1

        # Initialize maze with walls (1)
        maze = np.ones((height, width), dtype=np.uint8)

        # Start carving from (1, 1)
        start_r, start_c = (1, 1)
        maze[start_r, start_c] = 0  # Path
        stack = [(start_r, start_c)]

        while stack:
            r, c = stack[-1]

            # Get unvisited neighbors (2 cells away)
            neighbors = []
            for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nr, nc = r + dr, c + dc
                if 0 < nr < height - 1 and 0 < nc < width - 1 and maze[nr, nc] == 1:
                    neighbors.append((nr, nc))

            if neighbors:
                # Choose a random neighbor
                random_index = self.np_random.integers(len(neighbors))
                nr, nc = neighbors[random_index]

                # Carve path to neighbor
                maze[nr, nc] = 0
                maze[r + (nr - r) // 2, c + (nc - c) // 2] = 0

                stack.append((nr, nc))
            else:
                stack.pop()

        # Ensure exit is open
        maze[height - 2, width - 2] = 0
        return maze

    def _render_game(self):
        if self.maze is None:
            return

        maze_h, maze_w = self.maze.shape

        # Calculate cell size to fit the maze on screen
        cell_w = self.screen.get_width() / maze_w
        cell_h = self.screen.get_height() / maze_h

        # Render breadcrumbs
        for i, pos in enumerate(self.breadcrumbs):
            r, c = pos
            center_x = int((c + 0.5) * cell_w)
            center_y = int((r + 0.5) * cell_h)
            radius = int(min(cell_w, cell_h) * 0.2)

            # Fade effect
            alpha = int(150 * (i / len(self.breadcrumbs)))
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                temp_surf, (*self.COLOR_BREADCRUMB, alpha), (radius, radius), radius
            )
            self.screen.blit(temp_surf, (center_x - radius, center_y - radius))

        # Render maze walls, exit, and player
        for r in range(maze_h):
            for c in range(maze_w):
                screen_x, screen_y = int(c * cell_w), int(r * cell_h)
                rect = pygame.Rect(
                    screen_x, screen_y, math.ceil(cell_w), math.ceil(cell_h)
                )

                if self.maze[r, c] == 1:  # Wall
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                elif [r, c] == self.exit_pos:  # Exit
                    pygame.draw.rect(self.screen, self.COLOR_EXIT, rect)

        # Render player
        player_r, player_c = self.player_pos
        player_x = int((player_c + 0.5) * cell_w)
        player_y = int((player_r + 0.5) * cell_h)
        player_radius = int(min(cell_w, cell_h) * 0.35)

        # Glow effect
        glow_radius = int(player_radius * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surf, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius
        )
        self.screen.blit(glow_surf, (player_x - glow_radius, player_y - glow_radius))

        # Player circle
        pygame.draw.circle(
            self.screen, self.COLOR_PLAYER, (player_x, player_y), player_radius
        )

    def _render_ui(self):
        # --- UI Text Rendering Function ---
        def draw_text(text, font, color, pos, shadow_color=None, align="topleft"):
            text_surf = font.render(text, True, color)
            text_rect = text_surf.get_rect()
            setattr(text_rect, align, pos)

            if shadow_color:
                shadow_surf = font.render(text, True, shadow_color)
                shadow_rect = shadow_surf.get_rect()
                setattr(shadow_rect, align, (pos[0] + 2, pos[1] + 2))
                self.screen.blit(shadow_surf, shadow_rect)

            self.screen.blit(text_surf, text_rect)

        # --- Display Stats ---
        steps_text = f"Steps: {self.steps}/{self.max_steps}"
        draw_text(
            steps_text, self.font_small, self.COLOR_TEXT, (10, 10), self.COLOR_TEXT_SHADOW
        )

        elapsed_time = (pygame.time.get_ticks() - self.start_time) // 1000
        time_text = f"Time: {elapsed_time}s"
        draw_text(
            time_text,
            self.font_small,
            self.COLOR_TEXT,
            (self.screen.get_width() - 10, 10),
            self.COLOR_TEXT_SHADOW,
            align="topright",
        )

        # --- Game Over / Win Message ---
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.game_won:
                message = "LEVEL COMPLETE!"
            else:
                message = "OUT OF STEPS!"

            draw_text(
                message,
                self.font_large,
                self.COLOR_TEXT,
                (self.screen.get_width() // 2, self.screen.get_height() // 2),
                self.COLOR_TEXT_SHADOW,
                align="center",
            )

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == "__main__":
    # This block will now require a display driver
    # Unset the dummy driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")

    # --- Manual Play Loop ---
    obs, info = env.reset(seed=42)
    done = False

    # Pygame window for human interaction
    render_screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Maze Runner")
    clock = pygame.time.Clock()

    running = True
    while running:
        action = [0, 0, 0]  # Default action: no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                if event.key == pygame.K_SPACE:
                     action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                     action[2] = 1

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if done:
                    # Render the final state before resetting
                    render_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                    render_screen.blit(render_surf, (0, 0))
                    pygame.display.flip()
                    pygame.time.wait(2000)
                    
                    obs, info = env.reset()
                    done = False

        # Render the observation to the screen
        render_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(render_surf, (0, 0))
        pygame.display.flip()

        clock.tick(30)  # Limit frame rate for human play

    env.close()