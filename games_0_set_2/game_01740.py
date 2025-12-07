import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import Counter
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press Space to 'click' the selected shape, "
        "changing its color and the color of its neighbors."
    )

    game_description = (
        "A strategic color-matching puzzle. Your goal is to make all shapes on the 3x4 grid the "
        "same color within a limited number of moves. Each click alters a cross-shaped pattern of shapes."
    )

    auto_advance = False

    # --- Constants ---
    # Game mechanics
    GRID_ROWS, GRID_COLS = 3, 4
    NUM_COLORS = 3
    MAX_MOVES = 15
    MAX_STEPS = 1000

    # Screen dimensions
    WIDTH, HEIGHT = 640, 400

    # Visuals
    COLOR_BG = pygame.Color("#2c3e50")
    COLOR_GRID = pygame.Color("#34495e")
    SHAPE_COLORS = [
        pygame.Color("#e74c3c"),  # Red
        pygame.Color("#2ecc71"),  # Green
        pygame.Color("#3498db"),  # Blue
    ]
    COLOR_SELECTOR = pygame.Color("#f1c40f")
    COLOR_TEXT = pygame.Color("#ecf0f1")
    COLOR_WIN = pygame.Color("#2ecc71")
    COLOR_LOSS = pygame.Color("#e74c3c")

    CLICK_ANIMATION_FRAMES = 15  # Duration of the pulse effect

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)

        # Grid layout calculation
        self.padding = 40
        self.grid_width = self.WIDTH - 2 * self.padding
        self.grid_height = self.HEIGHT - 2 * self.padding
        self.cell_width = self.grid_width / self.GRID_COLS
        self.cell_height = self.grid_height / self.GRID_ROWS
        self.shape_radius = int(min(self.cell_width, self.cell_height) * 0.35)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.selector_pos = (0, 0)
        self.shape_colors = [[]]
        self.last_clicked_pos = None
        self.click_animation_timer = 0
        self.milestone_25 = False
        self.milestone_50 = False
        self.milestone_75 = False
        self.win_state = False

        # self.reset() is called here to ensure the environment is ready after init.
        # However, Gymnasium's standard practice is to call reset() externally after creation.
        # For compatibility and to ensure all attributes are set, we'll initialize them here.
        # A full reset will still be needed before starting an episode.
        self._initialize_state()


    def _initialize_state(self):
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_remaining = self.MAX_MOVES

        # Reset milestones
        self.milestone_25 = False
        self.milestone_50 = False
        self.milestone_75 = False

        # Center selector
        self.selector_pos = (self.GRID_ROWS // 2, self.GRID_COLS // 2)

        # Animation state
        self.last_clicked_pos = None
        self.click_animation_timer = 0

        # Create a dummy board state. A proper one will be generated in reset().
        self.shape_colors = [[0 for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()

        # Generate a non-winning board
        while True:
            self.shape_colors = self.np_random.integers(
                0, self.NUM_COLORS, size=(self.GRID_ROWS, self.GRID_COLS)
            ).tolist()
            if len(set(c for row in self.shape_colors for c in row)) > 1:
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, _ = action
        reward = 0.0

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1

        # 1. Handle Selector Movement
        if movement != 0:
            row, col = self.selector_pos
            if movement == 1:  # Up
                row = (row - 1 + self.GRID_ROWS) % self.GRID_ROWS
            elif movement == 2:  # Down
                row = (row + 1) % self.GRID_ROWS
            elif movement == 3:  # Left
                col = (col - 1 + self.GRID_COLS) % self.GRID_COLS
            elif movement == 4:  # Right
                col = (col + 1) % self.GRID_COLS
            self.selector_pos = (row, col)

        # 2. Handle Click Action
        if space_pressed:
            if self.moves_remaining > 0:
                self.moves_remaining -= 1
                reward -= 0.1  # Cost for taking an action

                row, col = self.selector_pos
                self._apply_color_change(row, col)

                # Trigger animation
                self.last_clicked_pos = (row, col)
                self.click_animation_timer = self.CLICK_ANIMATION_FRAMES

                # Calculate rewards based on new board state
                reward += self._calculate_board_state_reward()
            else:
                # Penalize trying to act with no moves left
                reward -= 1.0

        # 3. Check for Termination
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win_state:
                reward += 100
            else:
                reward -= 100

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _apply_color_change(self, r, c):
        """Applies the cross-pattern color change."""
        affected_cells = [(r, c)]
        if r > 0:
            affected_cells.append((r - 1, c))
        if r < self.GRID_ROWS - 1:
            affected_cells.append((r + 1, c))
        if c > 0:
            affected_cells.append((r, c - 1))
        if c < self.GRID_COLS - 1:
            affected_cells.append((r, c + 1))

        for row, col in affected_cells:
            self.shape_colors[row][col] = (
                self.shape_colors[row][col] + 1
            ) % self.NUM_COLORS

    def _calculate_board_state_reward(self):
        """Calculates rewards for majority color and milestones."""
        reward = 0
        all_colors = [color for row in self.shape_colors for color in row]
        if not all_colors:
            return 0

        counts = Counter(all_colors)
        majority_count = counts.most_common(1)[0][1]

        # Continuous reward for matching majority
        reward += majority_count

        total_shapes = self.GRID_ROWS * self.GRID_COLS
        match_ratio = majority_count / total_shapes

        # Milestone rewards
        if match_ratio >= 0.75 and not self.milestone_75:
            reward += 20
            self.milestone_75 = True
        if match_ratio >= 0.50 and not self.milestone_50:
            reward += 10
            self.milestone_50 = True
        if match_ratio >= 0.25 and not self.milestone_25:
            reward += 5
            self.milestone_25 = True

        return reward

    def _check_termination(self):
        """Checks for win, loss, or max steps."""
        all_colors = [color for row in self.shape_colors for color in row]

        # Win condition: all shapes are the same color
        if len(set(all_colors)) == 1:
            self.win_state = True
            return True

        # Loss condition: out of moves
        if self.moves_remaining <= 0:
            return True

        # Max steps reached
        if self.steps >= self.MAX_STEPS:
            return True

        return False

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
            "selector_pos": self.selector_pos,
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(1, self.GRID_COLS):
            x = self.padding + i * self.cell_width
            pygame.draw.line(
                self.screen, self.COLOR_GRID, (x, self.padding), (x, self.HEIGHT - self.padding), 2
            )
        for i in range(1, self.GRID_ROWS):
            y = self.padding + i * self.cell_height
            pygame.draw.line(
                self.screen, self.COLOR_GRID, (self.padding, y), (self.WIDTH - self.padding, y), 2
            )

        # Update and draw click animation
        if self.click_animation_timer > 0:
            self.click_animation_timer -= 1

        # Draw shapes
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                center_x = self.padding + c * self.cell_width + self.cell_width / 2
                center_y = self.padding + r * self.cell_height + self.cell_height / 2

                color_idx = self.shape_colors[r][c]
                color = self.SHAPE_COLORS[color_idx]

                radius = self.shape_radius

                # Pulse effect
                if self.last_clicked_pos:
                    dist = abs(r - self.last_clicked_pos[0]) + abs(
                        c - self.last_clicked_pos[1]
                    )
                    if dist <= 1:  # Center or adjacent
                        progress = self.click_animation_timer / self.CLICK_ANIMATION_FRAMES
                        pulse_factor = math.sin(
                            progress * math.pi
                        )  # Goes from 0 to 1 to 0
                        radius += int(self.shape_radius * 0.3 * pulse_factor)

                pygame.gfxdraw.filled_circle(
                    self.screen, int(center_x), int(center_y), int(radius), color
                )
                pygame.gfxdraw.aacircle(
                    self.screen, int(center_x), int(center_y), int(radius), color
                )

        # Draw selector
        sel_r, sel_c = self.selector_pos
        rect_x = self.padding + sel_c * self.cell_width
        rect_y = self.padding + sel_r * self.cell_height
        selector_rect = pygame.Rect(rect_x, rect_y, self.cell_width, self.cell_height)

        # Breathing glow effect
        glow_alpha = 128 + 127 * math.sin(self.steps * 0.1)
        self._draw_glowing_rect(selector_rect, self.COLOR_SELECTOR, 10, int(glow_alpha))

    def _draw_glowing_rect(self, rect, color, glow_size, alpha):
        """Draws a glowing rectangle using multiple blended surfaces."""
        for i in range(glow_size, 0, -2):
            s = pygame.Surface((rect.width + i * 2, rect.height + i * 2), pygame.SRCALPHA)
            current_alpha = int(alpha * ((glow_size - i) / glow_size))
            # FIX: Construct a valid RGBA tuple from the color's components and the calculated alpha.
            # The original code (*color, current_alpha) created a 5-element tuple, causing a ValueError.
            pygame.draw.rect(
                s, (color.r, color.g, color.b, current_alpha), s.get_rect(), border_radius=10
            )
            self.screen.blit(s, (rect.x - i, rect.y - i))
        pygame.draw.rect(self.screen, color, rect, 3, border_radius=8)

    def _render_ui(self):
        # Render moves remaining
        moves_text = self.font_medium.render(
            f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT
        )
        text_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(moves_text, text_rect)

        # Render game over screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))

            if self.win_state:
                message = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                message = "GAME OVER"
                color = self.COLOR_LOSS

            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))

            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually
    # Ensure the dummy video driver is NOT set for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False

    # Create a window to display the game
    pygame.display.set_caption("Chroma Shift")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))

    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op

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
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_r:  # Reset key
                    obs, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if not env.game_over and np.any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"Action: {action}, Reward: {reward:.2f}, Moves Left: {info['moves_remaining']}, Done: {terminated}"
            )
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}")

        # Get the latest observation from the environment
        obs = env._get_observation()
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    env.close()