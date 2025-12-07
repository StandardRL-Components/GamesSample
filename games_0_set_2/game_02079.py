import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your green square around the grid. "
        "Space and Shift do nothing in this game."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a shifting grid to collect 100 blue gems while avoiding red traps. "
        "You lose if you hit 5 traps or run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_SIZE = 10
        self.GRID_AREA_SIZE = 400
        self.CELL_SIZE = self.GRID_AREA_SIZE // self.GRID_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_AREA_SIZE) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_AREA_SIZE) // 2

        # Game Parameters
        self.MAX_STEPS = 1000
        self.START_LIVES = 5
        self.WIN_SCORE = 100
        self.NUM_GEMS = 5
        self.NUM_TRAPS = 10

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID_BG = (25, 25, 40)
        self.COLOR_GRID_LINES = (40, 40, 60)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_GEM = (50, 150, 255)
        self.COLOR_GEM_SPARKLE = (200, 220, 255)
        self.COLOR_TRAP = (255, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # State variables are initialized in reset().
        # We must call reset() once to set up the initial state.
        self.reset()

        # Validate implementation after setup
        try:
            self.validate_implementation()
        except AssertionError as e:
            print(f"Validation failed: {e}")

    def _get_empty_cell(self):
        """Finds a random empty cell on the grid."""
        occupied_cells = set(map(tuple, self.gems + self.traps))
        if self.player_pos:
            occupied_cells.add(tuple(self.player_pos))

        while True:
            pos = [
                self.np_random.integers(0, self.GRID_SIZE),
                self.np_random.integers(0, self.GRID_SIZE),
            ]
            if tuple(pos) not in occupied_cells:
                return pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = self.START_LIVES
        self.game_over = False

        self.player_pos = [0, 0]
        self.gems = []
        self.traps = []

        # Generate initial gem and trap positions
        # Temporarily add player to occupied cells for initial placement
        occupied_for_init = {tuple(self.player_pos)}
        
        def get_empty_for_init():
            while True:
                pos = (
                    self.np_random.integers(0, self.GRID_SIZE),
                    self.np_random.integers(0, self.GRID_SIZE),
                )
                if pos not in occupied_for_init:
                    occupied_for_init.add(pos)
                    return list(pos)

        for _ in range(self.NUM_GEMS):
            self.gems.append(get_empty_for_init())
        for _ in range(self.NUM_TRAPS):
            self.traps.append(get_empty_for_init())

        return self._get_observation(), self._get_info()

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_nearest_distance(self, pos, item_list):
        if not item_list:
            return float('inf')
        return min(self._manhattan_distance(pos, item) for item in item_list)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0

        new_pos = list(self.player_pos)

        # Handle movement
        if movement == 1:  # Up
            new_pos[1] -= 1
        elif movement == 2:  # Down
            new_pos[1] += 1
        elif movement == 3:  # Left
            new_pos[0] -= 1
        elif movement == 4:  # Right
            new_pos[0] += 1

        # Check boundaries
        if (0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE):
            dist_gem_before = self._get_nearest_distance(self.player_pos, self.gems)
            dist_trap_before = self._get_nearest_distance(self.player_pos, self.traps)

            self.player_pos = new_pos

            dist_gem_after = self._get_nearest_distance(self.player_pos, self.gems)
            dist_trap_after = self._get_nearest_distance(self.player_pos, self.traps)

            # Continuous rewards
            if dist_gem_after < dist_gem_before:
                reward += 0.1
            if dist_trap_after > dist_trap_before:
                reward += 0.01

        # Check for gem collection
        if self.player_pos in self.gems:
            reward += 10.0
            self.score += 1
            self.gems.remove(self.player_pos)
            self.gems.append(self._get_empty_cell())

        # Check for trap trigger
        if self.player_pos in self.traps:
            reward -= 20.0
            self.lives -= 1
            
            # Respawn the triggered trap
            self.traps.remove(self.player_pos)
            self.traps.append(self._get_empty_cell())
            
            # Reset player position after respawning trap to avoid placing trap on player
            self.player_pos = [0, 0]


        self.steps += 1
        terminated = False
        truncated = False

        # Check termination conditions
        if self.score >= self.WIN_SCORE:
            reward += 100.0
            terminated = True
        elif self.lives <= 0:
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            terminated = True

        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
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
            "lives": self.lives,
        }

    def _grid_to_pixel(self, grid_pos):
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.GRID_AREA_SIZE, self.GRID_AREA_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            x_start = self.GRID_OFFSET_X + i * self.CELL_SIZE
            y_start = self.GRID_OFFSET_Y
            y_end = self.GRID_OFFSET_Y + self.GRID_AREA_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x_start, y_start), (x_start, y_end))

            y_start = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            x_start = self.GRID_OFFSET_X
            x_end = self.GRID_OFFSET_X + self.GRID_AREA_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x_start, y_start), (x_end, y_start))

        # Draw traps
        for trap_pos in self.traps:
            px, py = self._grid_to_pixel(trap_pos)
            size = self.CELL_SIZE * 0.3
            pygame.draw.line(self.screen, self.COLOR_TRAP, (px - size, py - size), (px + size, py + size), 5)
            pygame.draw.line(self.screen, self.COLOR_TRAP, (px - size, py + size), (px + size, py - size), 5)

        # Draw gems
        for gem_pos in self.gems:
            px, py = self._grid_to_pixel(gem_pos)
            radius = int(self.CELL_SIZE * 0.35)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_GEM)
            # Sparkle effect
            sparkle_radius = int(radius * 0.3)
            pygame.gfxdraw.filled_circle(self.screen, px - int(radius*0.4), py - int(radius*0.4), sparkle_radius, self.COLOR_GEM_SPARKLE)
            pygame.gfxdraw.aacircle(self.screen, px - int(radius*0.4), py - int(radius*0.4), sparkle_radius, self.COLOR_GEM_SPARKLE)


        # Draw player
        if self.player_pos:
            px, py = self._grid_to_pixel(self.player_pos)
            size = int(self.CELL_SIZE * 0.7)
            
            # Glow effect
            glow_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (size, size), size)
            self.screen.blit(glow_surf, (px - size, py - size))

            # Player square
            player_rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
            
    def _render_text(self, text, font, position, color, shadow_color=None):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (position[0] + 2, position[1] + 2))
        
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, position)

    def _render_ui(self):
        # Score display (top left)
        score_text = f"Gems: {self.score} / {self.WIN_SCORE}"
        self._render_text(score_text, self.font_large, (20, 20), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Steps/Timer display (top right)
        steps_text = f"Moves: {self.steps} / {self.MAX_STEPS}"
        text_width = self.font_large.size(steps_text)[0]
        self._render_text(steps_text, self.font_large, (self.SCREEN_WIDTH - text_width - 20, 20), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Traps triggered display (bottom left)
        traps_hit = self.START_LIVES - (self.lives if self.lives is not None else self.START_LIVES)
        lives_text = f"Traps Hit: {traps_hit} / {self.START_LIVES}"
        self._render_text(lives_text, self.font_small, (20, self.SCREEN_HEIGHT - 40), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Game Over / Win message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            text_width, text_height = self.font_large.size(msg)
            pos = ((self.SCREEN_WIDTH - text_width) // 2, (self.SCREEN_HEIGHT - text_height) // 2)
            self._render_text(msg, self.font_large, pos, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)


    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # The environment is already initialized by __init__ calling reset().

        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the file directly to test the environment
    # Re-enable display for human play
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Grid")
    clock = pygame.time.Clock()
    
    # Game loop for human play
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
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
                elif event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_q: # Quit on 'q'
                    running = False

        if not done and action[0] != 0: # Only step if a move key was pressed
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()