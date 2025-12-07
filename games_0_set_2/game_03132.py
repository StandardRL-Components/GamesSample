import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the block. Press Space to drop it quickly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks as high as you can. A perfect stack wins, but one wrong move and it's game over!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 40, 20
        self.MAX_STEPS = 2500  # Increased to allow for careful play
        self.WIN_HEIGHT = 20
        self.BLOCK_MOVE_SPEED = 5

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_BASE = (80, 80, 90)
        self.COLOR_TARGET_LINE = (255, 255, 255)
        self.BLOCK_COLORS = [
            (239, 71, 111), (255, 209, 102), (6, 214, 160),
            (17, 138, 178), (7, 59, 76), (255, 128, 128)
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
            self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stacked_blocks = []
        self.height = 0
        self.current_block_rect = None
        self.current_block_color = None
        self.is_dropping = False
        self.block_fall_speed = 0.0
        self.last_space_press = False
        self._np_random = None

        self.reset()

        # Run self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Create a wide base platform
        base_rect = pygame.Rect(
            (self.WIDTH - self.BLOCK_WIDTH * 6) // 2,
            self.HEIGHT - self.BLOCK_HEIGHT,
            self.BLOCK_WIDTH * 6,
            self.BLOCK_HEIGHT
        )
        self.stacked_blocks = [(base_rect, self.COLOR_BASE)]
        self.height = 0

        self.block_fall_speed = 2.0
        self.last_space_press = False

        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def _spawn_new_block(self):
        start_x = self.WIDTH // 2 - self.BLOCK_WIDTH // 2
        self.current_block_rect = pygame.Rect(start_x, 0, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
        self.current_block_color = self.BLOCK_COLORS[self._np_random.integers(0, len(self.BLOCK_COLORS))]
        self.is_dropping = False
        # sfx: spawn_chime.wav

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Action Processing ---
        if movement == 3:  # Left
            self.current_block_rect.x -= self.BLOCK_MOVE_SPEED
        elif movement == 4:  # Right
            self.current_block_rect.x += self.BLOCK_MOVE_SPEED

        self.current_block_rect.left = max(0, self.current_block_rect.left)
        self.current_block_rect.right = min(self.WIDTH, self.current_block_rect.right)

        if space_held and not self.last_space_press:
            self.is_dropping = True
            # sfx: fast_fall_whoosh.wav
        self.last_space_press = space_held

        # --- Game Logic Update ---
        fall_distance = 20.0 if self.is_dropping else self.block_fall_speed
        self.current_block_rect.y += fall_distance

        # Continuous reward for being above the stack
        support_rect = self._get_support_base()
        if self.current_block_rect.left >= support_rect.left and self.current_block_rect.right <= support_rect.right:
            reward += 0.01  # Small reward for good positioning

        # Check for collision and handle placement
        placement_reward = self._handle_block_placement()
        reward += placement_reward

        self.score += reward
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_support_base(self):
        """Calculates the horizontal span of the highest blocks on the stack."""
        if not self.stacked_blocks:
            return pygame.Rect(0, self.HEIGHT, self.WIDTH, 0)

        topmost_y = min(b[0].top for b in self.stacked_blocks)
        top_blocks = [b[0] for b in self.stacked_blocks if b[0].top == topmost_y]

        min_x = min(b.left for b in top_blocks)
        max_x = max(b.right for b in top_blocks)
        return pygame.Rect(min_x, topmost_y, max_x - min_x, 1)

    def _handle_block_placement(self):
        """Checks if the block has landed and processes the result."""
        # Find all blocks the current block is currently colliding with
        colliding_blocks = [
            block_rect for block_rect, _ in self.stacked_blocks
            if self.current_block_rect.colliderect(block_rect)
        ]

        if colliding_blocks:
            # A collision occurred. Place the block and evaluate.

            # Determine the highest surface to land on from all collisions
            highest_surface_y = min(b.top for b in colliding_blocks)
            self.current_block_rect.bottom = highest_surface_y

            # Find all blocks that are actually providing support at this new position.
            # We use a test rect moved down by 1 pixel to robustly find touching blocks below.
            support_test_rect = self.current_block_rect.copy()
            support_test_rect.y += 1

            support_surface_blocks = [
                b[0] for b in self.stacked_blocks if support_test_rect.colliderect(b[0])
            ]

            # If there are no supporters, it means the block landed on a corner or missed.
            if not support_surface_blocks:
                self.game_over = True
                return -100.0

            support_min_x = min(b.left for b in support_surface_blocks)
            support_max_x = max(b.right for b in support_surface_blocks)

            # Check for game over (falling off the side)
            if self.current_block_rect.left < support_min_x or self.current_block_rect.right > support_max_x:
                self.game_over = True
                # sfx: block_crash_fail.wav
                return -100.0

            # --- Successful Placement ---
            # sfx: block_place_thud.wav
            self.stacked_blocks.append((self.current_block_rect.copy(), self.current_block_color))
            self.height = len(self.stacked_blocks) - 1

            placement_reward = 1.0

            # Penalty for large overhang (but stable)
            overhang_left = max(0, support_min_x - self.current_block_rect.left)
            overhang_right = max(0, self.current_block_rect.right - support_max_x)
            if (overhang_left + overhang_right) / self.BLOCK_WIDTH > 0.5:
                placement_reward -= 5.0

            # Check for win condition
            if self.height >= self.WIN_HEIGHT:
                self.game_over = True
                self.win = True
                # sfx: victory_fanfare.wav
                placement_reward += 100.0

            # Increase difficulty every 5 blocks
            if not self.game_over and self.height > 0 and self.height % 5 == 0:
                self.block_fall_speed = min(10.0, self.block_fall_speed + 0.5)

            if not self.game_over:
                self._spawn_new_block()

            return placement_reward

        # Check if block fell off the bottom of the screen
        if self.current_block_rect.top > self.HEIGHT:
            self.game_over = True
            # sfx: block_crash_fail.wav
            return -100.0

        return 0  # Still falling, no event

    def _get_observation(self):
        # --- Background and Grid ---
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, self.BLOCK_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.BLOCK_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # --- Target Line ---
        target_y = self.HEIGHT - (self.WIN_HEIGHT + 1) * self.BLOCK_HEIGHT
        if target_y > 0:
            for x in range(0, self.WIDTH, 20):
                pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (x, target_y), (x + 10, target_y), 1)

        # --- Game Elements ---
        for block_rect, block_color in self.stacked_blocks:
            self._draw_block(block_rect, block_color)

        if self.current_block_rect:
            self._draw_block(self.current_block_rect, self.current_block_color, shadow=True)

        # --- UI Overlay ---
        self._render_ui()

        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_block(self, rect, color, shadow=False):
        border_color = tuple(max(0, c - 40) for c in color)
        pygame.draw.rect(self.screen, border_color, rect)
        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, color, inner_rect)

        if shadow and not self.game_over:
            support_base = self._get_support_base()
            shadow_rect = self.current_block_rect.copy()
            shadow_rect.bottom = support_base.top

            shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            shadow_surface.fill((0, 0, 0, 60))
            self.screen.blit(shadow_surface, shadow_rect.topleft)

    def _render_ui(self):
        score_surf = self.font_ui.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_surf, (10, 10))

        height_surf = self.font_ui.render(f"HEIGHT: {self.height}/{self.WIN_HEIGHT}", True, (255, 255, 255))
        self.screen.blit(height_surf, (10, 35))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        text = "YOU WIN!" if self.win else "GAME OVER"
        color = (100, 255, 100) if self.win else (255, 100, 100)

        text_surf = self.font_game_over.render(text, True, color)
        pos = (
            self.WIDTH // 2 - text_surf.get_width() // 2,
            self.HEIGHT // 2 - text_surf.get_height() // 2,
        )
        self.screen.blit(text_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.height,
            "win": self.win,
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


# Example of how to run the environment for visualization
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    pygame.display.set_caption("Block Stacker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    done = False
    total_reward = 0

    print("\n" + "=" * 30)
    print(env.game_description)
    print(env.user_guide)
    print("=" * 30 + "\n")

    while not done:
        # --- Human Controls ---
        movement = 0  # no-op
        space = 0  # released

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space = 1

        action = [movement, space, 0]  # shift is unused

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(30)  # Run at 30 FPS

    print(f"Game Over! Final Info: {info}")
    pygame.quit()