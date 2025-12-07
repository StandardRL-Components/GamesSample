import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    A Gymnasium environment for a fast-paced arcade grid game.
    The player must collect gems while dodging enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move. Collect gems (squares) and avoid enemies (circles)."
    )

    game_description = (
        "Navigate a grid, grabbing gems while dodging increasingly aggressive enemies to amass a high score before time runs out."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20

    MAX_STEPS = 1000
    WIN_GEM_COUNT = 50
    INITIAL_GEMS = 5
    INITIAL_ENEMIES = 3
    INITIAL_GEM_RESPAWN_DELAY = 15
    INITIAL_ENEMY_MOVE_PERIOD = 50

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 50)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 64)
    COLOR_TEXT = (255, 255, 255)
    COLOR_ENEMY_RED = (255, 80, 80)
    COLOR_ENEMY_BLUE = (80, 80, 255)
    COLOR_ENEMY_GREEN = (80, 255, 80)
    COLOR_ENEMY_FLASH = (255, 255, 255)
    GEM_COLORS = [
        (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0)
    ]

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_huge = pygame.font.SysFont("monospace", 48, bold=True)

        self.np_random = None
        self.player_pos = None
        self.enemies = []
        self.gems = []
        self.gem_respawn_queue = []
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.win_message = ""
        self.enemy_speed_multiplier = 1.0
        self.gem_respawn_delay = self.INITIAL_GEM_RESPAWN_DELAY

        # Defer validation until after the first reset, as rendering requires an initialized state.
        # The validation function will call reset itself.
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.win_message = ""
        self.enemy_speed_multiplier = 1.0
        self.gem_respawn_delay = self.INITIAL_GEM_RESPAWN_DELAY

        self.player_pos = pygame.Vector2(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)

        occupied_cells = {tuple(self.player_pos)}

        self.gems = []
        for i in range(self.INITIAL_GEMS):
            pos = self._find_empty_cell(occupied_cells)
            self.gems.append({'pos': pos, 'color_idx': self.np_random.integers(0, len(self.GEM_COLORS))})
            occupied_cells.add(tuple(pos))

        self.gem_respawn_queue = []

        self.enemies = []
        enemy_types = ['red', 'blue', 'green']
        for i in range(self.INITIAL_ENEMIES):
            pos = self._find_empty_cell(occupied_cells)
            enemy_type = enemy_types[i % len(enemy_types)]
            direction = self._get_initial_enemy_direction(enemy_type)
            self.enemies.append({
                'pos': pos,
                'type': enemy_type,
                'dir': direction,
                'move_counter': 0,
                'flash_timer': 10  # Flash for 10 frames on spawn
            })
            occupied_cells.add(tuple(pos))

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0

        # --- 1. Calculate pre-move state for reward ---
        dist_before = self._get_distance_to_nearest_gem()

        # --- 2. Process player action ---
        movement = action[0]
        prev_player_pos = self.player_pos.copy()

        if movement == 1:  # Up
            self.player_pos.y -= 1
        elif movement == 2:  # Down
            self.player_pos.y += 1
        elif movement == 3:  # Left
            self.player_pos.x -= 1
        elif movement == 4:  # Right
            self.player_pos.x += 1

        # Clamp player position to grid
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.GRID_WIDTH - 1)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.GRID_HEIGHT - 1)

        # --- 3. Calculate movement reward ---
        dist_after = self._get_distance_to_nearest_gem()
        if dist_after < dist_before:
            reward += 1.0  # Moved closer to a gem
        elif self.player_pos != prev_player_pos:
            reward -= 0.1  # Moved away or parallel

        # --- 4. Update game state ---
        self._update_enemies()
        self._update_gem_spawns()

        # --- 5. Handle interactions and events ---
        # Gem collection
        collected_gem_index = -1
        for i, gem in enumerate(self.gems):
            if tuple(self.player_pos) == tuple(gem['pos']):
                collected_gem_index = i
                break

        if collected_gem_index != -1:
            self.gems.pop(collected_gem_index)
            self.score += 10
            self.gems_collected += 1
            reward += 10.0
            self.gem_respawn_queue.append(self.gem_respawn_delay)
            # Sound: GEM_COLLECT

        # --- 6. Check for termination conditions ---
        terminated = False
        # Enemy collision
        for enemy in self.enemies:
            if tuple(self.player_pos) == tuple(enemy['pos']):
                self.game_over = True
                terminated = True
                reward -= 100.0
                self.win_message = "GAME OVER"
                # Sound: GAME_OVER
                break

        if not terminated:
            # Win condition
            if self.gems_collected >= self.WIN_GEM_COUNT:
                self.game_over = True
                terminated = True
                time_bonus = (self.MAX_STEPS - self.steps) * 0.1
                reward += 100.0 + time_bonus
                self.score += int(time_bonus)
                self.win_message = "YOU WIN!"
                # Sound: WIN
            # Max steps reached
            elif self.steps >= self.MAX_STEPS - 1:
                self.game_over = True
                terminated = True
                self.win_message = "TIME UP"

        self.steps += 1
        self._update_difficulty()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_enemies(self):
        move_period = self.INITIAL_ENEMY_MOVE_PERIOD / self.enemy_speed_multiplier
        for enemy in self.enemies:
            if enemy['flash_timer'] > 0:
                enemy['flash_timer'] -= 1

            enemy['move_counter'] += 1
            if enemy['move_counter'] < move_period:
                continue

            enemy['move_counter'] = 0
            new_pos = enemy['pos'] + enemy['dir']

            # Wall collision logic
            if not (0 <= new_pos.x < self.GRID_WIDTH):
                enemy['dir'].x *= -1
            if not (0 <= new_pos.y < self.GRID_HEIGHT):
                enemy['dir'].y *= -1

            enemy['pos'] += enemy['dir']
            enemy['pos'].x = np.clip(enemy['pos'].x, 0, self.GRID_WIDTH - 1)
            enemy['pos'].y = np.clip(enemy['pos'].y, 0, self.GRID_HEIGHT - 1)

    def _update_gem_spawns(self):
        new_queue = []
        for timer in self.gem_respawn_queue:
            timer -= 1
            if timer <= 0:
                occupied = {tuple(self.player_pos)} | {tuple(e['pos']) for e in self.enemies} | {
                    tuple(g['pos']) for g in self.gems}
                pos = self._find_empty_cell(occupied)
                self.gems.append({'pos': pos, 'color_idx': self.np_random.integers(0, len(self.GEM_COLORS))})
            else:
                new_queue.append(timer)
        self.gem_respawn_queue = new_queue

    def _update_difficulty(self):
        # Increase enemy speed every 100 steps
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_speed_multiplier = min(self.enemy_speed_multiplier + 0.2, self.INITIAL_ENEMY_MOVE_PERIOD)

        # Decrease gem respawn delay every 250 steps
        if self.steps > 0 and self.steps % 250 == 0:
            self.gem_respawn_delay = max(1, self.gem_respawn_delay - 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw gems
        for gem in self.gems:
            rect = pygame.Rect(gem['pos'].x * self.CELL_SIZE, gem['pos'].y * self.CELL_SIZE, self.CELL_SIZE,
                               self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.GEM_COLORS[gem['color_idx']], rect.inflate(-4, -4))

        # Draw enemies
        for enemy in self.enemies:
            center_x = int(enemy['pos'].x * self.CELL_SIZE + self.CELL_SIZE / 2)
            center_y = int(enemy['pos'].y * self.CELL_SIZE + self.CELL_SIZE / 2)
            radius = self.CELL_SIZE // 2 - 2

            color = self.COLOR_ENEMY_FLASH if enemy['flash_timer'] > 0 else {
                'red': self.COLOR_ENEMY_RED, 'blue': self.COLOR_ENEMY_BLUE, 'green': self.COLOR_ENEMY_GREEN
            }[enemy['type']]

            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)

        # Draw player
        player_rect = pygame.Rect(self.player_pos.x * self.CELL_SIZE, self.player_pos.y * self.CELL_SIZE,
                                  self.CELL_SIZE, self.CELL_SIZE)

        # Glow effect
        glow_surf = pygame.Surface((self.CELL_SIZE * 2, self.CELL_SIZE * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.CELL_SIZE, self.CELL_SIZE), self.CELL_SIZE)
        self.screen.blit(glow_surf, player_rect.topleft - pygame.Vector2(self.CELL_SIZE / 2, self.CELL_SIZE / 2))

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-2, -2))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Timer
        time_left = self.MAX_STEPS - self.steps
        time_text = self.font_large.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 5))

        # Gems collected
        gem_text = self.font_small.render(f"GEMS: {self.gems_collected} / {self.WIN_GEM_COUNT}", True,
                                          self.COLOR_TEXT)
        self.screen.blit(gem_text,
                         ((self.SCREEN_WIDTH - gem_text.get_width()) / 2, self.SCREEN_HEIGHT - gem_text.get_height() - 5))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_huge.render(self.win_message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
        }

    def _find_empty_cell(self, occupied_cells):
        while True:
            pos = pygame.Vector2(
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if tuple(pos) not in occupied_cells:
                return pos

    def _get_initial_enemy_direction(self, enemy_type):
        if enemy_type == 'red':
            return pygame.Vector2(self.np_random.choice([-1, 1]), 0)
        elif enemy_type == 'blue':
            return pygame.Vector2(0, self.np_random.choice([-1, 1]))
        elif enemy_type == 'green':
            return pygame.Vector2(self.np_random.choice([-1, 1]), self.np_random.choice([-1, 1]))
        return pygame.Vector2(0, 0)

    def _get_manhattan_distance(self, pos1, pos2):
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def _get_distance_to_nearest_gem(self):
        if not self.gems:
            return float('inf')
        return min(self._get_manhattan_distance(self.player_pos, gem['pos']) for gem in self.gems)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset first to initialize the state needed for rendering observations
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)

        # Test observation space (now that state is initialized)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is recommended to run this in a proper python environment, not a notebook
    env = GameEnv()
    obs, info = env.reset()

    running = True
    game_over = False

    # Create a display for manual playing
    pygame.display.set_caption("Grid Gem Collector")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = np.array([0, 0, 0])  # No-op, no buttons

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                game_over = False

        if not game_over:
            keys = pygame.key.get_pressed()
            movement = 0  # no-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

            action = np.array([movement, 0, 0])  # Space and shift are not used

            obs, reward, terminated, truncated, info = env.step(action)
            game_over = terminated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(10)  # Control the speed of manual play

    env.close()