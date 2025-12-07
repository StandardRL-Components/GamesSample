import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import os
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Press space to attack adjacent squares. Reach the gold exit before time runs out!"
    )

    # Short, user-facing description of the game
    game_description = (
        "Navigate a dungeon, defeat enemies, and collect potions to reach the exit before the timer expires."
    )

    # Frames only advance when an action is received
    auto_advance = False

    # --- Constants ---
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 25
    GRID_HEIGHT = 15
    CELL_SIZE = 25
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_POTION = (50, 255, 50)
    COLOR_EXIT = (255, 200, 0)
    COLOR_ATTACK = (255, 255, 255)
    COLOR_HEALTH_BAR = (0, 200, 0)
    COLOR_HEALTH_BAR_BG = (100, 0, 0)
    COLOR_UI_TEXT = (220, 220, 220)

    # Game Parameters
    INITIAL_HEALTH = 3
    MAX_HEALTH = 3
    INITIAL_TIME = 60
    NUM_ENEMIES = 5
    NUM_POTIONS = 3
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        # Action: [Movement, Attack, Unused]
        # Movement: 0:No-op, 1:Up, 2:Down, 3:Left, 4:Right
        # Attack: 0:No, 1:Yes
        # Unused: 0:No, 1:Yes
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.player_health = None
        self.enemies = None
        self.potions = None
        self.exit_pos = None
        self.time_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.attack_vfx_positions = []
        self.last_reward_info = {}

        # This will be seeded in reset()
        self.np_random = None

        # The validation function requires a valid state, which is only created by reset().
        # We call reset() here to ensure the environment is ready for validation and subsequent use.
        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.player_health = self.INITIAL_HEALTH
        self.time_remaining = self.INITIAL_TIME
        self.attack_vfx_positions = []
        self.last_reward_info = {}

        # Procedurally generate the level
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = -0.1  # Time penalty
        self.time_remaining -= 1
        self.steps += 1
        self.attack_vfx_positions = []
        
        old_player_pos = self.player_pos
        old_dist_to_exit = self._manhattan_distance(old_player_pos, self.exit_pos)

        # --- Player Action Phase ---
        if space_pressed:
            reward += self._handle_player_attack()
        elif movement != 0:
            self._handle_player_movement(movement)
        
        # --- Player Interaction & Enemy Phase ---
        reward += self._handle_interactions()
        reward += self._handle_enemy_turns()

        # --- Reward & Termination Phase ---
        # Distance to exit reward
        new_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
        if new_dist_to_exit < old_dist_to_exit:
            reward += 1
        elif new_dist_to_exit > old_dist_to_exit:
            reward -= 2

        # Check termination conditions
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 50
            terminated = True
        elif self.player_health <= 0:
            terminated = True
        elif self.time_remaining <= 0:
            reward -= 50
            terminated = True

        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _generate_level(self):
        """Creates a new random level layout."""
        all_positions = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_positions)

        # Player starts on the left, exit on the right
        player_y = self.np_random.integers(0, self.GRID_HEIGHT)
        self.player_pos = (self.np_random.integers(0, self.GRID_WIDTH // 4), player_y)
        
        exit_y = self.np_random.integers(0, self.GRID_HEIGHT)
        self.exit_pos = (self.np_random.integers(self.GRID_WIDTH * 3 // 4, self.GRID_WIDTH), exit_y)

        # Ensure player/exit don't spawn on each other
        occupied = {self.player_pos, self.exit_pos}
        
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            pos = all_positions.pop()
            while pos in occupied:
                pos = all_positions.pop()
            self.enemies.append(pos)
            occupied.add(pos)

        self.potions = []
        for _ in range(self.NUM_POTIONS):
            pos = all_positions.pop()
            while pos in occupied:
                pos = all_positions.pop()
            self.potions.append(pos)
            occupied.add(pos)

    def _handle_player_attack(self):
        """Processes player attack action."""
        reward = 0
        attack_targets = [
            (self.player_pos[0] + dx, self.player_pos[1] + dy)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
        ]
        self.attack_vfx_positions = [t for t in attack_targets if 0 <= t[0] < self.GRID_WIDTH and 0 <= t[1] < self.GRID_HEIGHT]
        
        enemies_defeated = [e for e in self.enemies if e in attack_targets]
        if enemies_defeated:
            self.enemies = [e for e in self.enemies if e not in enemies_defeated]
            reward += 5 * len(enemies_defeated)
        return reward

    def _handle_player_movement(self, movement):
        """Processes player movement action."""
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
        if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
            self.player_pos = new_pos

    def _handle_interactions(self):
        """Handles player interactions with objects after moving."""
        reward = 0
        # Potion collection
        if self.player_pos in self.potions:
            self.potions.remove(self.player_pos)
            if self.player_health < self.MAX_HEALTH:
                self.player_health += 1
            reward += 2

        # Bumping into an enemy
        if self.player_pos in self.enemies:
            self.enemies.remove(self.player_pos)
            self.player_health -= 1
            reward += 5  # Reward for defeating enemy by bumping
        return reward

    def _handle_enemy_turns(self):
        """Processes movement and attacks for all enemies."""
        reward = 0
        enemies_defeated_by_bumping = []
        
        # Shuffle to avoid bias in movement order
        enemy_indices = list(range(len(self.enemies)))
        self.np_random.shuffle(enemy_indices)

        for i in enemy_indices:
            enemy_pos = self.enemies[i]
            # Simple random walk (including staying still)
            moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
            move = moves[self.np_random.integers(0, len(moves))]
            new_enemy_pos = (enemy_pos[0] + move[0], enemy_pos[1] + move[1])

            # Check boundaries and collisions with other enemies
            other_enemy_positions = [self.enemies[j] for j in range(len(self.enemies)) if i != j]
            if (0 <= new_enemy_pos[0] < self.GRID_WIDTH and
                0 <= new_enemy_pos[1] < self.GRID_HEIGHT and
                new_enemy_pos not in other_enemy_positions):
                self.enemies[i] = new_enemy_pos
            
            # Check if enemy moved into player
            if self.enemies[i] == self.player_pos:
                self.player_health -= 1
                enemies_defeated_by_bumping.append(self.enemies[i])
                reward += 5

        # Remove enemies that bumped into the player
        if enemies_defeated_by_bumping:
            self.enemies = [e for e in self.enemies if e not in enemies_defeated_by_bumping]
        
        return reward

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
            "health": self.player_health,
            "time": self.time_remaining,
            "enemies_left": len(self.enemies),
        }

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for rendering."""
        px = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        return int(px), int(py)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _render_game(self):
        """Renders all game elements onto the screen."""
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            start_pos = self._grid_to_pixel((x, 0))
            end_pos = self._grid_to_pixel((x, self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_pos[0], self.GRID_OFFSET_Y), (start_pos[0], end_pos[1]))
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = self._grid_to_pixel((0, y))
            end_pos = self._grid_to_pixel((self.GRID_WIDTH, y))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, start_pos[1]), (end_pos[0], start_pos[1]))
        
        # Draw exit
        exit_px = self._grid_to_pixel(self.exit_pos)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (exit_px[0], exit_px[1], self.CELL_SIZE, self.CELL_SIZE))
        pygame.gfxdraw.filled_circle(self.screen, exit_px[0] + self.CELL_SIZE // 2, exit_px[1] + self.CELL_SIZE // 2, self.CELL_SIZE // 4, (255,255,255,50))


        # Draw potions
        for potion_pos in self.potions:
            potion_px = self._grid_to_pixel(potion_pos)
            pygame.draw.rect(self.screen, self.COLOR_POTION, (potion_px[0] + self.CELL_SIZE//4, potion_px[1] + self.CELL_SIZE//4, self.CELL_SIZE//2, self.CELL_SIZE//2))

        # Draw enemies
        for enemy_pos in self.enemies:
            enemy_px = self._grid_to_pixel(enemy_pos)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (enemy_px[0], enemy_px[1], self.CELL_SIZE, self.CELL_SIZE))

        # Draw player
        player_px = self._grid_to_pixel(self.player_pos)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (player_px[0], player_px[1], self.CELL_SIZE, self.CELL_SIZE))

        # Draw player health bar
        health_bar_width = self.CELL_SIZE
        health_bar_height = 5
        health_bar_x = player_px[0]
        health_bar_y = player_px[1] - health_bar_height - 2
        
        current_health_width = int(health_bar_width * (max(0, self.player_health) / self.MAX_HEALTH))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (health_bar_x, health_bar_y, current_health_width, health_bar_height))

        # Draw attack VFX
        for vfx_pos in self.attack_vfx_positions:
            vfx_px = self._grid_to_pixel(vfx_pos)
            pygame.draw.rect(self.screen, self.COLOR_ATTACK, (vfx_px[0], vfx_px[1], self.CELL_SIZE, self.CELL_SIZE), 3)


    def _render_ui(self):
        """Renders UI elements like score and time."""
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_text = self.font_ui.render(f"TIME: {self.time_remaining}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU DIED"
            if self.player_pos == self.exit_pos:
                msg = "YOU ESCAPED!"
            elif self.time_remaining <= 0:
                msg = "TIME'S UP!"

            end_text = self.font_game_over.render(msg, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, end_rect)


    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset and initial observation
        # The environment state is initialized in __init__ by calling reset(),
        # so we can safely get an observation.
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    # Unset the dummy video driver to allow a window to be created
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Dungeon Crawler")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)
    
    # Simple turn-based input handling
    last_action_time = pygame.time.get_ticks()
    action_delay = 150 # ms

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if terminated: # Don't process game input if game is over
                    continue
                # Movement
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                # Actions
                elif event.key == pygame.K_SPACE:
                    action[1] = 1

        # Only step if an action was taken
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate
        
    env.close()