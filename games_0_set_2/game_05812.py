
# Generated: 2025-08-28T06:09:45.828245
# Source Brief: brief_05812.md
# Brief Index: 5812

        
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
    """
    An isometric arcade game where the player collects food while avoiding patrolling monsters.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move one grid unit per step. Collect all the yellow "
        "food items while avoiding the red monsters."
    )

    # Short, user-facing description of the game
    game_description = (
        "Collect food in an isometric 2D arena while avoiding hungry monsters to achieve a high score."
    )

    # Frames only advance when an action is received
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (30, 35, 40)
    COLOR_GRID = (50, 55, 60)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_MONSTER = (255, 50, 50)
    COLOR_FOOD = (255, 220, 0)
    COLOR_SHADOW = (20, 20, 25, 128)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_HEART = (255, 80, 80)
    COLOR_HEART_EMPTY = (80, 80, 80)

    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 16, 16
    TILE_WIDTH, TILE_HEIGHT = 32, 16
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 80

    # Game Parameters
    MAX_STEPS = 1000
    NUM_MONSTERS = 3
    NUM_FOOD = 15
    PLAYER_LIVES = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup (headless)
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("sans-serif", 48, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.monsters = []
        self.food_items = []
        self.last_reward_info = ""

        # Use a seeded random number generator for reproducibility
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.lives = self.PLAYER_LIVES
        self.game_over = False
        self.last_reward_info = ""

        # --- Place Game Objects ---
        # Create a set of all possible grid coordinates
        all_coords = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))

        # Place player
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        all_coords.remove(tuple(self.player_pos))

        # Place monsters with predefined patrol paths
        self.monsters = []
        monster_paths = [
            self._generate_patrol_path(2, 2, 5, 5),
            self._generate_patrol_path(self.GRID_WIDTH - 7, 2, 5, 10),
            self._generate_patrol_path(2, self.GRID_HEIGHT - 7, 10, 5),
        ]
        for path in monster_paths:
            start_pos = path[0]
            self.monsters.append({"pos": list(start_pos), "path": path, "path_index": 0})
            if tuple(start_pos) in all_coords:
                all_coords.remove(tuple(start_pos))

        # Place food randomly
        num_food_to_spawn = min(self.NUM_FOOD, len(all_coords))
        food_coords = self.np_random.choice(list(all_coords), size=num_food_to_spawn, replace=False)
        self.food_items = [list(coord) for coord in food_coords]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Process Action ---
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self.steps += 1
        reward = -0.01  # Small penalty for each step to encourage efficiency
        self.last_reward_info = ""

        # 1. Move Player
        if movement == 1:  # Up (Isometric North-East)
            self.player_pos[1] -= 1
        elif movement == 2:  # Down (Isometric South-West)
            self.player_pos[1] += 1
        elif movement == 3:  # Left (Isometric North-West)
            self.player_pos[0] -= 1
        elif movement == 4:  # Right (Isometric South-East)
            self.player_pos[0] += 1
        
        # Clamp player position to grid bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)

        # 2. Move Monsters
        for monster in self.monsters:
            monster["path_index"] = (monster["path_index"] + 1) % len(monster["path"])
            monster["pos"] = list(monster["path"][monster["path_index"]])

        # 3. Check for Collisions and Collections
        # Food collection
        collected_food_index = -1
        for i, food_pos in enumerate(self.food_items):
            if self.player_pos == food_pos:
                collected_food_index = i
                break
        
        if collected_food_index != -1:
            self.food_items.pop(collected_food_index)
            self.score += 1
            reward += 10
            self.last_reward_info = "+10 (Food)"
            # sfx: positive chime

        # Monster collision
        for monster in self.monsters:
            if self.player_pos == monster["pos"]:
                self.lives -= 1
                reward -= 5
                self.last_reward_info = "-5 (Hit)"
                # sfx: damage sound
                # Reset player to start to avoid repeated hits
                self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
                break

        # 4. Check for Termination Conditions
        terminated = False
        if self.lives <= 0:
            terminated = True
            reward += -50
            self.last_reward_info = "-50 (Game Over)"
            self.game_over = True
        elif not self.food_items:
            terminated = True
            reward += 50
            self.last_reward_info = "+50 (Victory!)"
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen
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
            "lives": self.lives,
            "food_remaining": len(self.food_items)
        }

    def _render_game(self):
        # --- Draw Grid ---
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # --- Collect and Sort All Drawable Entities ---
        # This ensures correct draw order (painter's algorithm)
        entities = []
        for food_pos in self.food_items:
            entities.append(("food", food_pos))
        for monster in self.monsters:
            entities.append(("monster", monster["pos"]))
        if not self.game_over or self.lives > 0: # Don't draw player if they lost
            entities.append(("player", self.player_pos))
        
        # Sort by depth (sum of grid coordinates)
        entities.sort(key=lambda e: e[1][0] + e[1][1])

        # --- Draw Entities ---
        for entity_type, pos in entities:
            if entity_type == "player":
                self._draw_iso_entity(pos, self.COLOR_PLAYER, bob=True)
            elif entity_type == "monster":
                self._draw_iso_entity(pos, self.COLOR_MONSTER)
            elif entity_type == "food":
                self._draw_iso_entity(pos, self.COLOR_FOOD, pulse=True)

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives display (hearts)
        for i in range(self.PLAYER_LIVES):
            heart_pos = (self.SCREEN_WIDTH - 30 - (i * 25), 15)
            color = self.COLOR_HEART if i < self.lives else self.COLOR_HEART_EMPTY
            self._draw_heart(self.screen, heart_pos, color)
        
        # Game Over / Victory message
        if self.game_over:
            if not self.food_items:
                msg = "VICTORY!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_MONSTER
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(end_text, text_rect)


    # --- Helper Methods ---

    def _iso_to_screen(self, grid_x, grid_y):
        """Converts grid coordinates to screen pixel coordinates."""
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * (self.TILE_WIDTH / 2)
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _generate_patrol_path(self, x, y, w, h):
        """Generates a list of coordinates for a rectangular patrol path."""
        path = []
        for i in range(w): path.append((x + i, y))
        for i in range(h): path.append((x + w - 1, y + i))
        for i in range(w): path.append((x + w - 1 - i, y + h - 1))
        for i in range(h): path.append((x, y + h - 1 - i))
        return path

    def _draw_iso_entity(self, pos, color, bob=False, pulse=False):
        """Draws a diamond-shaped entity on the isometric grid with effects."""
        center_x, center_y = self._iso_to_screen(pos[0], pos[1])
        
        # Vertical offset for animations
        offset_y = 0
        if bob:
            offset_y = -3 + math.sin(self.steps * 0.2) * 3
        
        # Size multiplier for animations
        size_mult = 1.0
        if pulse:
            size_mult = 1.0 + math.sin(self.steps * 0.3) * 0.15

        tile_w = self.TILE_WIDTH * 0.6 * size_mult
        tile_h = self.TILE_HEIGHT * 1.2 * size_mult

        # Shadow
        shadow_rect = (
            center_x - tile_w / 2, 
            center_y + self.TILE_HEIGHT / 2 - 4, 
            tile_w, 
            tile_h / 2
        )
        shadow_surface = pygame.Surface((shadow_rect[2], shadow_rect[3]), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, self.COLOR_SHADOW, (0, 0, shadow_rect[2], shadow_rect[3]))
        self.screen.blit(shadow_surface, (shadow_rect[0], shadow_rect[1]))

        # Main body (diamond shape)
        points = [
            (center_x, center_y - tile_h / 2 + offset_y),
            (center_x + tile_w / 2, center_y + offset_y),
            (center_x, center_y + tile_h / 2 + offset_y),
            (center_x - tile_w / 2, center_y + offset_y),
        ]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(self.screen, int_points, color)
        pygame.gfxdraw.filled_polygon(self.screen, int_points, color)

    def _draw_heart(self, surface, pos, color):
        """Draws a heart shape for the life counter."""
        x, y = pos
        s = 10  # Size of the heart
        points = [
            (x, y - s * 0.3),
            (x + s * 0.5, y - s),
            (x + s, y - s * 0.3),
            (x, y + s),
            (x - s, y - s * 0.3),
            (x - s * 0.5, y - s),
        ]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(surface, int_points, color)
        pygame.gfxdraw.filled_polygon(surface, int_points, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
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
        assert "score" in info and "steps" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert "score" in info and "steps" in info
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == "__main__":
    # Set SDL to dummy to run without a display
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    env.validate_implementation()

    # --- To visualize the game, comment out the os.environ line above ---
    # --- and run the following code. ---
    #
    # os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "quartz"
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    #
    # screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # pygame.display.set_caption("Isometric Collector")
    # clock = pygame.time.Clock()
    #
    # running = True
    # while running:
    #     movement = 0 # No-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]:
    #         movement = 1
    #     elif keys[pygame.K_DOWN]:
    #         movement = 2
    #     elif keys[pygame.K_LEFT]:
    #         movement = 3
    #     elif keys[pygame.K_RIGHT]:
    #         movement = 4
    #
    #     action = [movement, 0, 0] # Space/Shift not used
    #     obs, reward, terminated, truncated, info = env.step(action)
    #
    #     # Draw the observation from the environment
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    #         pygame.time.wait(3000) # Pause for 3 seconds
    #         obs, info = env.reset()
    #
    #     clock.tick(10) # Control game speed for human play
    #
    # env.close()