import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:26:36.773225
# Source Brief: brief_01697.md
# Brief Index: 1697
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A stealth-based survival game where you manipulate gravity to navigate
    procedurally generated spaceships, evade robots, and manage your
    resources to escape.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A stealth-based survival game where you manipulate gravity to navigate "
        "procedurally generated spaceships, evade robots, and manage your resources to escape."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press space to flip gravity. "
        "Collect food and water to survive and reach the escape pod."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_SIZE = 40
    GRID_WIDTH = SCREEN_WIDTH // TILE_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // TILE_SIZE

    # Colors
    COLOR_BG = (26, 26, 29) # #1a1a1d
    COLOR_WALL = (60, 60, 65) # #3c3c41
    COLOR_WALL_OUTLINE = (40, 40, 43) # #28282b
    COLOR_PLAYER = (77, 255, 77) # #4dff4d
    COLOR_ROBOT = (255, 77, 77) # #ff4d4d
    COLOR_WATER = (77, 148, 255) # #4d94ff
    COLOR_FOOD = (255, 255, 77) # #ffff4d
    COLOR_POD = (255, 153, 51) # #ff9933
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BAR_BG = (80, 80, 80)

    # Game Parameters
    MAX_STEPS = 1000
    INITIAL_HUNGER = 100
    INITIAL_HYDRATION = 100
    RESOURCE_DECAY_RATE = 10 # Decrease every X steps
    RESOURCE_GAIN = 50
    NUM_ROBOTS = 3
    NUM_FOOD = 2
    NUM_WATER = 2
    INITIAL_ROBOT_SPEED = 0.5
    MAX_ROBOT_SPEED = 2.0
    ROBOT_SPEED_INCREASE_RATE = 500 # Increase every X steps
    ROBOT_SPEED_INCREASE_AMOUNT = 0.1
    GRAVITY_FLIP_COOLDOWN = 10 # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # Initialize state variables to prevent attribute errors before reset
        self._initialize_state()
        
        # This is a critical self-check
        # self.validate_implementation() # Commented out for submission, but useful for dev

    def _initialize_state(self):
        """Initializes all state variables to a default value."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.player_pos = [0, 0]
        self.escape_pod_pos = [0, 0]
        self.food_items = []
        self.water_items = []
        
        self.robots = []
        
        self.hunger = self.INITIAL_HUNGER
        self.hydration = self.INITIAL_HYDRATION
        
        self.gravity_dir = pygame.Vector2(0, 1) # (0,1) = Down, (0,-1) = Up
        self.gravity_flip_cooldown_timer = 0
        
        self.particles = []
        self.last_space_press = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Procedurally generates a new level layout."""
        # 1. Fill grid with walls
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # 2. Carve out a path using a random walk
        empty_tiles = []
        start_pos = [self.np_random.integers(1, self.GRID_WIDTH - 1), 1]
        self.player_pos = start_pos
        
        carver_pos = list(start_pos)
        self.grid[carver_pos[0], carver_pos[1]] = 0
        empty_tiles.append(tuple(carver_pos))
        
        for _ in range(100): # Number of carving steps
            # Directions: N, S, E, W
            dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
            self.np_random.shuffle(dirs)
            moved = False
            for dx, dy in dirs:
                nx, ny = carver_pos[0] + dx, carver_pos[1] + dy
                # Check bounds (stay away from edges)
                if 1 <= nx < self.GRID_WIDTH - 1 and 1 <= ny < self.GRID_HEIGHT - 1:
                    if self.grid[nx, ny] == 1:
                        carver_pos = [nx, ny]
                        self.grid[nx, ny] = 0
                        empty_tiles.append(tuple(carver_pos))
                        moved = True
                        break
            if not moved: # If stuck, jump to a random empty tile
                carver_pos = list(self.np_random.choice(empty_tiles))

        # 3. Place escape pod far from player
        self.escape_pod_pos = list(empty_tiles[-1])
        
        # 4. Get a list of valid spawn points (not player start or pod end)
        valid_spawns = [t for t in empty_tiles if t != tuple(self.player_pos) and t != tuple(self.escape_pod_pos)]
        self.np_random.shuffle(valid_spawns)

        # 5. Place items and robots
        for _ in range(self.NUM_FOOD):
            if valid_spawns: self.food_items.append(list(valid_spawns.pop()))
        for _ in range(self.NUM_WATER):
            if valid_spawns: self.water_items.append(list(valid_spawns.pop()))
        
        robot_speed = self.INITIAL_ROBOT_SPEED
        for _ in range(self.NUM_ROBOTS):
            if valid_spawns:
                pos = list(valid_spawns.pop())
                self.robots.append({
                    "pos": pos,
                    "speed": robot_speed,
                    "move_accumulator": 0.0,
                    "dir": self.np_random.choice([-1, 1]) # -1 for left, 1 for right
                })

    def _apply_gravity(self, pos):
        """Applies gravity to a position, making it fall until it hits a wall."""
        current_pos = list(pos)
        while True:
            next_pos = [current_pos[0] + int(self.gravity_dir.x), current_pos[1] + int(self.gravity_dir.y)]
            if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT) or self.grid[next_pos[0], next_pos[1]] == 1:
                break # Hit a wall or edge
            current_pos = next_pos
        return current_pos

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1 # Survival reward
        self.steps += 1
        
        # --- 1. Handle Actions ---
        movement, space_press, _ = action
        space_held = space_press == 1

        # Gravity Flip (on press, not hold)
        if space_held and not self.last_space_press and self.gravity_flip_cooldown_timer == 0:
            self.gravity_dir *= -1
            self.gravity_flip_cooldown_timer = self.GRAVITY_FLIP_COOLDOWN
            reward += 10 # Successful flip reward
            # SFX: Gravity shift whoosh
            self._create_gravity_particles()
        self.last_space_press = space_held

        # Player Movement
        move_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
        target_pos = [self.player_pos[0] + move_dir[0], self.player_pos[1] + move_dir[1]]
        
        if 0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT and self.grid[target_pos[0], target_pos[1]] == 0:
            self.player_pos = target_pos
            # SFX: Player step sound

        # --- 2. Update World State ---
        # Apply gravity to player
        self.player_pos = self._apply_gravity(self.player_pos)
        
        # Update and move robots
        robot_speed = self.INITIAL_ROBOT_SPEED + (self.steps // self.ROBOT_SPEED_INCREASE_RATE) * self.ROBOT_SPEED_INCREASE_AMOUNT
        robot_speed = min(robot_speed, self.MAX_ROBOT_SPEED)

        for robot in self.robots:
            robot["speed"] = robot_speed
            robot["move_accumulator"] += robot["speed"]
            while robot["move_accumulator"] >= 1.0:
                robot["move_accumulator"] -= 1.0
                # Robots patrol horizontally
                r_target_pos = [robot["pos"][0] + robot["dir"], robot["pos"][1]]
                if 0 <= r_target_pos[0] < self.GRID_WIDTH and self.grid[r_target_pos[0], r_target_pos[1]] == 0:
                    robot["pos"] = r_target_pos
                else:
                    robot["dir"] *= -1 # Reverse direction on collision
            robot["pos"] = self._apply_gravity(robot["pos"])

        # Update timers and resources
        if self.gravity_flip_cooldown_timer > 0:
            self.gravity_flip_cooldown_timer -= 1
            
        if self.steps % self.RESOURCE_DECAY_RATE == 0:
            self.hunger = max(0, self.hunger - 1)
            self.hydration = max(0, self.hydration - 1)

        # --- 3. Check Events & Collisions ---
        # Collect items
        if self.player_pos in self.food_items:
            self.food_items.remove(self.player_pos)
            self.hunger = min(self.INITIAL_HUNGER, self.hunger + self.RESOURCE_GAIN)
            reward += 5
            self.score += 5
            # SFX: Item collect chime
        if self.player_pos in self.water_items:
            self.water_items.remove(self.player_pos)
            self.hydration = min(self.INITIAL_HYDRATION, self.hydration + self.RESOURCE_GAIN)
            reward += 5
            self.score += 5
            # SFX: Item collect chime

        # --- 4. Check Termination Conditions ---
        # Robot collision
        for robot in self.robots:
            if abs(self.player_pos[0] - robot["pos"][0]) <= 1 and abs(self.player_pos[1] - robot["pos"][1]) <= 1:
                self.game_over = True
                reward = -100
                self.score -= 100
                # SFX: Player death / detected sound
                break
        
        # Resource depletion
        if self.hunger <= 0 or self.hydration <= 0:
            self.game_over = True
            reward = -100
            self.score -= 100
            # SFX: Player death / starvation sound
        
        # Reached escape pod
        if self.player_pos == self.escape_pod_pos:
            self.game_over = True
            self.victory = True
            reward = 100
            self.score += 100
            # SFX: Victory fanfare
        
        # Max steps
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            # No specific penalty for timeout, just ends the episode.

        terminated = self.game_over
        truncated = False # Not using truncation based on time limit

        return self._get_observation(), reward, terminated, truncated, self._get_info()

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
            "hunger": self.hunger,
            "hydration": self.hydration,
            "victory": self.victory,
        }

    def _render_game(self):
        # Draw walls
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] == 1:
                    rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                    pygame.draw.rect(self.screen, self.COLOR_WALL_OUTLINE, rect, 2)
        
        # Draw items
        for pos in self.food_items:
            self._draw_item(pos, self.COLOR_FOOD)
        for pos in self.water_items:
            self._draw_item(pos, self.COLOR_WATER)
            
        # Draw escape pod
        pod_center = (int(self.escape_pod_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2),
                      int(self.escape_pod_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2))
        self._draw_glow_circle(self.screen, self.COLOR_POD, pod_center, int(self.TILE_SIZE * 0.4), 15)

        # Draw robots
        for robot in self.robots:
            robot_center = (int(robot["pos"][0] * self.TILE_SIZE + self.TILE_SIZE / 2),
                            int(robot["pos"][1] * self.TILE_SIZE + self.TILE_SIZE / 2))
            self._draw_glow_circle(self.screen, self.COLOR_ROBOT, robot_center, int(self.TILE_SIZE * 0.35), 15)
            # Draw detection radius
            pygame.gfxdraw.filled_circle(self.screen, robot_center[0], robot_center[1], int(self.TILE_SIZE * 1.5), (*self.COLOR_ROBOT, 20))

        # Draw player
        player_center = (int(self.player_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2),
                         int(self.player_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2))
        self._draw_glow_circle(self.screen, self.COLOR_PLAYER, player_center, int(self.TILE_SIZE * 0.4), 20)

        # Draw particles
        self._update_and_draw_particles()

    def _draw_item(self, pos, color):
        item_rect = pygame.Rect(pos[0] * self.TILE_SIZE + self.TILE_SIZE*0.25, 
                                pos[1] * self.TILE_SIZE + self.TILE_SIZE*0.25,
                                self.TILE_SIZE*0.5, self.TILE_SIZE*0.5)
        pygame.draw.rect(self.screen, color, item_rect, border_radius=3)
        pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), item_rect, 2, border_radius=3)

    def _draw_glow_circle(self, surface, color, center, radius, glow_strength):
        """Draws a circle with a soft glow effect."""
        for i in range(glow_strength, 0, -1):
            alpha = int(150 * (1 - (i / glow_strength))**2)
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius + i), glow_color)
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), radius, color)
        pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), radius, color)

    def _create_gravity_particles(self):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            self.particles.append({
                "pos": [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(20, 41),
                "color": random.choice([(200, 255, 255), (150, 200, 255)])
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p["life"] / 40))
                pygame.draw.circle(self.screen, (*p["color"], alpha), p["pos"], p["life"] / 10)

    def _render_ui(self):
        # Resource Bars
        bar_width = 150
        bar_height = 15
        
        # Hunger
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, bar_width, bar_height))
        hunger_width = int(bar_width * (self.hunger / self.INITIAL_HUNGER))
        pygame.draw.rect(self.screen, self.COLOR_FOOD, (10, 10, hunger_width, bar_height))
        hunger_text = self.font_ui.render("FOOD", True, self.COLOR_UI_TEXT)
        self.screen.blit(hunger_text, (15 + bar_width, 8))

        # Hydration
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 30, bar_width, bar_height))
        hydration_width = int(bar_width * (self.hydration / self.INITIAL_HYDRATION))
        pygame.draw.rect(self.screen, self.COLOR_WATER, (10, 30, hydration_width, bar_height))
        hydration_text = self.font_ui.render("WATER", True, self.COLOR_UI_TEXT)
        self.screen.blit(hydration_text, (15 + bar_width, 28))

        # Score and Steps
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 30))

        # Gravity Indicator
        arrow_center_x = self.SCREEN_WIDTH / 2
        if self.gravity_dir.y > 0: # Down
            points = [(arrow_center_x - 10, 15), (arrow_center_x + 10, 15), (arrow_center_x, 30)]
        else: # Up
            points = [(arrow_center_x - 10, 30), (arrow_center_x + 10, 30), (arrow_center_x, 15)]
        color = self.COLOR_UI_TEXT if self.gravity_flip_cooldown_timer == 0 else self.COLOR_ROBOT
        pygame.draw.polygon(self.screen, color, points)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "ESCAPE SUCCESSFUL" if self.victory else "MISSION FAILED"
            color = self.COLOR_PLAYER if self.victory else self.COLOR_ROBOT
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Need to reset to generate a valid observation
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To play, you need to remove the SDL_VIDEODRIVER dummy setting
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gravity Shift")
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    while running:
        # --- Human Input Handling ---
        movement = 0 # no-op
        space_press = 0
        shift_press = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_press = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_press = 1
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            terminated = False

        action = [movement, space_press, shift_press]
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Control game speed for human playability
        
    env.close()