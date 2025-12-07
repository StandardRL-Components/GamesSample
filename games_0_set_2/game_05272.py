
# Generated: 2025-08-28T04:29:55.598767
# Source Brief: brief_05272.md
# Brief Index: 5272

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Data structure for objects in the scene
SceneObject = namedtuple("SceneObject", ["pos", "height", "color", "type", "found"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the selector. Space to select the highlighted tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric hidden object game. Find all 20 hidden spheres among the clutter before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 18
    GRID_HEIGHT = 12
    TILE_WIDTH_HALF = 20
    TILE_HEIGHT_HALF = 10
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 80

    TOTAL_HIDDEN_OBJECTS = 20
    MAX_TIME = 600
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (40, 45, 55)
    COLOR_CURSOR = (255, 255, 0, 100) # Yellow, semi-transparent
    COLOR_TEXT = (255, 255, 255)
    COLOR_FLASH_CORRECT = (0, 255, 0, 80)
    COLOR_FLASH_INCORRECT = (255, 0, 0, 80)
    
    # Distractor block colors
    DISTR_COLORS = [
        (80, 80, 90), (90, 90, 100), (70, 70, 80)
    ]
    # Hidden object color
    HIDDEN_COLOR = (255, 80, 80) # Bright Red

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
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        self.game_objects = []
        self.cursor_pos = [0, 0]
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.objects_found = 0
        self.feedback_flash = {"color": None, "duration": 0}
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_left = self.MAX_TIME
        self.objects_found = 0
        self.game_over = False
        self.feedback_flash = {"color": None, "duration": 0}
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self._generate_scene()

        return self._get_observation(), self._get_info()

    def _generate_scene(self):
        self.game_objects = []
        
        # Generate all possible grid positions
        all_pos = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_pos)
        
        # Place hidden objects
        for i in range(self.TOTAL_HIDDEN_OBJECTS):
            pos = all_pos.pop(0)
            self.game_objects.append(
                SceneObject(pos=pos, height=0.5, color=self.HIDDEN_COLOR, type="hidden", found=False)
            )
            
        # Place distractor objects
        num_distractors = 40 + self.np_random.integers(0, 10)
        for i in range(num_distractors):
            if not all_pos: break
            pos = all_pos.pop(0)
            height = 1 + self.np_random.random() * 3
            color = random.choice(self.DISTR_COLORS)
            self.game_objects.append(
                SceneObject(pos=pos, height=height, color=color, type="distractor", found=False)
            )
            
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_pressed, _ = action
        reward = -0.1  # Time penalty

        # 1. Update game logic
        self.steps += 1
        self.time_left -= 1
        
        if self.feedback_flash["duration"] > 0:
            self.feedback_flash["duration"] -= 1

        # Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Handle selection
        if space_pressed:
            found_item = False
            for i, obj in enumerate(self.game_objects):
                if obj.type == "hidden" and not obj.found and tuple(self.cursor_pos) == obj.pos:
                    # Correct selection
                    self.game_objects[i] = obj._replace(found=True)
                    self.objects_found += 1
                    self.score += 10
                    reward += 10
                    self.feedback_flash = {"color": self.COLOR_FLASH_CORRECT, "duration": 5}
                    found_item = True
                    # SFX: Correct find
                    break
            
            if not found_item:
                # Incorrect selection
                self.score -= 1
                reward -= 1
                self.feedback_flash = {"color": self.COLOR_FLASH_INCORRECT, "duration": 5}
                # SFX: Incorrect find

        # 2. Check for termination
        terminated = False
        if self.objects_found == self.TOTAL_HIDDEN_OBJECTS:
            self.score += 50
            reward += 50
            terminated = True
            self.game_over = True
        elif self.time_left <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _iso_to_screen(self, x, y, z=0):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF - (z * self.TILE_HEIGHT_HALF * 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, pos, height, color):
        x, y = pos
        top_color = color
        left_color = tuple(max(0, c - 20) for c in color)
        right_color = tuple(max(0, c - 40) for c in color)

        # Points for the top face
        p1 = self._iso_to_screen(x, y, height)
        p2 = self._iso_to_screen(x + 1, y, height)
        p3 = self._iso_to_screen(x + 1, y + 1, height)
        p4 = self._iso_to_screen(x, y + 1, height)

        # Draw sides first
        if height > 0:
            # Right side
            pr1 = self._iso_to_screen(x + 1, y, 0)
            pr2 = self._iso_to_screen(x + 1, y + 1, 0)
            pygame.gfxdraw.filled_polygon(surface, [p2, p3, pr2, pr1], right_color)
            pygame.gfxdraw.aapolygon(surface, [p2, p3, pr2, pr1], right_color)

            # Left side
            pl1 = self._iso_to_screen(x, y + 1, 0)
            pygame.gfxdraw.filled_polygon(surface, [p4, p3, pr2, pl1], left_color)
            pygame.gfxdraw.aapolygon(surface, [p4, p3, pr2, pl1], left_color)

        # Draw top face
        pygame.gfxdraw.filled_polygon(surface, [p1, p2, p3, p4], top_color)
        pygame.gfxdraw.aapolygon(surface, [p1, p2, p3, p4], top_color)
        
    def _draw_iso_sphere(self, surface, pos, radius, color):
        center_x, center_y = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5, 0.5)
        pygame.gfxdraw.filled_circle(surface, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(surface, center_x, center_y, radius, color)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Sort objects for rendering (painter's algorithm)
        sorted_objects = sorted(self.game_objects, key=lambda obj: obj.pos[0] + obj.pos[1])
        
        # Draw objects and cursor tile
        cursor_drawn = False
        for obj in sorted_objects:
            # Check if we should draw the cursor at this depth
            if not cursor_drawn and (self.cursor_pos[0] + self.cursor_pos[1] < obj.pos[0] + obj.pos[1]):
                self._draw_cursor()
                cursor_drawn = True

            if obj.type == "distractor":
                self._draw_iso_cube(self.screen, obj.pos, obj.height, obj.color)
            elif obj.type == "hidden" and not obj.found:
                self._draw_iso_sphere(self.screen, obj.pos, 6, obj.color)

        # If cursor is on the last row, it wasn't drawn yet
        if not cursor_drawn:
            self._draw_cursor()
            
        # Draw feedback flash
        if self.feedback_flash["duration"] > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.feedback_flash["color"])
            self.screen.blit(flash_surface, (0, 0))

    def _draw_cursor(self):
        cx, cy = self.cursor_pos
        p1 = self._iso_to_screen(cx, cy)
        p2 = self._iso_to_screen(cx + 1, cy)
        p3 = self._iso_to_screen(cx + 1, cy + 1)
        p4 = self._iso_to_screen(cx, cy + 1)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], self.COLOR_CURSOR)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_text = self.font_main.render(f"TIME: {self.time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Objects Found
        found_text = self.font_small.render(f"FOUND: {self.objects_found}/{self.TOTAL_HIDDEN_OBJECTS}", True, self.COLOR_TEXT)
        self.screen.blit(found_text, (self.SCREEN_WIDTH - found_text.get_width() - 10, 35))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.objects_found == self.TOTAL_HIDDEN_OBJECTS else "TIME'S UP!"
            end_text = self.font_main.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "objects_found": self.objects_found,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert len(self.game_objects) > self.TOTAL_HIDDEN_OBJECTS
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play the game manually, you need to install pygame and run this script.
    # The environment will render to a window.
    
    try:
        import pygame
        from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_LSHIFT, K_ESCAPE, K_r
        
        window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Hidden Object Game")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            action = [0, 0, 0] # no-op, no-space, no-shift
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == K_ESCAPE):
                    done = True
            
            keys = pygame.key.get_pressed()
            if keys[K_UP]: action[0] = 1
            elif keys[K_DOWN]: action[0] = 2
            elif keys[K_LEFT]: action[0] = 3
            elif keys[K_RIGHT]: action[0] = 4
                
            if keys[K_SPACE]: action[1] = 1
            if keys[K_LSHIFT]: action[2] = 1
            if keys[K_r]:
                obs, info = env.reset()
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}")
                # Wait for a moment before resetting
                frame = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                window.blit(surf, (0, 0))
                pygame.display.flip()
                pygame.time.wait(3000)
                obs, info = env.reset()

            # Convert observation back to a surface for display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            window.blit(surf, (0, 0))
            
            pygame.display.flip()
            env.clock.tick(30) # Limit frame rate for manual play
            
    except ImportError:
        print("Pygame is not installed. Skipping manual play test.")
        print("To play the game, run: pip install pygame")
    
    env.close()