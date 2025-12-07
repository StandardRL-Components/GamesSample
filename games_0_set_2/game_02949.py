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


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle between element types (rocks and plants). Press Space to place the current element."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A serene isometric Zen garden puzzle. Strategically place rocks and plants to achieve visual harmony and maximize your aesthetic score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
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

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS  # 60 seconds
        self.MAX_SCORE = 1000

        # Colors
        self.COLOR_SAND = (210, 180, 140)
        self.COLOR_SAND_RAKE = (195, 165, 125)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (0, 0, 0, 128)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_TIMER_BAR = (100, 200, 100)
        self.COLOR_TIMER_BAR_BG = (80, 80, 80)

        # Fonts
        self.font_ui = pygame.font.SysFont("sans-serif", 24)
        self.font_ui_small = pygame.font.SysFont("sans-serif", 18)

        # Isometric grid settings
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 15
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = 20, 10
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # Element definitions
        self.ELEMENTS = [
            {"name": "Small Rock", "type": "rock", "color": (140, 140, 140), "base_score": 10},
            {"name": "Medium Rock", "type": "rock", "color": (120, 120, 120), "base_score": 15},
            {"name": "Tall Rock", "type": "rock", "color": (100, 100, 100), "base_score": 20},
            {"name": "Bush", "type": "plant", "color": (80, 150, 80), "base_score": 15},
            {"name": "Tree", "type": "plant", "color": (60, 120, 60), "base_score": 25},
        ]
        
        # State variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.placed_elements = []
        self.occupied_cells = set()
        self.cursor_pos = (0, 0)
        self.selected_element_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.placed_elements = []
        self.occupied_cells = set()
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_element_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        # Handle discrete presses (detect rising edge)
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Handle Input ---
        # 1. Cycle selection
        if shift_pressed:
            self.selected_element_index = (self.selected_element_index + 1) % len(self.ELEMENTS)
            # sfx: UI_CYCLE_SOUND

        # 2. Move cursor
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        new_cursor_x = max(0, min(self.GRID_WIDTH - 1, self.cursor_pos[0] + dx))
        new_cursor_y = max(0, min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + dy))
        if (new_cursor_x, new_cursor_y) != self.cursor_pos:
            self.cursor_pos = (new_cursor_x, new_cursor_y)
            # sfx: CURSOR_MOVE_SOUND
        
        # 3. Place element
        if space_pressed:
            placement_reward = self._place_element()
            reward += placement_reward

        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS
        
        if terminated and self.score >= self.MAX_SCORE:
            reward += 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _place_element(self):
        if self.cursor_pos in self.occupied_cells:
            # sfx: ACTION_FAIL_SOUND
            return 0 # Cannot place on an occupied cell

        # sfx: PLACE_ELEMENT_SOUND
        element_def = self.ELEMENTS[self.selected_element_index]
        new_element = {
            "grid_pos": self.cursor_pos,
            "def": element_def,
        }
        self.placed_elements.append(new_element)
        self.occupied_cells.add(self.cursor_pos)

        old_score = self.score
        self.score = self._calculate_aesthetic_score()
        score_change = self.score - old_score

        # Calculate reward based on score change
        reward = 0
        if score_change > 0:
            reward += 5 + (score_change / 10.0)
        elif score_change < 0:
            reward += -2 + (score_change * 0.1) # score_change is negative
        
        return reward

    def _calculate_aesthetic_score(self):
        total_score = 0
        
        # Sum of base scores
        for el in self.placed_elements:
            total_score += el["def"]["base_score"]

        # Harmony/Disharmony bonuses
        for i in range(len(self.placed_elements)):
            for j in range(i + 1, len(self.placed_elements)):
                el1 = self.placed_elements[i]
                el2 = self.placed_elements[j]
                
                dist = abs(el1["grid_pos"][0] - el2["grid_pos"][0]) + abs(el1["grid_pos"][1] - el2["grid_pos"][1])
                
                if dist <= 2:
                    type1, type2 = el1["def"]["type"], el2["def"]["type"]
                    bonus_multiplier = 2.5 if dist == 1 else 1.0

                    if type1 == "rock" and type2 == "rock":
                        total_score += 15 * bonus_multiplier
                    elif type1 == "plant" and type2 == "plant":
                        total_score -= 20 * bonus_multiplier
                    elif (type1 == "rock" and type2 == "plant") or (type1 == "plant" and type2 == "rock"):
                        total_score += 30 * bonus_multiplier
        
        return min(self.MAX_SCORE, max(0, int(total_score)))

    def _grid_to_iso(self, grid_pos):
        gx, gy = grid_pos
        sx = self.ORIGIN_X + (gx - gy) * self.TILE_WIDTH_HALF
        sy = self.ORIGIN_Y + (gx + gy) * self.TILE_HEIGHT_HALF
        return int(sx), int(sy)

    def _get_observation(self):
        self.screen.fill(self.COLOR_SAND)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw raked sand lines
        for i in range(-self.GRID_HEIGHT, self.GRID_WIDTH + self.GRID_HEIGHT):
            x1, y1 = self._grid_to_iso((i, 0))
            x2, y2 = self._grid_to_iso((i-self.GRID_HEIGHT, self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_SAND_RAKE, (x1, y1), (x2, y2), 1)

        # Create a list of all drawable objects (elements + cursor) to sort for correct Z-ordering
        drawable_objects = []
        for el in self.placed_elements:
            drawable_objects.append(("element", el))
        
        # Add cursor to be sorted
        drawable_objects.append(("cursor", {"grid_pos": self.cursor_pos}))

        # Sort objects by their grid y+x value to draw from back to front
        drawable_objects.sort(key=lambda item: sum(item[1]["grid_pos"]))

        for obj_type, obj_data in drawable_objects:
            screen_pos = self._grid_to_iso(obj_data["grid_pos"])
            if obj_type == "element":
                self._draw_iso_object(obj_data["def"], screen_pos)
            elif obj_type == "cursor":
                self._draw_cursor(screen_pos)

    def _draw_cursor(self, screen_pos):
        x, y = screen_pos
        # Draw a translucent highlight on the ground tile
        points = [
            (x, y),
            (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
            (x, y + self.TILE_HEIGHT_HALF * 2),
            (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CURSOR)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CURSOR)

    def _draw_iso_object(self, element_def, screen_pos):
        x, y = screen_pos
        color = element_def["color"]
        darker_color = tuple(max(0, c - 30) for c in color)
        
        if element_def["name"] == "Small Rock":
            points = [(x, y - 5), (x + 12, y - 12), (x, y - 19), (x - 12, y - 12)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif element_def["name"] == "Medium Rock":
            # Body
            pygame.gfxdraw.filled_polygon(self.screen, [(x, y-15), (x+15, y-22), (x, y-29), (x-15, y-22)], color)
            pygame.gfxdraw.aapolygon(self.screen, [(x, y-15), (x+15, y-22), (x, y-29), (x-15, y-22)], color)
            # Left face
            pygame.gfxdraw.filled_polygon(self.screen, [(x,y), (x, y-15), (x-15, y-22), (x-15, y-7)], darker_color)
        elif element_def["name"] == "Tall Rock":
            # Top
            pygame.gfxdraw.filled_polygon(self.screen, [(x, y-30), (x+10, y-35), (x, y-40), (x-10, y-35)], color)
            # Left face
            pygame.gfxdraw.filled_polygon(self.screen, [(x, y), (x, y-30), (x-10, y-35), (x-10, y-5)], darker_color)
            # Right face
            pygame.gfxdraw.filled_polygon(self.screen, [(x, y), (x, y-30), (x+10, y-35), (x+10, y-5)], tuple(max(0, c - 15) for c in color))
        elif element_def["name"] == "Bush":
            pygame.gfxdraw.filled_circle(self.screen, x, y - 20, 15, color)
            pygame.gfxdraw.aacircle(self.screen, x, y - 20, 15, color)
            pygame.gfxdraw.filled_circle(self.screen, x-8, y - 15, 12, darker_color)
            pygame.gfxdraw.aacircle(self.screen, x-8, y - 15, 12, darker_color)
        elif element_def["name"] == "Tree":
            # Trunk
            pygame.draw.line(self.screen, (90, 60, 30), (x, y), (x, y - 35), 4)
            # Leaves
            pygame.gfxdraw.filled_circle(self.screen, x, y - 50, 20, color)
            pygame.gfxdraw.aacircle(self.screen, x, y - 50, 20, color)
            pygame.gfxdraw.filled_circle(self.screen, x+10, y-45, 18, darker_color)
            pygame.gfxdraw.aacircle(self.screen, x+10, y-45, 18, darker_color)

    def _render_ui(self):
        # Score display
        score_surf = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Timer display
        timer_width = 200
        progress = 1.0 - (self.steps / self.MAX_STEPS)
        
        bar_rect = pygame.Rect(self.WIDTH - timer_width - 10, 10, timer_width, 25)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, bar_rect, border_radius=5)
        
        progress_rect = pygame.Rect(bar_rect.x, bar_rect.y, int(timer_width * progress), bar_rect.height)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, progress_rect, border_radius=5)
        
        # Selected element preview
        preview_bg_rect = pygame.Rect(10, self.HEIGHT - 70, 180, 60)
        s = pygame.Surface(preview_bg_rect.size, pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, preview_bg_rect.topleft)

        selected_def = self.ELEMENTS[self.selected_element_index]
        name_surf = self.font_ui_small.render(selected_def["name"], True, self.COLOR_UI_TEXT)
        self.screen.blit(name_surf, (20, self.HEIGHT - 65))
        
        preview_pos = (140, self.HEIGHT - 20)
        self._draw_iso_object(selected_def, preview_pos)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    # Re-enable video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zen Garden")
    clock = pygame.time.Clock()
    
    # Game loop
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Map keys to action space
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}")
            obs, info = env.reset()

        # Control the frame rate
        clock.tick(env.FPS)

    env.close()