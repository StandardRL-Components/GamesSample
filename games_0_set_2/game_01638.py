
# Generated: 2025-08-27T17:46:49.519608
# Source Brief: brief_01638.md
# Brief Index: 1638

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move shape, ↑↓ to rotate. Hold Space to drop faster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide falling geometric shapes through matching openings. You have a limited number of moves and time to complete the puzzle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1800 # 60 seconds * 30 FPS
        self.MAX_MOVES = 30
        self.NUM_SHAPES_TO_PLACE = 15

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (60, 70, 90)
        self.COLOR_WALL_HIGHLIGHT = (120, 140, 180)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_TIME_BAR = (70, 180, 255)
        self.COLOR_TIME_BAR_BG = (40, 50, 70)

        # Game physics constants
        self.FALL_SPEED_NORMAL = 1.5
        self.FALL_SPEED_FAST = 8.0
        self.MOVE_SPEED = 4.0
        self.ROTATION_SPEED = 0.08 # Radians per frame

        # Define shapes
        self._define_shapes()
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.time_remaining = 0
        self.successful_placements = 0
        self.level_openings = []
        self.placed_shapes = []
        self.current_shape = None
        self.target_opening_index = -1

        self.reset()
        self.validate_implementation()

    def _define_shapes(self):
        """Defines the geometric shapes used in the game."""
        s = 30 # Scale factor
        self.SHAPE_DEFINITIONS = {
            0: { # T-shape
                "points": [(-1, -1), (1, -1), (1, 0), (-1, 0), (-0.5, 0), (-0.5, 1), (0.5, 1), (0.5, 0)],
                "color": (255, 80, 80)
            },
            1: { # L-shape
                "points": [(-1, -1), (0, -1), (0, 1), (-1, 1)],
                "color": (80, 255, 80)
            },
            2: { # Square
                "points": [(-1, -1), (1, -1), (1, 1), (-1, 1)],
                "color": (80, 150, 255)
            },
            3: { # I-shape (vertical)
                "points": [(-0.5, -1.5), (0.5, -1.5), (0.5, 1.5), (-0.5, 1.5)],
                "color": (255, 255, 80)
            },
            4: { # S-shape
                "points": [(-1, 0), (1, 0), (1, 1), (0, 1), (0, -1), (-1, -1)],
                "color": (255, 120, 255)
            }
        }
        # Scale all points
        for i in self.SHAPE_DEFINITIONS:
            self.SHAPE_DEFINITIONS[i]["points"] = [(p[0] * s/2, p[1] * s/2) for p in self.SHAPE_DEFINITIONS[i]["points"]]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = self.MAX_MOVES
        self.time_remaining = self.MAX_STEPS
        self.successful_placements = 0
        
        self.placed_shapes = []
        self.level_openings = self._generate_level()
        self.target_opening_index = 0
        self._spawn_new_shape()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Generates the sequence of openings for the level."""
        openings = []
        opening_y_step = (self.HEIGHT - 100) / self.NUM_SHAPES_TO_PLACE
        for i in range(self.NUM_SHAPES_TO_PLACE):
            shape_type = self.np_random.integers(0, len(self.SHAPE_DEFINITIONS))
            openings.append({
                "y": 80 + i * opening_y_step,
                "width": 100,
                "shape_type": shape_type,
                "color": self.SHAPE_DEFINITIONS[shape_type]["color"],
                "x": self.np_random.uniform(100, self.WIDTH - 100)
            })
        return openings

    def _spawn_new_shape(self):
        """Spawns the next shape in the sequence."""
        if self.target_opening_index >= self.NUM_SHAPES_TO_PLACE:
            self.current_shape = None
            return

        opening = self.level_openings[self.target_opening_index]
        self.current_shape = {
            "x": self.WIDTH / 2,
            "y": -40,
            "rot": 0.0,
            "shape_type": opening["shape_type"],
            "color": opening["color"],
        }

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        if not self.game_over:
            self.steps += 1
            self.time_remaining -= 1

            if self.current_shape:
                old_dist_to_target_x = abs(self.current_shape["x"] - self.level_openings[self.target_opening_index]["x"])

                # Handle player input
                if movement == 1:  # Rotate Right (CW)
                    self.current_shape["rot"] += self.ROTATION_SPEED
                elif movement == 2:  # Rotate Left (CCW)
                    self.current_shape["rot"] -= self.ROTATION_SPEED
                elif movement == 3:  # Move Left
                    self.current_shape["x"] -= self.MOVE_SPEED
                elif movement == 4:  # Move Right
                    self.current_shape["x"] += self.MOVE_SPEED

                # Clamp position
                self.current_shape["x"] = np.clip(self.current_shape["x"], 0, self.WIDTH)
                self.current_shape["rot"] %= (2 * math.pi)

                # Movement reward
                new_dist_to_target_x = abs(self.current_shape["x"] - self.level_openings[self.target_opening_index]["x"])
                if new_dist_to_target_x < old_dist_to_target_x:
                    reward += 0.01
                elif new_dist_to_target_x > old_dist_to_target_x:
                    reward -= 0.01

                # Apply gravity
                fall_speed = self.FALL_SPEED_FAST if space_held else self.FALL_SPEED_NORMAL
                self.current_shape["y"] += fall_speed

                # Check for collision/resolution
                reward += self._check_collision_and_resolve()

        terminated = self._check_termination()
        if terminated and self.successful_placements == self.NUM_SHAPES_TO_PLACE:
            reward += 50.0
            self.score += 50
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_collision_and_resolve(self):
        """Checks if the current shape has reached its target opening and resolves the outcome."""
        if self.current_shape is None:
            return 0
        
        target_opening = self.level_openings[self.target_opening_index]
        
        # Check if shape has reached the opening's y-level
        if self.current_shape["y"] >= target_opening["y"]:
            # Check for a successful fit
            x_diff = abs(self.current_shape["x"] - target_opening["x"])
            rot_diff = abs((self.current_shape["rot"] % math.pi) - 0) # Openings are not rotated
            
            # Simple "gamey" check for fit
            if x_diff < 15 and rot_diff < 0.2:
                # SUCCESS
                self.successful_placements += 1
                self.score += 10
                self.target_opening_index += 1
                self._spawn_new_shape()
                # sound: success_chime.wav
                return 10.0
            else:
                # FAILURE
                self.moves_remaining -= 1
                self.placed_shapes.append(self.current_shape) # Add to list of failed shapes
                self.target_opening_index += 1
                self._spawn_new_shape()
                # sound: thud.wav
                return -1.0
        
        # Check for collision with bottom of screen (always a failure)
        if self.current_shape["y"] > self.HEIGHT + 40:
            self.moves_remaining -= 1
            self.target_opening_index += 1
            self._spawn_new_shape()
            return -1.0

        return 0.0

    def _check_termination(self):
        if self.game_over:
            return True
        if self.successful_placements == self.NUM_SHAPES_TO_PLACE:
            self.game_over = True
            return True
        if self.moves_remaining <= 0 or self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
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
            "successful_placements": self.successful_placements,
        }

    def _render_game(self):
        # Render failed shapes (desaturated)
        for shape in self.placed_shapes:
            color = tuple(int(c * 0.3) for c in shape["color"])
            points = self._transform_shape_points(shape)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Render walls and openings
        for i, opening in enumerate(self.level_openings):
            is_target = (i == self.target_opening_index)
            wall_color = self.COLOR_WALL_HIGHLIGHT if is_target else self.COLOR_WALL
            
            # Left wall segment
            left_rect = pygame.Rect(0, opening["y"] - 2, opening["x"] - opening["width"]/2, 4)
            pygame.draw.rect(self.screen, wall_color, left_rect)
            
            # Right wall segment
            right_rect = pygame.Rect(opening["x"] + opening["width"]/2, opening["y"] - 2, self.WIDTH, 4)
            pygame.draw.rect(self.screen, wall_color, right_rect)
            
            # Highlight target opening
            if is_target:
                highlight_rect = pygame.Rect(opening["x"] - opening["width"]/2, opening["y"] - 5, opening["width"], 10)
                pygame.draw.rect(self.screen, (*self.COLOR_WALL_HIGHLIGHT, 50), highlight_rect, border_radius=3)


        # Render current shape
        if self.current_shape:
            self._draw_polygon_glow(
                self.screen,
                self._transform_shape_points(self.current_shape),
                self.current_shape["color"],
                15, 10
            )
            pygame.gfxdraw.filled_polygon(self.screen, self._transform_shape_points(self.current_shape), self.current_shape["color"])
            pygame.gfxdraw.aapolygon(self.screen, self._transform_shape_points(self.current_shape), (255,255,255))

    def _render_ui(self):
        # Render Moves Remaining
        moves_text = self.font_small.render(f"MOVES: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Render Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(centerx=self.WIDTH/2, y=10)
        self.screen.blit(score_text, score_rect)

        # Render Time Bar
        time_bar_width = 200
        time_ratio = self.time_remaining / self.MAX_STEPS
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR_BG, (self.WIDTH - time_bar_width - 10, 10, time_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR, (self.WIDTH - time_bar_width - 10, 10, time_bar_width * time_ratio, 20))
        
        # Render Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.successful_placements == self.NUM_SHAPES_TO_PLACE:
                msg = "LEVEL COMPLETE"
                color = (150, 255, 150)
            else:
                msg = "GAME OVER"
                color = (255, 150, 150)
                
            text = self.font_large.render(msg, True, color)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _transform_shape_points(self, shape):
        """Applies rotation and translation to a shape's base points."""
        points = self.SHAPE_DEFINITIONS[shape["shape_type"]]["points"]
        angle = shape["rot"]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        transformed_points = []
        for x, y in points:
            new_x = (x * cos_a - y * sin_a) + shape["x"]
            new_y = (x * sin_a + y * cos_a) + shape["y"]
            transformed_points.append((int(new_x), int(new_y)))
        return transformed_points

    def _draw_polygon_glow(self, surface, points, color, radius, alpha_decay):
        """Draws a glowing effect around a polygon."""
        if not points: return
        
        # Create a padded surface to draw the glow on
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        
        width = max_x - min_x + 2 * radius
        height = max_y - min_y + 2 * radius
        
        if width <= 0 or height <= 0: return
        
        glow_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Shift points to be local to the glow surface
        local_points = [(p[0] - min_x + radius, p[1] - min_y + radius) for p in points]
        
        for i in range(radius, 0, -1):
            alpha = int(255 * (i / radius)**2 * (alpha_decay / 100))
            if alpha > 255: alpha = 255
            
            # Create slightly larger polygons for the glow
            glow_points = []
            for px, py in local_points:
                # Simple expansion from center - not perfect but fast
                dx, dy = px - width/2, py - height/2
                norm = math.hypot(dx, dy)
                if norm == 0:
                    glow_points.append((px, py))
                    continue
                glow_points.append((px + dx/norm * i*0.5, py + dy/norm * i*0.5))

            if len(glow_points) >= 3:
                try:
                    pygame.gfxdraw.aapolygon(glow_surf, glow_points, (*color, alpha))
                except (ValueError, TypeError):
                    pass # Ignore errors from malformed polygons during rapid movement

        surface.blit(glow_surf, (min_x - radius, min_y - radius))

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a display for manual playing
    pygame.display.set_caption("Geometric Shape Fitter")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Blit the observation from the env to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    pygame.quit()