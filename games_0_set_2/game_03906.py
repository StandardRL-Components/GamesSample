
# Generated: 2025-08-28T00:47:41.051326
# Source Brief: brief_03906.md
# Brief Index: 3906

        
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
        "Controls: Arrows to move cursor. Space to place a crystal. Shift to cycle crystal type."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a laser through a crystalline cavern by placing reflector crystals to guide the beam to the exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 20
        self.GRID_HEIGHT = 15
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_WALL = (50, 55, 70)
        self.COLOR_WALL_TOP = (80, 85, 100)
        self.COLOR_EXIT = (0, 255, 150)
        self.COLOR_LASER = (255, 20, 50)
        self.COLOR_LASER_GLOW = (180, 0, 30)
        self.COLOR_CURSOR = (220, 220, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.CRYSTAL_COLORS = [
            ((80, 150, 255), (60, 120, 220)), # Blue: /
            ((255, 220, 80), (220, 180, 60)), # Yellow: \
        ]

        # Isometric projection parameters
        self.TILE_WIDTH_HALF = 16
        self.TILE_HEIGHT_HALF = 8
        self.ISO_ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ISO_ORIGIN_Y = 80
        
        # Game parameters
        self.MAX_STEPS = 200
        self.INITIAL_CRYSTALS = 15

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.walls = set()
        self.laser_origin = (0, 0)
        self.laser_initial_dir = (0, 0)
        self.exit_pos = (0, 0)
        self.crystals = {}
        self.crystal_inventory = []
        self.selected_crystal_type = 0
        self.laser_path = []
        self.laser_particles = []
        self.last_shift_press = False
        self.last_space_press = False
        self.outcome_message = ""
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.outcome_message = ""
        
        self._generate_level()
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.crystals = {}
        self.crystal_inventory = [self.INITIAL_CRYSTALS] * len(self.CRYSTAL_COLORS)
        self.selected_crystal_type = 0
        self.laser_particles = []
        self.last_shift_press = False
        self.last_space_press = False
        
        self._update_laser_path()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.walls = set()
        # Border walls
        for i in range(-1, self.GRID_WIDTH + 1):
            self.walls.add((i, -1))
            self.walls.add((i, self.GRID_HEIGHT))
        for i in range(-1, self.GRID_HEIGHT + 1):
            self.walls.add((-1, i))
            self.walls.add((self.GRID_WIDTH, i))

        # Internal walls
        for _ in range(int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.1)):
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            self.walls.add(pos)

        # Find valid start/exit points
        valid_points = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (c, r) not in self.walls:
                    valid_points.append((c, r))
        
        self.np_random.shuffle(valid_points)
        
        self.laser_origin = valid_points.pop()
        self.exit_pos = valid_points.pop()

        # Ensure laser doesn't start facing a wall
        possible_dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        self.np_random.shuffle(possible_dirs)
        for d in possible_dirs:
            next_pos = (self.laser_origin[0] + d[0], self.laser_origin[1] + d[1])
            if next_pos not in self.walls:
                self.laser_initial_dir = d
                break
        else: # If all directions are blocked, regenerate
             self._generate_level()


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_val, shift_val = action
        space_press = space_val == 1 and not self.last_space_press
        shift_press = shift_val == 1 and not self.last_shift_press
        self.last_space_press = space_val == 1
        self.last_shift_press = shift_val == 1
        
        reward = 0
        
        # Handle actions
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        if movement == 2: self.cursor_pos[1] += 1 # Down
        if movement == 3: self.cursor_pos[0] -= 1 # Left
        if movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        if shift_press:
            # sfx: cycle_weapon
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.CRYSTAL_COLORS)
        
        if space_press:
            cursor_tuple = tuple(self.cursor_pos)
            if (cursor_tuple not in self.crystals and 
                cursor_tuple not in self.walls and
                cursor_tuple != self.laser_origin and
                cursor_tuple != self.exit_pos and
                self.crystal_inventory[self.selected_crystal_type] > 0):
                
                # sfx: place_crystal
                self.crystals[cursor_tuple] = self.selected_crystal_type
                self.crystal_inventory[self.selected_crystal_type] -= 1
                reward -= 1.0

        self.steps += 1
        
        # Update laser and check for game end
        path_reward, terminated, outcome = self._update_laser_path()
        reward += path_reward
        
        if terminated:
            self.game_over = True
            if outcome == "exit":
                # sfx: win_game
                reward += 100
                self.outcome_message = "SUCCESS!"
            elif outcome == "wall":
                # sfx: fail_game
                reward -= 100
                self.outcome_message = "LASER DESTROYED"
        
        if sum(self.crystal_inventory) == 0 and not (terminated and outcome == "exit"):
            if not self.game_over:
                # sfx: out_of_resources
                reward -= 50
                self.outcome_message = "OUT OF CRYSTALS"
            self.game_over = True
            terminated = True
            
        if self.steps >= self.MAX_STEPS:
            if not self.game_over:
                self.outcome_message = "STEP LIMIT REACHED"
            self.game_over = True
            terminated = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _update_laser_path(self):
        self.laser_path = []
        pos = list(self.laser_origin)
        direction = self.laser_initial_dir
        path_reward = 0
        
        for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT): # Max laser segments
            segment_start = tuple(pos)
            
            # Trace segment
            while True:
                pos[0] += direction[0]
                pos[1] += direction[1]
                
                current_pos = tuple(pos)
                path_reward += 0.1

                if current_pos in self.walls:
                    self.laser_path.append((segment_start, current_pos))
                    return path_reward, True, "wall"
                
                if current_pos == self.exit_pos:
                    self.laser_path.append((segment_start, current_pos))
                    return path_reward, True, "exit"
                
                if current_pos in self.crystals:
                    self.laser_path.append((segment_start, current_pos))
                    crystal_type = self.crystals[current_pos]
                    
                    # Reflect direction
                    # Type 0: /
                    if crystal_type == 0:
                        direction = (-direction[1], -direction[0])
                    # Type 1: \
                    elif crystal_type == 1:
                        direction = (direction[1], direction[0])
                    break # Start new segment
        
        return path_reward, False, "traveling"

    def _iso_to_screen(self, r, c):
        x = self.ISO_ORIGIN_X + (c - r) * self.TILE_WIDTH_HALF
        y = self.ISO_ORIGIN_Y + (c + r) * self.TILE_HEIGHT_HALF
        return int(x), int(y)

    def _draw_iso_cube(self, surface, r, c, color_main, color_top, height=2):
        x, y = self._iso_to_screen(r, c)
        
        h_half, w_half = self.TILE_HEIGHT_HALF, self.TILE_WIDTH_HALF
        
        points_top = [
            (x, y),
            (x + w_half, y + h_half),
            (x, y + 2 * h_half),
            (x - w_half, y + h_half)
        ]
        
        color_side1 = tuple(int(i * 0.7) for i in color_main)
        color_side2 = tuple(int(i * 0.5) for i in color_main)
        
        # Draw sides first for correct layering
        pygame.gfxdraw.filled_polygon(surface, [
            (points_top[3][0], points_top[3][1] + height * 2 * h_half),
            (points_top[2][0], points_top[2][1] + height * 2 * h_half),
            points_top[2], points_top[3]
        ], color_side1)
        pygame.gfxdraw.filled_polygon(surface, [
            (points_top[2][0], points_top[2][1] + height * 2 * h_half),
            (points_top[1][0], points_top[1][1] + height * 2 * h_half),
            points_top[1], points_top[2]
        ], color_side2)

        # Draw top
        pygame.gfxdraw.filled_polygon(surface, points_top, color_top)
        pygame.gfxdraw.aapolygon(surface, points_top, color_top)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid floor
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                x, y = self._iso_to_screen(r, c)
                points = [
                    (x, y), (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
                    (x, y + 2 * self.TILE_HEIGHT_HALF), (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
                ]
                if (c, r) not in self.walls:
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw walls
        for c, r in self.walls:
            if 0 <= c < self.GRID_WIDTH and 0 <= r < self.GRID_HEIGHT:
                self._draw_iso_cube(self.screen, r, c, self.COLOR_WALL, self.COLOR_WALL_TOP)
        
        # Draw exit
        ex, ey = self.exit_pos
        self._draw_iso_cube(self.screen, ey, ex, self.COLOR_EXIT, self.COLOR_EXIT, height=0.5)

        # Draw crystals
        for (c, r), type_id in self.crystals.items():
            color_main, color_top = self.CRYSTAL_COLORS[type_id]
            self._draw_iso_cube(self.screen, r, c, color_main, color_top, height=1)
            
            # Draw symbol on top
            x, y = self._iso_to_screen(r, c)
            y += self.TILE_HEIGHT_HALF
            if type_id == 0: # /
                pygame.draw.aaline(self.screen, (255,255,255), (x - 5, y + 5), (x + 5, y - 5), 2)
            else: # \
                pygame.draw.aaline(self.screen, (255,255,255), (x - 5, y - 5), (x + 5, y + 5), 2)

        # Draw laser path
        for start, end in self.laser_path:
            start_screen = self._iso_to_screen(start[1], start[0])
            end_screen = self._iso_to_screen(end[1], end[0])
            start_screen = (start_screen[0], start_screen[1] + self.TILE_HEIGHT_HALF)
            end_screen = (end_screen[0], end_screen[1] + self.TILE_HEIGHT_HALF)
            
            pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, start_screen, end_screen, 6)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_screen, end_screen, 2)
            
            # Add particles
            if self.np_random.random() < 0.8:
                self.laser_particles.append([list(start_screen), [(self.np_random.random()-0.5)*2, (self.np_random.random()-0.5)*2], 20])
        
        # Update and draw particles
        for p in self.laser_particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
            radius = p[2] // 4
            if radius > 0:
                pygame.draw.circle(self.screen, self.COLOR_LASER, p[0], radius)
        self.laser_particles = [p for p in self.laser_particles if p[2] > 0]

        # Draw laser origin
        ox, oy = self.laser_origin
        origin_screen_pos = self._iso_to_screen(oy, ox)
        pygame.draw.circle(self.screen, self.COLOR_LASER, (origin_screen_pos[0], origin_screen_pos[1] + self.TILE_HEIGHT_HALF), 8)
        pygame.draw.circle(self.screen, self.COLOR_BG, (origin_screen_pos[0], origin_screen_pos[1] + self.TILE_HEIGHT_HALF), 5)


        # Draw cursor
        if not self.game_over:
            cx, cy = self.cursor_pos
            x, y = self._iso_to_screen(cy, cx)
            points = [
                (x, y), (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
                (x, y + 2 * self.TILE_HEIGHT_HALF), (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
            ]
            pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, points, 2)


    def _render_ui(self):
        # Crystal inventory display
        ui_y = 15
        for i, (count, colors) in enumerate(zip(self.crystal_inventory, self.CRYSTAL_COLORS)):
            base_x = 20 + i * 120
            
            # Draw a small representation of the crystal
            rect = pygame.Rect(base_x, ui_y, 30, 30)
            pygame.draw.rect(self.screen, colors[1], rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 2)
            
            # Draw symbol
            center_x, center_y = rect.center
            if i == 0: # /
                 pygame.draw.aaline(self.screen, (255,255,255), (center_x - 7, center_y + 7), (center_x + 7, center_y - 7), 2)
            else: # \
                 pygame.draw.aaline(self.screen, (255,255,255), (center_x - 7, center_y - 7), (center_x + 7, center_y + 7), 2)

            # Draw count
            text = self.font_large.render(f"x {count}", True, self.COLOR_TEXT)
            self.screen.blit(text, (base_x + 35, ui_y + 2))
            
            # Highlight selected
            if i == self.selected_crystal_type and not self.game_over:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (base_x - 4, ui_y - 4, rect.width + 8, rect.height + 8), 2)

        # Score and Steps
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 15, 15))
        
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 15, 35))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.outcome_message, True, self.COLOR_EXIT if self.outcome_message == "SUCCESS!" else self.COLOR_LASER)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "crystal_inventory": self.crystal_inventory,
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
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # To run the game with manual controls:
    # Use arrow keys for movement, space to place, left shift to cycle.
    # The window will not be rendered directly, but we can save frames.
    
    # Simple keyboard mapping
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # For interactive play, you would need a rendering loop.
    # This example demonstrates the API for an agent.
    print("Starting random agent test...")
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        if terminated:
            print("Episode finished.")
            obs, info = env.reset()
    
    env.close()
    print("Random agent test complete.")