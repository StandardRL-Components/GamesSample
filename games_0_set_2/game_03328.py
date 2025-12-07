
# Generated: 2025-08-27T23:01:58.686976
# Source Brief: brief_03328.md
# Brief Index: 3328

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An isometric puzzle game where the player directs a laser beam to a target
    by placing different types of crystals. The game is played in real-time
    with a 60-second timer.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Shift to cycle crystal type. Space to place a crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Redirect a laser to its target by placing crystals in a cavern within 60 seconds."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID_LINES = (40, 50, 70)
    COLOR_WALL = (80, 90, 110)
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_LASER = (255, 20, 20)
    COLOR_LASER_GLOW = (255, 50, 50, 70)
    COLOR_TARGET = (50, 255, 50)
    COLOR_OBSTACLE = (120, 40, 180)
    CRYSTAL_COLORS = [
        (60, 120, 255),  # Reflect (Blue)
        (255, 220, 60),  # Split (Yellow)
        (200, 60, 200),  # Absorb (Purple)
    ]
    CRYSTAL_NAMES = ["REFLECT", "SPLIT", "ABSORB"]
    
    # Game parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    GRID_WIDTH, GRID_HEIGHT = 18, 12
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 24, 12
    MAX_LASER_BOUNCES = 25
    MAX_STEPS = 1000

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
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 30, bold=True)

        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.timer = None
        self.grid = None
        self.laser_source_pos = None
        self.laser_source_dir = None
        self.target_pos = None
        self.obstacles = None
        self.placed_crystals = None
        self.laser_path = None
        self.cursor_pos = None
        self.selected_crystal_type = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.cursor_cooldown = None
        self.shift_cooldown = None
        
        self.reset()
        
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.timer = self.GAME_DURATION_SECONDS * self.FPS
        
        self._generate_level()

        self.laser_path = []
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_crystal_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.cursor_cooldown = 0
        self.shift_cooldown = 0

        self._calculate_laser_path()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.placed_crystals = {}
        
        # Place laser source on a wall
        side = self.np_random.integers(4)
        if side == 0: # Top
            self.laser_source_pos = (self.np_random.integers(1, self.GRID_WIDTH-1), -1)
            self.laser_source_dir = (0, 1)
        elif side == 1: # Bottom
            self.laser_source_pos = (self.np_random.integers(1, self.GRID_WIDTH-1), self.GRID_HEIGHT)
            self.laser_source_dir = (0, -1)
        elif side == 2: # Left
            self.laser_source_pos = (-1, self.np_random.integers(1, self.GRID_HEIGHT-1))
            self.laser_source_dir = (1, 0)
        else: # Right
            self.laser_source_pos = (self.GRID_WIDTH, self.np_random.integers(1, self.GRID_HEIGHT-1))
            self.laser_source_dir = (-1, 0)

        # Place target on a different wall
        while True:
            target_side = self.np_random.integers(4)
            if target_side == 0:
                self.target_pos = (self.np_random.integers(1, self.GRID_WIDTH-1), -1)
            elif target_side == 1:
                self.target_pos = (self.np_random.integers(1, self.GRID_WIDTH-1), self.GRID_HEIGHT)
            elif target_side == 2:
                self.target_pos = (-1, self.np_random.integers(1, self.GRID_HEIGHT-1))
            else:
                self.target_pos = (self.GRID_WIDTH, self.np_random.integers(1, self.GRID_HEIGHT-1))
            if self.target_pos != self.laser_source_pos:
                break
        
        # Place obstacles
        self.obstacles = []
        num_obstacles = self.np_random.integers(2, 5)
        for _ in range(num_obstacles):
            while True:
                pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
                if pos != self.target_pos and pos != self.laser_source_pos and pos not in self.obstacles:
                    self.obstacles.append(pos)
                    break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        reward = -0.01  # Cost of time

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        crystal_placed_this_step = self._place_crystal(space_held)
        
        if crystal_placed_this_step:
            # Recalculate laser path only when state changes
            interaction_reward = self._calculate_laser_path()
            reward += interaction_reward
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and self.game_won:
            reward += 100 # Large reward for winning
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Cooldowns to prevent rapid-fire actions
        if self.cursor_cooldown > 0: self.cursor_cooldown -= 1
        if self.shift_cooldown > 0: self.shift_cooldown -= 1

        # Move cursor
        if movement != 0 and self.cursor_cooldown == 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1  # Right
            
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_HEIGHT - 1)
            self.cursor_cooldown = 3 # 3 frames delay

        # Cycle crystal type
        is_shift_press = shift_held and not self.prev_shift_held
        if is_shift_press and self.shift_cooldown == 0:
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.CRYSTAL_COLORS)
            self.shift_cooldown = 10 # 10 frames delay
            # sfx: crystal_select.wav
        
        self.prev_shift_held = shift_held

    def _place_crystal(self, space_held):
        is_space_press = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        cursor_tuple = tuple(self.cursor_pos)
        if is_space_press and cursor_tuple not in self.obstacles and cursor_tuple not in self.placed_crystals:
            self.placed_crystals[cursor_tuple] = self.selected_crystal_type
            # sfx: place_crystal.wav
            return True
        return False

    def _check_termination(self):
        if self.game_won:
            self.game_over = True
            return True
        if self.timer <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
    
    def _calculate_laser_path(self):
        self.laser_path = []
        self.game_won = False # Reset win state before check
        interaction_reward = 0
        
        q = [(self.laser_source_pos, self.laser_source_dir, 0)]
        
        visited_crystal_interactions = set()

        while q:
            pos, direction, bounce_count = q.pop(0)
            
            if bounce_count > self.MAX_LASER_BOUNCES:
                continue

            start_point = pos
            current_pos = list(pos)

            for _ in range(max(self.GRID_WIDTH, self.GRID_HEIGHT) * 2):
                current_pos[0] += direction[0] * 0.5
                current_pos[1] += direction[1] * 0.5
                
                grid_pos = (int(round(current_pos[0])), int(round(current_pos[1])))

                # Check for hits
                hit = False
                
                # Target Hit
                if grid_pos == self.target_pos:
                    self.game_won = True
                    self.laser_path.append((start_point, grid_pos))
                    hit = True
                    # sfx: win.wav
                    break # Stop all laser calculations on win

                # Obstacle Hit
                if grid_pos in self.obstacles:
                    self.game_over = True
                    self.laser_path.append((start_point, grid_pos))
                    hit = True
                    # sfx: lose_obstacle.wav
                    break

                # Crystal Hit
                if grid_pos in self.placed_crystals:
                    crystal_type = self.placed_crystals[grid_pos]
                    self.laser_path.append((start_point, grid_pos))
                    
                    if (grid_pos, direction) not in visited_crystal_interactions:
                        interaction_reward += 0.1
                        visited_crystal_interactions.add((grid_pos, direction))
                    
                    # Absorb
                    if self.CRYSTAL_NAMES[crystal_type] == "ABSORB":
                        # sfx: absorb.wav
                        hit = True
                    # Reflect
                    elif self.CRYSTAL_NAMES[crystal_type] == "REFLECT":
                        # sfx: reflect.wav
                        # Simple reflection based on entry direction
                        if abs(direction[0]) > 0: # Horizontal entry
                            new_dir = (-direction[0], 0)
                        else: # Vertical entry
                            new_dir = (0, -direction[1])
                        q.append((grid_pos, new_dir, bounce_count + 1))
                        hit = True
                    # Split
                    elif self.CRYSTAL_NAMES[crystal_type] == "SPLIT":
                        # sfx: split.wav
                        if abs(direction[0]) > 0: # Horizontal entry
                            q.append((grid_pos, (0, 1), bounce_count + 1))
                            q.append((grid_pos, (0, -1), bounce_count + 1))
                        else: # Vertical entry
                            q.append((grid_pos, (1, 0), bounce_count + 1))
                            q.append((grid_pos, (-1, 0), bounce_count + 1))
                        hit = True
                    
                # Wall Hit
                if not (0 <= grid_pos[0] < self.GRID_WIDTH and 0 <= grid_pos[1] < self.GRID_HEIGHT):
                    self.laser_path.append((start_point, grid_pos))
                    new_dir = list(direction)
                    if not (0 <= grid_pos[0] < self.GRID_WIDTH): new_dir[0] *= -1
                    if not (0 <= grid_pos[1] < self.GRID_HEIGHT): new_dir[1] *= -1
                    q.append((grid_pos, tuple(new_dir), bounce_count + 1))
                    hit = True

                if hit:
                    break
            
            if self.game_won or self.game_over:
                break
        
        self.score += interaction_reward
        return interaction_reward

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
            "time_left": self.timer / self.FPS,
            "selected_crystal": self.CRYSTAL_NAMES[self.selected_crystal_type]
        }

    def _iso_to_screen(self, x, y):
        screen_x = (self.SCREEN_WIDTH / 2) + (x - y) * self.TILE_WIDTH_HALF
        screen_y = 100 + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, color, x, y, size=1.0):
        hw = self.TILE_WIDTH_HALF * size
        hh = self.TILE_HEIGHT_HALF * size
        cx, cy = self._iso_to_screen(x, y)
        
        points_top = [
            (cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)
        ]
        points_left = [
            (cx - hw, cy), (cx, cy + hh), (cx, cy + hh * 2), (cx - hw, cy + hh)
        ]
        points_right = [
            (cx + hw, cy), (cx, cy + hh), (cx, cy + hh * 2), (cx + hw, cy + hh)
        ]
        
        # Darken side colors for 3D effect
        dark_color1 = tuple(max(0, c - 40) for c in color)
        dark_color2 = tuple(max(0, c - 60) for c in color)

        pygame.gfxdraw.filled_polygon(surface, points_left, dark_color2)
        pygame.gfxdraw.aapolygon(surface, points_left, dark_color2)
        pygame.gfxdraw.filled_polygon(surface, points_right, dark_color1)
        pygame.gfxdraw.aapolygon(surface, points_right, dark_color1)
        pygame.gfxdraw.filled_polygon(surface, points_top, color)
        pygame.gfxdraw.aapolygon(surface, points_top, color)

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, r)
            end = self._iso_to_screen(self.GRID_WIDTH, r)
            pygame.draw.aaline(self.screen, self.COLOR_GRID_LINES, start, end)
        for c in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(c, 0)
            end = self._iso_to_screen(c, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID_LINES, start, end)
        
        # Draw walls
        for y in range(-1, self.GRID_HEIGHT + 1):
            self._draw_iso_cube(self.screen, self.COLOR_WALL, -1, y, 0.9)
            self._draw_iso_cube(self.screen, self.COLOR_WALL, self.GRID_WIDTH, y, 0.9)
        for x in range(0, self.GRID_WIDTH):
             self._draw_iso_cube(self.screen, self.COLOR_WALL, x, -1, 0.9)
             self._draw_iso_cube(self.screen, self.COLOR_WALL, x, self.GRID_HEIGHT, 0.9)

        # Draw target and obstacles
        self._draw_iso_cube(self.screen, self.COLOR_TARGET, self.target_pos[0], self.target_pos[1], 1.0)
        for ox, oy in self.obstacles:
            self._draw_iso_cube(self.screen, self.COLOR_OBSTACLE, ox, oy, 1.0)
        
        # Draw placed crystals
        for (cx, cy), ctype in self.placed_crystals.items():
            self._draw_iso_cube(self.screen, self.CRYSTAL_COLORS[ctype], cx, cy, 0.8)

        # Draw laser path
        for start, end in self.laser_path:
            p1 = self._iso_to_screen(start[0], start[1])
            p2 = self._iso_to_screen(end[0], end[1])
            pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, p1, p2, 7)
            pygame.draw.aaline(self.screen, self.COLOR_LASER, p1, p2, 2)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_screen_pos = self._iso_to_screen(cx, cy)
        hw, hh = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        points = [(cursor_screen_pos[0], cursor_screen_pos[1] - hh), (cursor_screen_pos[0] + hw, cursor_screen_pos[1]),
                  (cursor_screen_pos[0], cursor_screen_pos[1] + hh), (cursor_screen_pos[0] - hw, cursor_screen_pos[1])]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CURSOR)

    def _render_ui(self):
        # Timer
        time_str = f"TIME: {max(0, self.timer / self.FPS):.1f}"
        time_surf = self.font_ui.render(time_str, True, (255, 255, 255))
        self.screen.blit(time_surf, (self.SCREEN_WIDTH // 2 - time_surf.get_width() // 2, 10))

        # Score
        score_str = f"SCORE: {self.score:.2f}"
        score_surf = self.font_ui.render(score_str, True, (255, 255, 255))
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))
        
        # Selected Crystal
        crystal_name = self.CRYSTAL_NAMES[self.selected_crystal_type]
        crystal_color = self.CRYSTAL_COLORS[self.selected_crystal_type]
        crystal_surf = self.font_ui.render(f"CRYSTAL: {crystal_name}", True, crystal_color)
        self.screen.blit(crystal_surf, (10, 10))

        # Game Over Message
        if self.game_over:
            msg_text = "SUCCESS!" if self.game_won else "FAILURE"
            msg_color = self.COLOR_TARGET if self.game_won else self.COLOR_LASER
            msg_surf = self.font_msg.render(msg_text, True, msg_color)
            
            # Add a dark background for readability
            bg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            bg_rect.inflate_ip(20, 20)
            pygame.draw.rect(self.screen, (0,0,0,180), bg_rect, border_radius=10)
            
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)))

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- For human play ---
    # To play, you need to display the output. This is not part of the env itself.
    
    # --- Pygame window setup for human play ---
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Laser Cavern")
    
    obs, info = env.reset()
    terminated = False
    
    # Main game loop for human interaction
    while not terminated:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        
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
        
        # Render the observation to the human-facing screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # Control the frame rate
        env.clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over. Final Score: {info['score']:.2f}")
            # Wait a moment before closing
            pygame.time.wait(2000)

    env.close()