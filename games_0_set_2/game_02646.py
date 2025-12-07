
# Generated: 2025-08-27T20:59:35.100818
# Source Brief: brief_02646.md
# Brief Index: 2646

        
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
        "Controls: ↑↓←→ to move and push crates. Solve the puzzle before the timer runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race against the clock to push crates onto target locations in this isometric Sokoban-inspired puzzle game."
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
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.FPS = 30
        self.MAX_TIME_SECONDS = 45
        self.MAX_STEPS = 1350 # 45s * 30fps

        # Grid and Isometric constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 10
        self.TILE_WIDTH_ISO = 48
        self.TILE_HEIGHT_ISO = 24
        self.CUBE_HEIGHT = 20
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_TILE = (60, 70, 80)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_SIDES = (40, 120, 200)
        self.COLOR_CRATE = (160, 110, 70)
        self.COLOR_CRATE_SIDES = (130, 90, 50)
        self.COLOR_TARGET = (80, 180, 80)
        self.COLOR_WALL = (140, 150, 160)
        self.COLOR_WALL_SIDES = (110, 120, 130)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_UI_TEXT_WARN = (255, 80, 80)
        self.COLOR_UI_BG = (40, 45, 50, 200)

        # Fonts
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Static level data
        self._define_level()
        
        # Initialize state variables
        self.player_pos = None
        self.crate_positions = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining_frames = 0
        
        self.reset()
        self.validate_implementation()
    
    def _define_level(self):
        level_layout = [
            "WWWWWWWWWWWW",
            "W..........W",
            "W.T.WW.C.T.W",
            "W...WW...C.W",
            "W....P.....W",
            "W.C........W",
            "W...WW.....W",
            "W.T.WW.....W",
            "W..........W",
            "WWWWWWWWWWWW",
        ]
        self.walls = set()
        self.initial_crate_positions = []
        self.initial_player_pos = (0, 0)
        self.targets = set()
        for r, row in enumerate(level_layout):
            for c, char in enumerate(row):
                pos = (c, r)
                if char == 'W':
                    self.walls.add(pos)
                elif char == 'P':
                    self.initial_player_pos = pos
                elif char == 'C':
                    self.initial_crate_positions.append(pos)
                elif char == 'T':
                    self.targets.add(pos)
        self.num_targets = len(self.targets)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = self.initial_player_pos
        self.crate_positions = list(self.initial_crate_positions)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining_frames = self.MAX_TIME_SECONDS * self.FPS
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.02  # Time penalty
        self.time_remaining_frames -= 1
        
        movement = action[0]
        dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]

        if dx != 0 or dy != 0:
            px, py = self.player_pos
            next_player_pos = (px + dx, py + dy)

            if next_player_pos in self.walls:
                pass # Player bumps into a wall
            elif next_player_pos in self.crate_positions:
                crate_idx = self.crate_positions.index(next_player_pos)
                crate_pos = self.crate_positions[crate_idx]
                next_crate_pos = (crate_pos[0] + dx, crate_pos[1] + dy)

                if next_crate_pos not in self.walls and next_crate_pos not in self.crate_positions:
                    # Valid push
                    # Sound: Push crate
                    old_dist = min(self._manhattan_distance(crate_pos, t) for t in self.targets) if self.targets else 0
                    new_dist = min(self._manhattan_distance(next_crate_pos, t) for t in self.targets) if self.targets else 0
                    
                    if new_dist < old_dist: reward += 0.1
                    elif new_dist > old_dist: reward -= 0.2

                    was_on_target = crate_pos in self.targets
                    is_on_target = next_crate_pos in self.targets
                    if is_on_target and not was_on_target:
                        reward += 1.0
                        # Sound: Crate lock

                    self.crate_positions[crate_idx] = next_crate_pos
                    self.player_pos = next_player_pos
            else:
                # Valid move into empty space
                # Sound: Player step
                self.player_pos = next_player_pos

        self.score += reward
        self.steps += 1
        
        # Check termination conditions
        crates_on_target = sum(1 for c in self.crate_positions if c in self.targets)
        
        win = crates_on_target == self.num_targets
        timeout = self.time_remaining_frames <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS

        terminated = win or timeout or max_steps_reached

        if terminated:
            self.game_over = True
            if win:
                reward += 50.0
                # Sound: Level complete
            elif timeout:
                reward -= 50.0
                # Sound: Time up failure
            self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
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
            "time_remaining": max(0, self.time_remaining_frames / self.FPS),
            "crates_on_target": sum(1 for c in self.crate_positions if c in self.targets)
        }

    def _render_game(self):
        # Sort all dynamic objects by their grid y-then-x position for correct draw order
        sorted_objects = []
        for gx, gy in self.crate_positions:
            sorted_objects.append(((gx, gy), 'crate'))
        sorted_objects.append((self.player_pos, 'player'))
        
        # Sort key makes objects with higher y (further "down" on screen) draw later/on top
        sorted_objects.sort(key=lambda item: (item[0][1], item[0][0]))

        object_map = {pos: type for pos, type in sorted_objects}

        # Draw floor, targets, and then objects
        for gy in range(self.GRID_HEIGHT):
            for gx in range(self.GRID_WIDTH):
                pos = (gx, gy)
                iso_x, iso_y = self._grid_to_iso(gx, gy)
                
                # Draw base tile unless it's a wall
                if pos not in self.walls:
                    self._draw_iso_tile(self.screen, iso_x, iso_y, self.COLOR_TILE)
                
                # Highlight target locations
                if pos in self.targets:
                    self._draw_iso_tile(self.screen, iso_x, iso_y, self.COLOR_TARGET)
        
        # Draw walls, crates, and player in correct isometric order
        for gy in range(self.GRID_HEIGHT):
            for gx in range(self.GRID_WIDTH):
                pos = (gx, gy)
                iso_x, iso_y = self._grid_to_iso(gx, gy)

                if pos in self.walls:
                    self._draw_iso_cube(self.screen, iso_x, iso_y, self.COLOR_WALL, self.COLOR_WALL_SIDES)
                elif pos in object_map:
                    obj_type = object_map[pos]
                    if obj_type == 'crate':
                        self._draw_iso_cube(self.screen, iso_x, iso_y, self.COLOR_CRATE, self.COLOR_CRATE_SIDES)
                    elif obj_type == 'player':
                        self._draw_iso_cube(self.screen, iso_x, iso_y, self.COLOR_PLAYER, self.COLOR_PLAYER_SIDES)


    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((180, 80), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (self.WIDTH - 190, 10))

        # Timer
        time_left = max(0, self.time_remaining_frames / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        timer_color = self.COLOR_UI_TEXT if time_left > 10 else self.COLOR_UI_TEXT_WARN
        timer_surf = self.font_large.render(timer_text, True, timer_color)
        self.screen.blit(timer_surf, (self.WIDTH - 180, 20))

        # Targets
        crates_on_target = sum(1 for c in self.crate_positions if c in self.targets)
        targets_text = f"FILLED: {crates_on_target} / {self.num_targets}"
        targets_surf = self.font_small.render(targets_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(targets_surf, (self.WIDTH - 180, 55))

    def _grid_to_iso(self, x, y):
        iso_x = self.ORIGIN_X + (x - y) * (self.TILE_WIDTH_ISO / 2)
        iso_y = self.ORIGIN_Y + (x + y) * (self.TILE_HEIGHT_ISO / 2)
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, iso_x, iso_y, top_color, side_color):
        ch = self.CUBE_HEIGHT
        twh = self.TILE_WIDTH_ISO / 2
        thh = self.TILE_HEIGHT_ISO / 2

        top_points = [
            (iso_x, iso_y - ch),
            (iso_x + twh, iso_y - ch + thh),
            (iso_x, iso_y - ch + thh * 2),
            (iso_x - twh, iso_y - ch + thh)
        ]
        left_side_points = [
            (iso_x - twh, iso_y + thh),
            (iso_x - twh, iso_y - ch + thh),
            (iso_x, iso_y - ch + thh * 2),
            (iso_x, iso_y + thh * 2)
        ]
        right_side_points = [
            (iso_x + twh, iso_y + thh),
            (iso_x + twh, iso_y - ch + thh),
            (iso_x, iso_y - ch + thh * 2),
            (iso_x, iso_y + thh * 2)
        ]

        pygame.gfxdraw.aapolygon(surface, left_side_points, side_color)
        pygame.gfxdraw.filled_polygon(surface, left_side_points, side_color)
        pygame.gfxdraw.aapolygon(surface, right_side_points, side_color)
        pygame.gfxdraw.filled_polygon(surface, right_side_points, side_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)

    def _draw_iso_tile(self, surface, iso_x, iso_y, color):
        twh = self.TILE_WIDTH_ISO / 2
        thh = self.TILE_HEIGHT_ISO / 2
        points = [
            (iso_x, iso_y),
            (iso_x + twh, iso_y + thh),
            (iso_x, iso_y + thh * 2),
            (iso_x - twh, iso_y + thh)
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)
        
    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Use a dummy window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action is no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation back to pygame surface for display
        display_obs = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(display_obs)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        env.clock.tick(env.FPS)

    env.close()