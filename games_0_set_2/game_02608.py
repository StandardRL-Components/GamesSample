import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import os
import math
import os
import pygame


# Set Pygame to run headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to select/place a crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A puzzle game where you refract a light beam to an exit by moving crystals."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 12
        self.TILE_W, self.TILE_H = 32, 16
        self.ORIGIN_X, self.ORIGIN_Y = self.WIDTH // 2, 80
        self.MAX_STEPS = 1000
        self.INITIAL_TIME = 60
        self.NUM_CRYSTALS = 5
        self.NUM_DISTRACTORS = 3

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (30, 35, 60)
        self.COLOR_WALL = (60, 70, 100)
        self.COLOR_WALL_TOP = (80, 90, 120)
        self.COLOR_LIGHT_BEAM = (255, 255, 100)
        self.COLOR_LIGHT_GLOW = (200, 200, 0, 90)
        self.COLOR_CRYSTAL = (0, 255, 255)
        self.COLOR_CRYSTAL_GLOW = (0, 200, 200, 100)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SOURCE = (255, 200, 0)
        self.COLOR_EXIT = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        
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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.grid = None
        self.crystals = None
        self.light_source_pos = None
        self.light_source_dir = None
        self.exit_pos = None
        self.cursor_pos = None
        self.selected_crystal_idx = None
        self.light_path = None
        self.last_beam_dist = 0
        self.particles = []
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.win_message = ""
        
        # This is called to initialize np_random and other gym internals
        # We call it here so that _generate_level can use self.np_random
        super().reset(seed=None)
        
        # Now we can fully initialize the game state
        self._generate_level()
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self._calculate_light_path()
        endpoint = self.light_path[-1]
        self.last_beam_dist = abs(endpoint[0] - self.exit_pos[0]) + abs(endpoint[1] - self.exit_pos[1])
        
    def _iso_transform(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_W / 2
        y = self.ORIGIN_Y + (c + r) * self.TILE_H / 2
        return int(x), int(y)

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        # 0: empty, 1: wall, 2: crystal, 3: source, 4: exit
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        possible_coords = []
        for r in range(1, self.GRID_HEIGHT - 1):
            for c in range(1, self.GRID_WIDTH - 1):
                possible_coords.append((r, c))
        
        # Use a copy for shuffling if needed elsewhere, but here it's fine
        self.np_random.shuffle(possible_coords)

        # Backwards generation for a guaranteed solution
        self.exit_pos = possible_coords.pop()
        self.grid[self.exit_pos] = 4

        path_crystals = []
        current_pos = self.exit_pos
        # Directions: (dr, dc) -> 0:E, 1:N, 2:W, 3:S
        dir_vectors = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        # Pick a random entry direction for the exit
        entry_dir_idx = self.np_random.integers(0, 4)
        
        for _ in range(self.NUM_CRYSTALS):
            # Move back from current pos along entry_dir
            dr, dc = dir_vectors[entry_dir_idx]
            path_len = self.np_random.integers(2, 5)
            
            valid_spot = False
            for i in range(path_len, 0, -1):
                next_pos = (current_pos[0] - dr*i, current_pos[1] - dc*i)
                if next_pos in possible_coords:
                    possible_coords.remove(next_pos)
                    path_crystals.append(next_pos)
                    current_pos = next_pos
                    valid_spot = True
                    break
            if not valid_spot: # Could not find a spot, end path generation
                break

            # New direction after reflection (all crystals are '/' mirrors)
            # E(0)->N(1), N(1)->E(0), W(2)->S(3), S(3)->W(2)
            if entry_dir_idx == 0: entry_dir_idx = 1
            elif entry_dir_idx == 1: entry_dir_idx = 0
            elif entry_dir_idx == 2: entry_dir_idx = 3
            elif entry_dir_idx == 3: entry_dir_idx = 2
        
        # Set light source
        dr, dc = dir_vectors[entry_dir_idx]
        self.light_source_pos = (current_pos[0] - dr, current_pos[1] - dc)
        if self.light_source_pos in possible_coords:
            possible_coords.remove(self.light_source_pos)
        self.light_source_dir = (dr, dc)
        self.grid[self.light_source_pos] = 3

        # Place crystals
        self.crystals = []
        for pos in path_crystals:
            self.crystals.append(list(pos))
            self.grid[pos] = 2
        
        # Place distractor crystals
        for _ in range(self.NUM_DISTRACTORS):
            if not possible_coords: break
            pos = possible_coords.pop()
            self.crystals.append(list(pos))
            self.grid[pos] = 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.INITIAL_TIME
        self.game_over = False
        self.win_message = ""
        self.selected_crystal_idx = None
        self.particles = []

        self._generate_level()
        
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        
        self._calculate_light_path()
        endpoint = self.light_path[-1]
        self.last_beam_dist = abs(endpoint[0] - self.exit_pos[0]) + abs(endpoint[1] - self.exit_pos[1])
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        action_taken = False

        # --- Action Handling ---
        # 1. Cursor Movement
        if movement > 0:
            dr, dc = [(0,0), (-1,0), (1,0), (0,-1), (0,1)][movement]
            new_r, new_c = self.cursor_pos[0] + dr, self.cursor_pos[1] + dc
            if 1 <= new_r < self.GRID_HEIGHT -1 and 1 <= new_c < self.GRID_WIDTH -1:
                self.cursor_pos = [new_r, new_c]

        # 2. Crystal Interaction (Spacebar)
        if space_pressed:
            cursor_tuple = tuple(self.cursor_pos)
            
            if self.selected_crystal_idx is not None:
                # Try to place selected crystal
                if self.grid[cursor_tuple] == 0: # Empty space
                    # Move crystal
                    old_pos = tuple(self.crystals[self.selected_crystal_idx])
                    self.grid[old_pos] = 0
                    self.crystals[self.selected_crystal_idx] = list(cursor_tuple)
                    self.grid[cursor_tuple] = 2
                    self.selected_crystal_idx = None
                    action_taken = True
                    self._spawn_particles(self.cursor_pos, 15, self.COLOR_CURSOR)
                elif cursor_tuple == tuple(self.crystals[self.selected_crystal_idx]):
                    # Deselect
                    self.selected_crystal_idx = None
            else:
                # Try to select a crystal
                if self.grid[cursor_tuple] == 2:
                    for i, crys_pos in enumerate(self.crystals):
                        if tuple(crys_pos) == cursor_tuple:
                            self.selected_crystal_idx = i
                            self._spawn_particles(self.cursor_pos, 10, self.COLOR_CRYSTAL)
                            break
        
        # --- Game State Update ---
        if action_taken:
            self.timer -= 1
            self.steps += 1
            self._calculate_light_path()
        else: # No-op or invalid move
            self.steps += 1

        self._update_particles()
        reward = self._calculate_reward()
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_light_path(self):
        self.light_path = [self.light_source_pos]
        pos = list(self.light_source_pos)
        dr, dc = self.light_source_dir
        
        for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT):
            pos[0] += dr
            pos[1] += dc
            
            r, c = int(pos[0]), int(pos[1])
            if not (0 <= r < self.GRID_HEIGHT and 0 <= c < self.GRID_WIDTH):
                break

            self.light_path.append((r, c))

            if self.grid[r, c] == 1: # Wall
                break
            elif self.grid[r, c] == 2: # Crystal
                # Correct logic for '/' mirror:
                # E (0,1) -> N (-1,0)
                # W (0,-1) -> S (1,0)
                # N (-1,0) -> E (0,1)
                # S (1,0) -> W (0,-1)
                if dr == 0: # Horizontal
                    new_dr, new_dc = -dc, 0
                else: # Vertical
                    new_dr, new_dc = 0, -dr
                dr, dc = new_dr, new_dc
            elif self.grid[r, c] == 4: # Exit
                break

    def _calculate_reward(self):
        reward = 0
        endpoint = self.light_path[-1]
        
        if endpoint == self.exit_pos:
            reward += 5.0
        
        new_dist = abs(endpoint[0] - self.exit_pos[0]) + abs(endpoint[1] - self.exit_pos[1])
        
        reward += (self.last_beam_dist - new_dist) * 0.1
        self.last_beam_dist = new_dist
        
        self.score += reward
        return reward

    def _check_termination(self):
        terminated = False
        endpoint = self.light_path[-1]

        if endpoint == self.exit_pos:
            self.score += 50
            terminated = True
            self.win_message = "SUCCESS"
        elif self.timer <= 0:
            self.score -= 50
            terminated = True
            self.win_message = "TIME UP"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.win_message = "STEP LIMIT"

        if terminated:
            self.game_over = True
        return terminated

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cursor_pos": self.cursor_pos,
            "selected_crystal": self.selected_crystal_idx,
        }
        
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
                screen_pos = self._iso_transform(r, c)
                self._draw_iso_tile(self.screen, screen_pos, self.COLOR_GRID, filled=False)

        # Draw walls, source, exit
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r,c] == 1:
                    screen_pos = self._iso_transform(r, c)
                    self._draw_iso_block(self.screen, screen_pos, self.COLOR_WALL, self.COLOR_WALL_TOP, 2)
                elif self.grid[r,c] == 3:
                    screen_pos = self._iso_transform(r, c)
                    self._draw_iso_tile(self.screen, screen_pos, self.COLOR_SOURCE, filled=True)
                elif self.grid[r,c] == 4:
                    screen_pos = self._iso_transform(r, c)
                    self._draw_iso_tile(self.screen, screen_pos, self.COLOR_EXIT, filled=True)

        self._draw_light_beam()
        self._draw_crystals()
        self._draw_particles()
        self._draw_cursor()

    def _draw_iso_tile(self, surface, pos, color, filled=True):
        points = [
            (pos[0], pos[1] - self.TILE_H // 2),
            (pos[0] + self.TILE_W // 2, pos[1]),
            (pos[0], pos[1] + self.TILE_H // 2),
            (pos[0] - self.TILE_W // 2, pos[1]),
        ]
        if filled:
            pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _draw_iso_block(self, surface, pos, side_color, top_color, height_mult):
        h = self.TILE_H * height_mult
        top_pos = (pos[0], pos[1] - h)
        self._draw_iso_tile(surface, top_pos, top_color, filled=True)
        
        p = [
            (pos[0] - self.TILE_W / 2, pos[1]),
            (pos[0] + self.TILE_W / 2, pos[1]),
            (pos[0] + self.TILE_W / 2, pos[1] - h),
            (pos[0] - self.TILE_W / 2, pos[1] - h)
        ]
        
        # right face
        points_right = [
            (pos[0] + self.TILE_W/2, pos[1]),
            (pos[0], pos[1] + self.TILE_H/2),
            (top_pos[0], top_pos[1] + self.TILE_H/2),
            (top_pos[0] + self.TILE_W/2, top_pos[1]),
        ]
        pygame.gfxdraw.filled_polygon(surface, points_right, side_color)
        pygame.gfxdraw.aapolygon(surface, points_right, side_color)

        # left face
        points_left = [
            (pos[0] - self.TILE_W/2, pos[1]),
            (pos[0], pos[1] + self.TILE_H/2),
            (top_pos[0], top_pos[1] + self.TILE_H/2),
            (top_pos[0] - self.TILE_W/2, top_pos[1]),
        ]
        pygame.gfxdraw.filled_polygon(surface, points_left, side_color)
        pygame.gfxdraw.aapolygon(surface, points_left, side_color)

    def _draw_crystals(self):
        for i, pos in enumerate(self.crystals):
            screen_pos = self._iso_transform(pos[0], pos[1])
            points = [
                (screen_pos[0], screen_pos[1] - self.TILE_H),
                (screen_pos[0] + self.TILE_W / 2, screen_pos[1] - self.TILE_H/2),
                (screen_pos[0], screen_pos[1]),
                (screen_pos[0] - self.TILE_W / 2, screen_pos[1] - self.TILE_H/2),
            ]
            
            # Glow
            if self.selected_crystal_idx == i:
                pygame.gfxdraw.filled_polygon(self.screen, points, (255,255,255,100))
                pygame.gfxdraw.aapolygon(self.screen, points, (255,255,255))
            else:
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL_GLOW)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)
            
            # Facet
            pygame.gfxdraw.line(self.screen, int(points[0][0]), int(points[0][1]), int(points[2][0]), int(points[2][1]), self.COLOR_CRYSTAL)
            pygame.gfxdraw.line(self.screen, int(points[1][0]), int(points[1][1]), int(points[3][0]), int(points[3][1]), self.COLOR_CRYSTAL)

    def _draw_light_beam(self):
        if len(self.light_path) < 2: return
        
        path_points = [self._iso_transform(r, c) for r, c in self.light_path]

        # Spawn particles along the beam
        if not self.game_over and self.np_random.random() < 0.8:
            for i in range(len(path_points) - 1):
                p1, p2 = path_points[i], path_points[i+1]
                if self.np_random.random() < 0.3: # Spawn chance per segment
                    lerp_factor = self.np_random.random()
                    start_pos = [p1[0] * (1-lerp_factor) + p2[0] * lerp_factor, p1[1] * (1-lerp_factor) + p2[1] * lerp_factor]
                    self.particles.append([start_pos, [0,0], self.np_random.integers(10, 20), self.COLOR_LIGHT_BEAM])

        # Draw glow
        pygame.draw.lines(self.screen, self.COLOR_LIGHT_GLOW, False, path_points, width=10)
        # Draw core beam
        pygame.draw.lines(self.screen, self.COLOR_LIGHT_BEAM, False, path_points, width=3)
        
    def _draw_cursor(self):
        screen_pos = self._iso_transform(self.cursor_pos[0], self.cursor_pos[1])
        bob = math.sin(pygame.time.get_ticks() * 0.005) * 3
        pos_y = screen_pos[1] - self.TILE_H - 5 - bob
        points = [
            (screen_pos[0] - 8, pos_y),
            (screen_pos[0] + 8, pos_y),
            (screen_pos[0], pos_y + 8)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CURSOR)
        if self.selected_crystal_idx is not None:
             pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CURSOR)

    def _spawn_particles(self, grid_pos, count, color):
        screen_pos = self._iso_transform(grid_pos[0], grid_pos[1])
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append([list(screen_pos), vel, life, color])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _draw_particles(self):
        for pos, vel, life, color in self.particles:
            radius = int(max(0, life / 5))
            if radius > 0:
                pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), radius)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        timer_text = self.font_small.render(f"TIME: {self.timer}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_EXIT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

# Example of how to run the environment
if __name__ == '__main__':
    # For human play, we want a window.
    # The "dummy" driver is set at the top for headless testing.
    # To run with a window, you might need to unset it or set it to a valid driver.
    # For example:
    os.environ["SDL_VIDEODRIVER"] = "x11" 

    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        movement = 0 # 0: none
        space_pressed = 0 # 0: released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                space_pressed = 1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # The environment steps only when a key is pressed (auto_advance=False)
        # For human play, we need to decide when to step. Let's step on any key press.
        action = [movement, space_pressed, 0] # shift is not used
        
        # Only step if an action is taken or if it's the first frame
        if any(action) or env.steps == 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over. Press 'R' to reset.")
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Limit FPS for human play

    pygame.quit()