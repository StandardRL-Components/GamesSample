
# Generated: 2025-08-28T01:40:11.384297
# Source Brief: brief_04180.md
# Brief Index: 4180

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to cycle through crystal types. "
        "Press Shift to place a crystal and redirect the laser."
    )

    game_description = (
        "A puzzle game of light and crystal. Place reflective crystals in a cavern to guide a laser beam to the exit portal. "
        "You have a limited number of crystals, so place them wisely!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 12
        self.TILE_W, self.TILE_H = 28, 14
        self.MAX_STEPS = 1000
        self.INITIAL_MOVES = 15
        
        self.COLORS = {
            "bg": (15, 18, 32),
            "grid_line": (30, 35, 60),
            "wall_top": (70, 80, 110),
            "wall_side": (50, 60, 90),
            "start": (0, 255, 128),
            "exit": (255, 200, 0),
            "laser": (255, 20, 50),
            "laser_glow": (180, 20, 50),
            "cursor": (255, 255, 255),
            "text": (220, 220, 240),
            "ui_bg": (40, 45, 75, 180),
            "crystal_types": [
                (255, 80, 80),   # 0: / mirror
                (80, 255, 80),   # 1: \ mirror
                (80, 80, 255),   # 2: - mirror (reflects vertical)
                (255, 80, 255),  # 3: | mirror (reflects horizontal)
                (150, 150, 150), # 4: passthrough / remove
            ]
        }
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # --- Game State Initialization ---
        self.grid = None
        self.cursor_pos = None
        self.start_pos = None
        self.start_dir = None
        self.exit_pos = None
        self.crystals = None
        self.laser_path = None
        self.particles = None
        
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.level = 1
        self.selected_crystal_type = 0
        self.exit_reached = False
        
        self.previous_space_held = False
        self.previous_shift_held = False
        self.last_laser_dist = float('inf')
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.game_over = False
        self.exit_reached = False
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.crystals = {}
        self.particles = []
        self.selected_crystal_type = 0
        
        self.previous_space_held = False
        self.previous_shift_held = False

        self._generate_level()
        self._calculate_laser_path()
        self.last_laser_dist = self._get_laser_dist_to_exit()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_action, shift_action = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        turn_taken = False

        # --- Handle Actions ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)
            
        # Cycle crystal on space PRESS
        if space_action and not self.previous_space_held:
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.COLORS["crystal_types"])
            # sfx: crystal_cycle.wav

        # Place crystal on shift PRESS
        if shift_action and not self.previous_shift_held:
            pos = tuple(self.cursor_pos)
            if self.grid[pos[1], pos[0]] == 0: # Can only place on empty tiles
                if self.selected_crystal_type == 4: # Type 4 is "remove"
                    if pos in self.crystals:
                        del self.crystals[pos]
                        turn_taken = True
                        # sfx: crystal_remove.wav
                elif self.moves_left > 0:
                    self.crystals[pos] = self.selected_crystal_type
                    self.moves_left -= 1
                    turn_taken = True
                    # sfx: crystal_place.wav

        self.previous_space_held = space_action
        self.previous_shift_held = shift_action
        
        # --- Update Game State if a turn was taken ---
        if turn_taken:
            self._calculate_laser_path()
            reward = self._calculate_reward()
            self.score += reward
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.exit_reached:
            terminal_reward = -100
            self.score += terminal_reward
            reward += terminal_reward
        elif self.exit_reached:
            terminal_reward = 100
            self.score += terminal_reward
            reward += terminal_reward
            self.level += 1 # Progress to next level on win
            
        # Update particles
        self._update_particles()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_level(self):
        """Creates a pre-defined level layout."""
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        # Use seed to select from a few layouts for variety
        rng = np.random.default_rng(self.level)
        layout_choice = rng.integers(0, 3)

        if layout_choice == 0:
            self.start_pos, self.start_dir = (1, 5), (1, 0)
            self.exit_pos = (self.GRID_WIDTH - 2, 5)
            for i in range(self.GRID_HEIGHT):
                if i not in [4, 5, 6]: self.grid[i, 8] = 1
            for i in range(4, self.GRID_WIDTH - 4):
                 self.grid[2, i] = 1
                 self.grid[self.GRID_HEIGHT - 3, i] = 1
        elif layout_choice == 1:
            self.start_pos, self.start_dir = (1, 1), (1, 0)
            self.exit_pos = (self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2)
            for i in range(5, self.GRID_WIDTH - 5): self.grid[self.GRID_HEIGHT // 2, i] = 1
            for i in range(0, self.GRID_HEIGHT // 2 - 1): self.grid[i, 5] = 1
            for i in range(self.GRID_HEIGHT // 2 + 2, self.GRID_HEIGHT): self.grid[i, self.GRID_WIDTH - 6] = 1
        else:
            self.start_pos, self.start_dir = (self.GRID_WIDTH-2, 1), (-1, 0)
            self.exit_pos = (1, self.GRID_HEIGHT - 2)
            self.grid[3:self.GRID_HEIGHT-3, 5] = 1
            self.grid[3:self.GRID_HEIGHT-3, self.GRID_WIDTH-6] = 1


        self.grid[self.start_pos[1], self.start_pos[0]] = 2
        self.grid[self.exit_pos[1], self.exit_pos[0]] = 3

    def _calculate_laser_path(self):
        self.laser_path = []
        self.exit_reached = False
        pos = list(self.start_pos)
        direction = list(self.start_dir)
        
        for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT * 2): # Max path length
            self.laser_path.append(tuple(pos))
            
            # Move one step
            pos[0] += direction[0]
            pos[1] += direction[1]
            
            # Check boundaries
            if not (0 <= pos[0] < self.GRID_WIDTH and 0 <= pos[1] < self.GRID_HEIGHT):
                break
            
            # Check for interactions
            grid_val = self.grid[pos[1], pos[0]]
            
            if tuple(pos) == self.exit_pos:
                self.laser_path.append(tuple(pos))
                self.exit_reached = True
                # sfx: win.wav
                self._add_particles(pos, self.COLORS["exit"], 30)
                break
                
            if grid_val == 1: # Wall
                self.laser_path.append(tuple(pos))
                # sfx: laser_hit_wall.wav
                self._add_particles(pos, (200, 200, 200), 5)
                # Reflect
                pos[0] -= direction[0]
                pos[1] -= direction[1]
                # Assume walls are axis-aligned
                # This simple check works for grid movement
                if self.grid[min(self.GRID_HEIGHT-1, max(0, pos[1]))][min(self.GRID_WIDTH-1, max(0, pos[0] + direction[0]))] == 1:
                    direction[0] *= -1
                if self.grid[min(self.GRID_HEIGHT-1, max(0, pos[1] + direction[1]))][min(self.GRID_WIDTH-1, max(0, pos[0]))] == 1:
                    direction[1] *= -1
                continue

            if tuple(pos) in self.crystals:
                self.laser_path.append(tuple(pos))
                # sfx: laser_hit_crystal.wav
                crystal_type = self.crystals[tuple(pos)]
                self._add_particles(pos, self.COLORS["crystal_types"][crystal_type], 10)

                dx, dy = direction
                if crystal_type == 0: # / mirror
                    direction = [-dy, -dx]
                elif crystal_type == 1: # \ mirror
                    direction = [dy, dx]
                elif crystal_type == 2: # - mirror
                    if dy != 0: direction = [dx, -dy]
                elif crystal_type == 3: # | mirror
                    if dx != 0: direction = [-dx, dy]
                # Type 4 (passthrough) does nothing
                continue

    def _get_laser_dist_to_exit(self):
        if not self.laser_path:
            return abs(self.start_pos[0] - self.exit_pos[0]) + abs(self.start_pos[1] - self.exit_pos[1])
        
        laser_end = self.laser_path[-1]
        return abs(laser_end[0] - self.exit_pos[0]) + abs(laser_end[1] - self.exit_pos[1])

    def _calculate_reward(self):
        if self.exit_reached:
            return 0 # Terminal reward is handled in step()
        
        current_dist = self._get_laser_dist_to_exit()
        dist_change = self.last_laser_dist - current_dist
        
        reward = float(dist_change) # Continuous feedback
        
        if dist_change > 0:
            reward += 5.0 # Event: good redirection
        elif dist_change <= 0:
            reward -= 2.0 # Event: bad placement
        
        self.last_laser_dist = current_dist
        return reward

    def _check_termination(self):
        if self.exit_reached:
            self.game_over = True
            return True
        if self.moves_left <= 0 and not self.exit_reached:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _iso_to_screen(self, x, y):
        screen_x = (self.WIDTH / 2) + (x - y) * self.TILE_W
        screen_y = (self.HEIGHT / 4) + (x + y) * self.TILE_H
        return int(screen_x), int(screen_y)

    def _add_particles(self, pos, color, count):
        screen_pos = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(10, 25)
            self.particles.append([list(screen_pos), vel, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLORS["bg"])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid, walls, and special tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                tile_type = self.grid[y, x]
                
                # Draw floor tile
                p1 = self._iso_to_screen(x, y)
                p2 = self._iso_to_screen(x + 1, y)
                p3 = self._iso_to_screen(x + 1, y + 1)
                p4 = self._iso_to_screen(x, y + 1)
                pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3, p4), self.COLORS["grid_line"])
                
                if tile_type == 1: # Wall
                    wall_height = 2.5
                    p1_top = self._iso_to_screen(x, y)
                    p2_top = self._iso_to_screen(x + 1, y)
                    p3_top = self._iso_to_screen(x + 1, y + 1)
                    p4_top = self._iso_to_screen(x, y + 1)
                    pygame.gfxdraw.filled_polygon(self.screen, (p1_top, p2_top, p3_top, p4_top), self.COLORS["wall_top"])
                    
                    p3_bot = (p3_top[0], p3_top[1] + self.TILE_H * wall_height)
                    p4_bot = (p4_top[0], p4_top[1] + self.TILE_H * wall_height)
                    pygame.gfxdraw.filled_polygon(self.screen, (p4_top, p3_top, p3_bot, p4_bot), self.COLORS["wall_side"])
                    
                    p2_bot = (p2_top[0], p2_top[1] + self.TILE_H * wall_height)
                    pygame.gfxdraw.filled_polygon(self.screen, (p2_top, p3_top, p3_bot, p2_bot), self.COLORS["wall_side"])

                elif tile_type == 2: # Start
                    center = self._iso_to_screen(x + 0.5, y + 0.5)
                    pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.TILE_H, self.COLORS["start"])
                    pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.TILE_H, self.COLORS["start"])

                elif tile_type == 3: # Exit
                    center = self._iso_to_screen(x + 0.5, y + 0.5)
                    pulse = abs(math.sin(self.steps * 0.1))
                    radius = int(self.TILE_H * (1 + 0.2 * pulse))
                    pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLORS["exit"])
                    pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLORS["exit"])
        
        # Draw laser path
        if len(self.laser_path) > 1:
            path_points = [self._iso_to_screen(p[0] + 0.5, p[1] + 0.5) for p in self.laser_path]
            pulse_width = 2 + 2 * abs(math.sin(self.steps * 0.2))
            pygame.draw.lines(self.screen, self.COLORS["laser_glow"], False, path_points, int(pulse_width * 2.5))
            pygame.draw.lines(self.screen, self.COLORS["laser"], False, path_points, int(pulse_width))
            
        # Draw crystals
        for pos, c_type in self.crystals.items():
            center = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
            color = self.COLORS["crystal_types"][c_type]
            radius = self.TILE_H // 2
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, color)
            
            # Draw symbol
            if c_type == 0: pygame.draw.aaline(self.screen, (255,255,255), (center[0]-radius//2, center[1]+radius//2), (center[0]+radius//2, center[1]-radius//2), 2)
            elif c_type == 1: pygame.draw.aaline(self.screen, (255,255,255), (center[0]-radius//2, center[1]-radius//2), (center[0]+radius//2, center[1]+radius//2), 2)
            elif c_type == 2: pygame.draw.aaline(self.screen, (255,255,255), (center[0]-radius//2, center[1]), (center[0]+radius//2, center[1]), 2)
            elif c_type == 3: pygame.draw.aaline(self.screen, (255,255,255), (center[0], center[1]-radius//2), (center[0], center[1]+radius//2), 2)

        # Draw particles
        for p in self.particles:
            size = int(p[2] / 5)
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p[0][0]), int(p[0][1]), size, p[3])

        # Draw cursor
        pulse = 0.6 + 0.4 * abs(math.sin(self.steps * 0.2))
        color = (*self.COLORS["cursor"], int(255 * pulse))
        p1 = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        p2 = self._iso_to_screen(self.cursor_pos[0] + 1, self.cursor_pos[1])
        p3 = self._iso_to_screen(self.cursor_pos[0] + 1, self.cursor_pos[1] + 1)
        p4 = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1] + 1)
        pygame.draw.lines(self.screen, color, True, (p1, p2, p3, p4), 2)

    def _render_ui(self):
        # UI Background
        ui_surf = pygame.Surface((self.WIDTH, 50), pygame.SRCALPHA)
        ui_surf.fill(self.COLORS["ui_bg"])
        self.screen.blit(ui_surf, (0, 0))
        
        # Moves Left
        moves_text = self.font_large.render(f"Crystals: {self.moves_left}", True, self.COLORS["text"])
        self.screen.blit(moves_text, (15, 12))
        
        # Level
        level_text = self.font_large.render(f"Cavern: {self.level}", True, self.COLORS["text"])
        self.screen.blit(level_text, (self.WIDTH - level_text.get_width() - 15, 12))
        
        # Selected Crystal
        selected_text = self.font_small.render("Selected:", True, self.COLORS["text"])
        self.screen.blit(selected_text, (self.WIDTH // 2 - 100, 15))
        
        c_type = self.selected_crystal_type
        color = self.COLORS["crystal_types"][c_type]
        center = (self.WIDTH // 2, 25)
        radius = self.TILE_H
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, color)
        
        if c_type == 0: pygame.draw.aaline(self.screen, (255,255,255), (center[0]-radius//2, center[1]+radius//2), (center[0]+radius//2, center[1]-radius//2), 2)
        elif c_type == 1: pygame.draw.aaline(self.screen, (255,255,255), (center[0]-radius//2, center[1]-radius//2), (center[0]+radius//2, center[1]+radius//2), 2)
        elif c_type == 2: pygame.draw.aaline(self.screen, (255,255,255), (center[0]-radius//2, center[1]), (center[0]+radius//2, center[1]), 2)
        elif c_type == 3: pygame.draw.aaline(self.screen, (255,255,255), (center[0], center[1]-radius//2), (center[0], center[1]+radius//2), 2)
        elif c_type == 4:
            pygame.draw.line(self.screen, (255, 50, 50), (center[0]-radius//2, center[1]-radius//2), (center[0]+radius//2, center[1]+radius//2), 3)
            pygame.draw.line(self.screen, (255, 50, 50), (center[0]-radius//2, center[1]+radius//2), (center[0]+radius//2, center[1]-radius//2), 3)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            msg = "VICTORY!" if self.exit_reached else "OUT OF CRYSTALS"
            color = self.COLORS["exit"] if self.exit_reached else self.COLORS["laser"]
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "level": self.level,
            "exit_reached": self.exit_reached,
        }
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Level: {info['level']}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Limit to 30 FPS for human play
        
    env.close()