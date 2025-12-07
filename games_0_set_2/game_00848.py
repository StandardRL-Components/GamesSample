
# Generated: 2025-08-27T14:59:17.454967
# Source Brief: brief_00848.md
# Brief Index: 848

        
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
    """
    An isometric physics puzzler where the player tilts a cavern floor
    to slide and stack crystals. The objective is to align five crystals
    of the same color before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to tilt the cavern floor. "
        "The goal is to align 5 crystals of the same color."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric physics puzzler. Tilt the cavern to slide "
        "crystals. Align five of the same color horizontally, vertically, "
        "or diagonally to win before the timer runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.NUM_CRYSTALS = 35
        self.NUM_COLORS = 5
        self.MAX_STEPS = 300 # Effectively the timer
        self.CRYSTAL_BASE_SIZE = 28
        self.CRYSTAL_HEIGHT = 12

        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (50, 55, 70)
        self.COLOR_WALL = (35, 38, 48)
        self.CRYSTAL_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.COLOR_TIMER_GOOD = (100, 220, 100)
        self.COLOR_TIMER_BAD = (220, 100, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_ALIGN_LINE = (255, 255, 255)

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
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 48, bold=True)
        
        # --- Isometric Projection ---
        self.tile_w_half = self.CRYSTAL_BASE_SIZE / 2
        self.tile_h_half = self.CRYSTAL_BASE_SIZE / 4
        self.origin_x = self.WIDTH // 2
        self.origin_y = self.HEIGHT // 2 - self.GRID_HEIGHT * self.tile_h_half

        # --- Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.grid = {}
        self.crystals = []
        self.particles = []
        self.last_alignments = []
        self.np_random = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.particles = []
        self.last_alignments = []

        # Procedural generation of crystals
        while True:
            self.grid = {}
            self.crystals = []
            occupied_coords = set()
            for i in range(self.NUM_CRYSTALS):
                while True:
                    x = self.np_random.integers(0, self.GRID_WIDTH)
                    y = self.np_random.integers(0, self.GRID_HEIGHT)
                    if (x, y) not in occupied_coords:
                        occupied_coords.add((x, y))
                        break
                
                color_idx = self.np_random.integers(0, self.NUM_COLORS)
                crystal = {
                    "id": i,
                    "pos": (x, y),
                    "color_idx": color_idx
                }
                self.crystals.append(crystal)
                self.grid[(x, y)] = [crystal]
            
            # Ensure it's not an immediate win
            if self._check_alignments()[1] == 0:
                break
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        reward = -0.1 # Cost of living

        if movement != 0:
            self._apply_gravity(movement)
            # Sound: sfx_crystals_sliding.wav
        
        self._update_particles()

        fours, fives = self._check_alignments()
        reward += fours * 1.0
        reward += fives * 5.0

        terminated = False
        if fives > 0:
            terminated = True
            reward += 100.0
            # Sound: sfx_win_chime.wav
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if fives == 0: # Only apply penalty if not a win on the last step
                reward -= 100.0
                # Sound: sfx_lose_buzzer.wav

        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _apply_gravity(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        if movement == 1: dx, dy = 0, -1   # Up
        elif movement == 2: dx, dy = 0, 1  # Down
        elif movement == 3: dx, dy = -1, 0 # Left
        elif movement == 4: dx, dy = 1, 0  # Right
        else: return False

        sort_key = lambda pos: -pos[0] * dx - pos[1] * dy
        
        occupied_cells = sorted([pos for pos in self.grid.keys()], key=sort_key)

        for x, y in occupied_cells:
            if (x, y) in self.grid:
                stack = self.grid.pop((x, y))
                
                cx, cy = x, y
                while True:
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT):
                        break
                    if (nx, ny) in self.grid:
                        break
                    cx, cy = nx, ny
                
                if (cx, cy) != (x, y):
                    # Sound: sfx_crystal_land.wav
                    self._spawn_particles(cx, cy, len(stack))

                self.grid[(cx, cy)] = stack
                for crystal in stack:
                    crystal['pos'] = (cx, cy)
        return True

    def _check_alignments(self):
        top_crystal_grid = {}
        for (x, y), stack in self.grid.items():
            if stack:
                top_crystal_grid[(x, y)] = stack[-1]['color_idx']

        fours, fives = 0, 0
        found_alignments = []
        
        for i in range(self.NUM_COLORS):
            # Horizontal, Vertical, Diagonal (\), Diagonal (/)
            for d in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                for y in range(self.GRID_HEIGHT):
                    for x in range(self.GRID_WIDTH):
                        line = []
                        for k in range(5):
                            px, py = x + k * d[0], y + k * d[1]
                            if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT:
                                if top_crystal_grid.get((px, py)) == i:
                                    line.append((px, py))
                                else:
                                    break
                            else:
                                break
                        
                        if len(line) >= 4:
                            is_new = True
                            # Check if any point in this new line is part of an already found line
                            # This prevents over-counting (e.g., a line of 6 being one 5 and two 4s)
                            flat_found = [item for sublist in found_alignments for item in sublist]
                            if any(p in flat_found for p in line):
                                is_new = False
                            
                            if is_new:
                                if len(line) >= 5: fives += 1
                                else: fours += 1
                                found_alignments.append(line)
        
        self.last_alignments = found_alignments
        return fours, fives
    
    def _cart_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * self.tile_w_half
        iso_y = self.origin_y + (x + y) * self.tile_h_half
        return int(iso_x), int(iso_y)

    def _spawn_particles(self, grid_x, grid_y, count):
        iso_x, iso_y = self._cart_to_iso(grid_x, grid_y)
        for _ in range(count * 3):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed - 1.5)
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({'pos': [iso_x, iso_y + self.tile_h_half], 'vel': vel, 'life': lifespan, 'color': self.COLOR_GRID})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'] = (p['vel'][0] * 0.95, p['vel'][1] * 0.95 + 0.1)
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_grid()
        self._render_crystals()
        self._render_particles()
        self._render_alignment_lines()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw cavern walls
        for y in range(self.GRID_HEIGHT + 1):
            start_iso = self._cart_to_iso(-1, y)
            end_iso = self._cart_to_iso(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_WALL, start_iso, end_iso, 1)
        for x in range(self.GRID_WIDTH + 1):
            start_iso = self._cart_to_iso(x, -1)
            end_iso = self._cart_to_iso(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_WALL, start_iso, end_iso, 1)

    def _render_grid(self):
        for y in range(self.GRID_HEIGHT + 1):
            start_iso = self._cart_to_iso(0, y)
            end_iso = self._cart_to_iso(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_iso, end_iso)
        for x in range(self.GRID_WIDTH + 1):
            start_iso = self._cart_to_iso(x, 0)
            end_iso = self._cart_to_iso(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_iso, end_iso)

    def _render_crystals(self):
        render_list = []
        for (gx, gy), stack in self.grid.items():
            iso_x, iso_y = self._cart_to_iso(gx, gy)
            for i, crystal in enumerate(stack):
                render_list.append({
                    "iso_pos": (iso_x, iso_y - i * self.CRYSTAL_HEIGHT),
                    "color": self.CRYSTAL_COLORS[crystal['color_idx']],
                    "depth": iso_y - i * self.CRYSTAL_HEIGHT
                })
        
        render_list.sort(key=lambda item: item['depth'])

        for item in render_list:
            self._draw_iso_cube(item['iso_pos'], item['color'])
    
    def _draw_iso_cube(self, pos, color):
        iso_x, iso_y = pos
        
        # Glow effect
        glow_color = (*color, 60)
        pygame.gfxdraw.filled_circle(self.screen, int(iso_x), int(iso_y), int(self.CRYSTAL_BASE_SIZE * 0.6), glow_color)
        
        top_points = [
            (iso_x, iso_y - self.tile_h_half),
            (iso_x + self.tile_w_half, iso_y),
            (iso_x, iso_y + self.tile_h_half),
            (iso_x - self.tile_w_half, iso_y)
        ]
        
        left_face_points = [
            (iso_x - self.tile_w_half, iso_y),
            (iso_x, iso_y + self.tile_h_half),
            (iso_x, iso_y + self.tile_h_half + self.CRYSTAL_HEIGHT),
            (iso_x - self.tile_w_half, iso_y + self.CRYSTAL_HEIGHT)
        ]

        right_face_points = [
            (iso_x + self.tile_w_half, iso_y),
            (iso_x, iso_y + self.tile_h_half),
            (iso_x, iso_y + self.tile_h_half + self.CRYSTAL_HEIGHT),
            (iso_x + self.tile_w_half, iso_y + self.CRYSTAL_HEIGHT)
        ]
        
        darker_color = tuple(max(0, c - 50) for c in color)
        darkest_color = tuple(max(0, c - 80) for c in color)
        
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)
        pygame.gfxdraw.aapolygon(self.screen, top_points, color)
        
        pygame.gfxdraw.filled_polygon(self.screen, left_face_points, darker_color)
        pygame.gfxdraw.aapolygon(self.screen, left_face_points, darker_color)

        pygame.gfxdraw.filled_polygon(self.screen, right_face_points, darkest_color)
        pygame.gfxdraw.aapolygon(self.screen, right_face_points, darkest_color)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, p['life'] / 5)
            pygame.draw.circle(self.screen, p['color'], p['pos'], size)

    def _render_alignment_lines(self):
        if self.steps % 10 < 5: # Flashing effect
            for line in self.last_alignments:
                points = []
                for gx, gy in line:
                    stack_height = len(self.grid.get((gx, gy), []))
                    iso_x, iso_y = self._cart_to_iso(gx, gy)
                    iso_y -= (stack_height -1) * self.CRYSTAL_HEIGHT
                    points.append((iso_x, iso_y))
                if len(points) > 1:
                    pygame.draw.lines(self.screen, self.COLOR_ALIGN_LINE, False, points, 3)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Steps
        steps_text = self.font_ui.render(f"MOVES: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 30))

        # Timer bar
        time_ratio = max(0, (self.MAX_STEPS - self.steps)) / self.MAX_STEPS
        timer_color = self.COLOR_TIMER_GOOD if time_ratio > 0.25 else self.COLOR_TIMER_BAD
        bar_width = 150
        bar_height = 20
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, timer_color, (bar_x, bar_y, bar_width * time_ratio, bar_height))
        timer_text = self.font_ui.render("TIME", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (bar_x + (bar_width - timer_text.get_width()) // 2, bar_y + 2))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self._check_alignments()[1] > 0:
                msg = "YOU WIN!"
                msg_color = self.COLOR_TIMER_GOOD
            else:
                msg = "TIME UP!"
                msg_color = self.COLOR_TIMER_BAD
            
            msg_surf = self.font_msg.render(msg, True, msg_color)
            self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))

    def _get_info(self):
        fours, fives = self._check_alignments()
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.MAX_STEPS - self.steps,
            "alignments_4": fours,
            "alignments_5": fives
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play ---
    # To run, you'll need to install pygame: pip install pygame
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Crystal Cavern")
    
    obs, info = env.reset()
    terminated = False
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if not env.game_over:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    terminated = False
                    action[0] = 0 # No move on reset
                    
        # Since auto_advance is False, we only step if an action is taken
        if action[0] != 0 and not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # Draw the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for human play
        
    env.close()