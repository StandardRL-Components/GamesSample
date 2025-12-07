import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:58:05.390516
# Source Brief: brief_00773.md
# Brief Index: 773
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a shape-shifting unit navigates a 7x7 grid.
    The goal is to collect 15 energy orbs. The player's form changes upon
    collecting an orb, which alters its movement capabilities.
    - Square: Moves 1 space.
    - Circle: Moves 2 spaces and can destroy obstacles.
    - Triangle: Moves 3 spaces.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a grid as a shape-shifting unit to collect energy orbs. "
        "Each orb collected changes your form and movement abilities."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move your unit. Your movement distance "
        "depends on your current shape."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 7
    CELL_SIZE = 50
    GRID_WIDTH = GRID_SIZE * CELL_SIZE
    GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (50, 50, 70)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 200, 255, 50)
    COLOR_ORB = (255, 255, 0)
    COLOR_ORB_GLOW = (255, 255, 0, 60)
    COLOR_OBSTACLE = (50, 50, 50)
    COLOR_OBSTACLE_BORDER = (80, 80, 80)
    COLOR_TEXT = (220, 220, 240)
    
    PLAYER_FORMS = {
        0: {"name": "SQUARE", "move": 1},
        1: {"name": "CIRCLE", "move": 2},
        2: {"name": "TRIANGLE", "move": 3},
    }
    
    WIN_SCORE = 15
    MAX_STEPS = 1000
    INITIAL_OBSTACLES = 8
    INITIAL_ORBS = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Game state variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_form = 0
        self.orbs = []
        self.obstacles = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_form = 0
        self.particles = []

        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Generates a new random level layout."""
        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        empty_cells = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if [c, r] != self.player_pos:
                    empty_cells.append([c, r])
        
        # Use self.np_random for reproducibility
        self.np_random.shuffle(empty_cells)
        
        num_obstacles = min(self.INITIAL_OBSTACLES, len(empty_cells))
        self.obstacles = [tuple(pos) for pos in empty_cells[:num_obstacles]]
        
        empty_cells = empty_cells[num_obstacles:]
        
        num_orbs = min(self.INITIAL_ORBS, len(empty_cells))
        self.orbs = [tuple(pos) for pos in empty_cells[:num_orbs]]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        
        dist_before = self._get_distance_to_nearest_orb(self.player_pos)

        if movement != 0:
            move_dist = self.PLAYER_FORMS[self.player_form]["move"]
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            
            target_pos = [self.player_pos[0] + dx * move_dist, self.player_pos[1] + dy * move_dist]

            path = []
            for i in range(1, move_dist + 1):
                path.append((self.player_pos[0] + dx * i, self.player_pos[1] + dy * i))

            is_valid_move = 0 <= target_pos[0] < self.GRID_SIZE and 0 <= target_pos[1] < self.GRID_SIZE
            
            path_is_blocked = False
            if is_valid_move:
                if self.player_form != 1: 
                    for pos in path:
                        if pos in self.obstacles:
                            path_is_blocked = True
                            break
            
            if is_valid_move and not path_is_blocked:
                self.player_pos = target_pos
                
                if tuple(self.player_pos) in self.orbs:
                    self.orbs.remove(tuple(self.player_pos))
                    self.score += 1
                    reward += 1.0
                    self.player_form = (self.player_form + 1) % 3
                    # sfx: orb_collect.wav
                    self._create_particles(self._grid_to_pixel(self.player_pos), self.COLOR_ORB, 30)
                    self._spawn_item(is_orb=True)

                if self.player_form == 1:
                    for pos in path:
                        if pos in self.obstacles:
                            self.obstacles.remove(pos)
                            # sfx: obstacle_destroy.wav
                            self._create_particles(self._grid_to_pixel(pos), (100,100,110), 20, speed=2)
        
        dist_after = self._get_distance_to_nearest_orb(self.player_pos)
        if dist_after is not None and dist_before is not None and dist_after < dist_before:
            reward += 0.1

        self.steps += 1
        terminated = self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
        truncated = False
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Victory reward
                # sfx: victory.wav

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_item(self, is_orb):
        """Spawns a new orb or obstacle in a random empty cell."""
        empty_cells = []
        occupied = set(self.obstacles) | set(self.orbs) | {tuple(self.player_pos)}
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (c, r) not in occupied:
                    empty_cells.append((c, r))
        
        if empty_cells:
            idx = self.np_random.integers(0, len(empty_cells))
            if is_orb:
                self.orbs.append(empty_cells[idx])
            else:
                self.obstacles.append(empty_cells[idx])

    def _get_distance_to_nearest_orb(self, pos):
        if not self.orbs:
            return None
        distances = [abs(pos[0] - o[0]) + abs(pos[1] - o[1]) for o in self.orbs]
        return min(distances)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_particles()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_pixel(self, grid_pos):
        px = self.OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return (px, py)

    def _render_game(self):
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (self.OFFSET_X + i * self.CELL_SIZE, self.OFFSET_Y), 
                             (self.OFFSET_X + i * self.CELL_SIZE, self.OFFSET_Y + self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                             (self.OFFSET_X, self.OFFSET_Y + i * self.CELL_SIZE), 
                             (self.OFFSET_X + self.GRID_WIDTH, self.OFFSET_Y + i * self.CELL_SIZE))

        for pos in self.obstacles:
            px, py = self._grid_to_pixel(pos)
            rect = pygame.Rect(px - self.CELL_SIZE // 2, py - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_BORDER, rect, width=2, border_radius=4)

        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        orb_radius = int(self.CELL_SIZE * 0.2)
        glow_radius = orb_radius + int(pulse * 6)
        for pos in self.orbs:
            px, py = self._grid_to_pixel(pos)
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_ORB_GLOW)
            pygame.gfxdraw.aacircle(self.screen, px, py, glow_radius, self.COLOR_ORB_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, px, py, orb_radius, self.COLOR_ORB)
            pygame.gfxdraw.aacircle(self.screen, px, py, orb_radius, self.COLOR_ORB)

        self._draw_player()

    def _draw_player(self):
        px, py = self._grid_to_pixel(self.player_pos)
        form = self.player_form
        size = int(self.CELL_SIZE * 0.7)
        glow_size = int(size * 1.5)

        if form == 0: # Square
            self._draw_antialiased_rect(px - glow_size // 2, py - glow_size // 2, glow_size, glow_size, self.COLOR_PLAYER_GLOW, 8)
            self._draw_antialiased_rect(px - size // 2, py - size // 2, size, size, self.COLOR_PLAYER, 6)
        elif form == 1: # Circle
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_size // 2, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.aacircle(self.screen, px, py, glow_size // 2, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, px, py, size // 2, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, px, py, size // 2, self.COLOR_PLAYER)
        elif form == 2: # Triangle
            self._draw_antialiased_polygon(px, py, glow_size, 3, math.pi / 2, self.COLOR_PLAYER_GLOW)
            self._draw_antialiased_polygon(px, py, size, 3, math.pi / 2, self.COLOR_PLAYER)

    def _draw_antialiased_rect(self, x, y, w, h, color, radius):
        pygame.draw.rect(self.screen, color, (x, y, w, h), border_radius=radius)

    def _draw_antialiased_polygon(self, cx, cy, r, num_sides, angle_offset, color):
        points = []
        for i in range(num_sides):
            angle = (2 * math.pi * i / num_sides) + angle_offset
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((int(x), int(y)))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_ui(self):
        score_text = self.font_main.render(f"ORBS: {self.score} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        form_info = self.PLAYER_FORMS[self.player_form]
        form_text = self.font_small.render(f"FORM: {form_info['name']} (MOVE {form_info['move']})", True, self.COLOR_TEXT)
        self.screen.blit(form_text, (20, 45))
        
        step_text = self.font_small.render(f"STEPS: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        text_rect = step_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(step_text, text_rect)

    def _create_particles(self, pos, color, count, speed=4, lifetime=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(1, speed)
            vel = [math.cos(angle) * vel_mag, math.sin(angle) * vel_mag]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifetime': lifetime + self.np_random.integers(-5, 6),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            p['size'] *= 0.95
            
            if p['lifetime'] <= 0 or p['size'] < 1:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['lifetime'] / 20))
                color = (*p['color'], max(0, min(255, alpha)))
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                particle_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(particle_surf, (pos[0] - p['size'], pos[1] - p['size']))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # The __main__ block is for local testing and visualization, not part of the official API
    # It will not be executed by the grading environment.
    # To use it, you will need to install pygame (`pip install pygame`)
    # and remove the `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` line.
    
    # We need to re-init pygame with the default video driver
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit() # Quit the dummy instance
    pygame.init() # Re-init with default driver

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Shape Shifter Grid")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("UP/W, DOWN/S, LEFT/A, RIGHT/D to move.")
    print("R to reset.")
    print("Q to quit.")
    
    while running:
        action = [0, 0, 0] # Default to no-op
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q]:
                    running = False
                if event.key in [pygame.K_r]:
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"--- Env Reset ---")
                
                if event.key in [pygame.K_UP, pygame.K_w]: action[0] = 1
                elif event.key in [pygame.K_DOWN, pygame.K_s]: action[0] = 2
                elif event.key in [pygame.K_LEFT, pygame.K_a]: action[0] = 3
                elif event.key in [pygame.K_RIGHT, pygame.K_d]: action[0] = 4

        # Only step if a move action was taken
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Score: {info['score']}")
            
            if terminated:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
        
        # Render the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()