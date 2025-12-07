# Generated: 2025-08-28T06:28:21.393757
# Source Brief: brief_02939.md
# Brief Index: 2939

        
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


class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, x, y, color, life, size_range=(2, 5), speed_range=(1, 3)):
        self.x = x
        self.y = y
        self.vx = random.uniform(-speed_range[1], speed_range[1])
        self.vy = random.uniform(-speed_range[1], speed_range[1])
        self.life = life
        self.max_life = life
        self.color = color
        self.size = random.uniform(size_range[0], size_range[1])

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.vx *= 0.98
        self.vy *= 0.98

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (self.size, self.size), self.size)
            surface.blit(temp_surf, (self.x - self.size, self.y - self.size), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move your robot. Avoid the red lasers and reach the green exit."
    )

    game_description = (
        "Navigate a robot through deadly, shifting laser grids to reach the exit in this challenging puzzle game."
    )

    auto_advance = False

    # --- Constants ---
    GRID_SIZE = 10
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_DIM = 360
    CELL_SIZE = GRID_DIM // GRID_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_DIM) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_DIM) // 2

    COLOR_BG = (15, 18, 28)
    COLOR_GRID = (40, 45, 60)
    COLOR_WALL = (80, 90, 110)
    COLOR_ROBOT = (60, 180, 255)
    COLOR_ROBOT_GLOW = (10, 80, 150)
    COLOR_EXIT = (80, 255, 150)
    COLOR_EXIT_GLOW = (20, 120, 70)
    COLOR_LASER = (255, 50, 100)
    COLOR_LASER_GLOW = (150, 10, 40)
    COLOR_EMITTER = pygame.Color(200, 0, 50)
    COLOR_EMITTER_CHARGE = pygame.Color(255, 200, 200)
    COLOR_TEXT = (220, 220, 240)
    
    MAX_STEPS = 1000

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
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.robot_pos = [0, 0]
        self.exit_pos = [0, 0]
        self.walls = []
        self.lasers = []
        self.particles = []
        self.level = 1
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.level = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        while True:
            self.walls = []
            
            self.exit_pos = self.np_random.integers(0, self.GRID_SIZE, size=2).tolist()
            self.robot_pos = self.np_random.integers(0, self.GRID_SIZE, size=2).tolist()
            while self.robot_pos == self.exit_pos:
                self.robot_pos = self.np_random.integers(0, self.GRID_SIZE, size=2).tolist()

            num_walls = min(15, self.level + 2 + self.np_random.integers(0, 3))
            for _ in range(num_walls):
                wall_pos = self.np_random.integers(0, self.GRID_SIZE, size=2).tolist()
                if wall_pos != self.robot_pos and wall_pos != self.exit_pos:
                    self.walls.append(wall_pos)
            
            if self._is_solvable():
                break
        
        self.lasers = []
        num_lasers = min(10, self.level + self.np_random.integers(0, 2))
        for _ in range(num_lasers):
            laser_type = self.np_random.choice(['horizontal', 'vertical'])
            if self.level >= 2 and self.np_random.random() > 0.6:
                 laser_type = 'diagonal'

            pos = self.np_random.integers(0, self.GRID_SIZE)
            
            difficulty_factor = 1.0 - (self.level * 0.05)
            base_cycle_len = self.np_random.integers(6, 12)
            cycle_len = max(3, int(base_cycle_len * difficulty_factor))
            on_len = self.np_random.integers(1, max(2, cycle_len // 2))
            offset = self.np_random.integers(0, cycle_len)

            new_laser = {
                'pos': pos, 'type': laser_type,
                'cycle_len': cycle_len, 'on_len': on_len, 'offset': offset, 'timer': 0,
                'orientation': 1 if self.np_random.random() > 0.5 else -1
            }
            self.lasers.append(new_laser)

    def _is_solvable(self):
        q = [self.robot_pos]
        visited = {tuple(self.robot_pos)}
        while q:
            x, y = q.pop(0)
            if [x, y] == self.exit_pos:
                return True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and \
                   [nx, ny] not in self.walls and tuple([nx, ny]) not in visited:
                    visited.add(tuple([nx, ny]))
                    q.append([nx, ny])
        return False

    def step(self, action):
        if self.game_over:
            # On termination, subsequent steps do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        self.steps += 1
        for laser in self.lasers:
            laser['timer'] = (laser['timer'] + 1) % laser['cycle_len']
        
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        target_pos = list(self.robot_pos)
        if movement == 1: target_pos[1] -= 1
        elif movement == 2: target_pos[1] += 1
        elif movement == 3: target_pos[0] -= 1
        elif movement == 4: target_pos[0] += 1

        if 0 <= target_pos[0] < self.GRID_SIZE and \
           0 <= target_pos[1] < self.GRID_SIZE and \
           target_pos not in self.walls:
            self.robot_pos = target_pos
        
        terminated = False
        reward = -0.1

        if self._is_pos_in_laser(self.robot_pos):
            # sfx: player_explosion
            self._create_particles(self.robot_pos, self.COLOR_LASER, 50)
            reward = -100.0
            terminated = True
            self.game_over = True
        
        elif self.robot_pos == self.exit_pos:
            # sfx: level_complete
            self._create_particles(self.exit_pos, self.COLOR_EXIT, 50)
            reward = 100.0
            terminated = True
            self.game_over = True
        
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        if not terminated:
            is_risky = False
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                adj_pos = [self.robot_pos[0] + dx, self.robot_pos[1] + dy]
                if self._is_pos_in_laser(adj_pos):
                    is_risky = True
                    break
            reward += 2.0 if is_risky else -0.2
        
        self.score += reward

        # If win, increment level for next episode's reset
        if self.robot_pos == self.exit_pos and terminated:
             self.level += 1

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_laser_active(self, laser):
        return (laser['timer'] - laser['offset'] + laser['cycle_len']) % laser['cycle_len'] < laser['on_len']

    def _is_pos_in_diagonal_laser(self, pos, laser):
        if not (laser['type'] == 'diagonal' and self._is_laser_active(laser)):
            return False
        
        time_in_cycle = (laser['timer'] - laser['offset'] + laser['cycle_len']) % laser['cycle_len']
        path_len = self.GRID_SIZE * 2 - 1
        start_offset = laser['pos']
        
        if laser['orientation'] == 1: # top-left to bottom-right trend
            diag_index = (time_in_cycle + start_offset) % path_len
            return pos[0] + pos[1] == diag_index
        else: # top-right to bottom-left trend
            diag_index = (time_in_cycle + start_offset) % path_len
            mapped_index = diag_index - (self.GRID_SIZE - 1)
            return pos[0] - pos[1] == mapped_index

    def _is_pos_in_laser(self, pos):
        for laser in self.lasers:
            if self._is_laser_active(laser):
                if laser['type'] == 'horizontal' and pos[1] == laser['pos']:
                    return True
                if laser['type'] == 'vertical' and pos[0] == laser['pos']:
                    return True
                if self._is_pos_in_diagonal_laser(pos, laser):
                    return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.level}

    def _render_game(self):
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_X_OFFSET + i * self.CELL_SIZE
            y = self.GRID_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_DIM))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_DIM, y))

        for wx, wy in self.walls:
            rect = self._get_cell_rect(wx, wy)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        self._draw_glowing_square(self.exit_pos, self.COLOR_EXIT, self.COLOR_EXIT_GLOW)

        for laser in self.lasers: self._draw_laser(laser)
        
        # Only draw robot if not dead on a laser
        if not (self.game_over and self._is_pos_in_laser(self.robot_pos)):
             self._draw_glowing_square(self.robot_pos, self.COLOR_ROBOT, self.COLOR_ROBOT_GLOW)

        for p in self.particles: p.draw(self.screen)

    def _render_ui(self):
        level_text = self.font_large.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (20, 15))

        score_text = self.font_large.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 15))
        self.screen.blit(score_text, score_rect)
        
        moves_text = self.font_small.render(f"MOVES: {self.steps}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 45))
        self.screen.blit(moves_text, moves_rect)

    def _get_cell_rect(self, gx, gy, inset=0):
        return pygame.Rect(
            self.GRID_X_OFFSET + gx * self.CELL_SIZE + inset,
            self.GRID_Y_OFFSET + gy * self.CELL_SIZE + inset,
            self.CELL_SIZE - 2 * inset, self.CELL_SIZE - 2 * inset)

    def _get_cell_center(self, gx, gy):
        rect = self._get_cell_rect(gx, gy)
        return rect.center

    def _draw_glowing_square(self, pos, color, glow_color):
        rect = self._get_cell_rect(pos[0], pos[1])
        center = rect.center
        for i in range(4):
            s = self.CELL_SIZE * (1.2 + i * 0.2)
            alpha = 80 - i * 20
            temp_surf = pygame.Surface((s, s), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, (*glow_color, alpha), (0, 0, s, s), border_radius=int(s*0.2))
            self.screen.blit(temp_surf, (center[0] - s/2, center[1] - s/2), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.rect(self.screen, color, self._get_cell_rect(pos[0], pos[1], inset=4), border_radius=4)

    def _draw_laser(self, laser):
        is_active = self._is_laser_active(laser)
        time_until_active = (laser['offset'] - laser['timer'] + laser['cycle_len']) % laser['cycle_len']
        charge_level = 1.0 - min(1.0, time_until_active / (laser['cycle_len'] - laser['on_len'] + 1))
        
        emitter_color = self.COLOR_EMITTER.lerp(self.COLOR_EMITTER_CHARGE, charge_level)

        if laser['type'] == 'horizontal':
            emitter_pos = (self.GRID_X_OFFSET - self.CELL_SIZE / 2, self._get_cell_center(0, laser['pos'])[1])
            points = [(emitter_pos[0]-8, emitter_pos[1]), (emitter_pos[0]+8, emitter_pos[1]-8), (emitter_pos[0]+8, emitter_pos[1]+8)]
        elif laser['type'] == 'vertical':
            emitter_pos = (self._get_cell_center(laser['pos'], 0)[0], self.GRID_Y_OFFSET - self.CELL_SIZE / 2)
            points = [(emitter_pos[0], emitter_pos[1]-8), (emitter_pos[0]-8, emitter_pos[1]+8), (emitter_pos[0]+8, emitter_pos[1]+8)]
        else:
            emitter_pos = (self.GRID_X_OFFSET - self.CELL_SIZE / 2, self.GRID_Y_OFFSET - self.CELL_SIZE / 2)
            points = [(emitter_pos[0], emitter_pos[1]), (emitter_pos[0]+16, emitter_pos[1]), (emitter_pos[0], emitter_pos[1]+16)]
        
        pygame.gfxdraw.aapolygon(self.screen, points, emitter_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, emitter_color)
        
        if is_active:
            if laser['type'] == 'horizontal':
                y = self._get_cell_center(0, laser['pos'])[1]
                pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_DIM, y), 7)
                pygame.draw.line(self.screen, self.COLOR_LASER, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_DIM, y), 3)
            elif laser['type'] == 'vertical':
                x = self._get_cell_center(laser['pos'], 0)[0]
                pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_DIM), 7)
                pygame.draw.line(self.screen, self.COLOR_LASER, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_DIM), 3)
            else: # Diagonal
                for x in range(self.GRID_SIZE):
                    for y in range(self.GRID_SIZE):
                        if self._is_pos_in_diagonal_laser([x,y], laser):
                             self._draw_glowing_square([x, y], self.COLOR_LASER, self.COLOR_LASER_GLOW)

    def _create_particles(self, grid_pos, color, count):
        cx, cy = self._get_cell_rect(grid_pos[0], grid_pos[1]).center
        for _ in range(count):
            self.particles.append(Particle(cx, cy, color, life=random.randint(20, 40)))

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Laser Grid Robot")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    
    while True:
        action_taken = False
        mov_action = 0 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_UP: mov_action = 1
                elif event.key == pygame.K_DOWN: mov_action = 2
                elif event.key == pygame.K_LEFT: mov_action = 3
                elif event.key == pygame.K_RIGHT: mov_action = 4
                elif event.key == pygame.K_SPACE: mov_action = 0 # No-op
                else: action_taken = False
        
        if terminated:
            # On game over, wait for any key to reset
            if action_taken:
                obs, info = env.reset()
                terminated = False
        elif action_taken:
            current_action = [mov_action, 0, 0]
            obs, reward, terminated, truncated, info = env.step(current_action)
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Terminated: {terminated}")

        screen.blit(pygame.transform.scale(env.screen, screen.get_rect().size), (0, 0))
        pygame.display.flip()
        env.clock.tick(30)