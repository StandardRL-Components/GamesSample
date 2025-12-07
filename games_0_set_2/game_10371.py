import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:21:52.902461
# Source Brief: brief_00371.md
# Brief Index: 371
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Navigate a shifting maze as a water droplet, collecting puddles to grow larger
    while avoiding maze walls to reach the exit.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Navigate a shifting maze as a water droplet. Collect puddles to grow larger "
        "and avoid walls to reach the exit before you evaporate."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move your water droplet through the maze."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)

        # --- Game Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.CELL_SIZE = self.WIDTH // self.GRID_WIDTH
        self.INITIAL_VOLUME = 100.0
        self.MAX_VOLUME = 200.0
        self.WIN_THRESHOLD = 0.50 
        self.MAX_STEPS = 1000
        self.WALL_OSCILLATION_PERIOD = 20 # Slower for better visual tracking
        self.PUDDLE_RESPAWN_INTERVAL = 20

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (100, 110, 120)
        self.COLOR_DROPLET = (0, 170, 255)
        self.COLOR_DROPLET_GLOW = (0, 170, 255, 50)
        self.COLOR_PUDDLE = (136, 204, 255)
        self.COLOR_EXIT = (100, 255, 100)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BAR_BG = (50, 60, 70)
        self.COLOR_UI_BAR_FILL = (0, 170, 255)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.droplet_pos = None
        self.droplet_volume = 0.0
        self.droplet_radius = 0.0
        self.droplet_visual_radius = 0.0 # For smooth animations
        self.walls = set()
        self.oscillating_walls = []
        self.puddles = set()
        self.exit_pos = None
        self.puddle_respawn_timer = 0
        self.particles = []
        self.ui_volume_bar_width = 0.0

        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.droplet_pos = np.array([1, 1], dtype=int)
        self.droplet_volume = self.INITIAL_VOLUME
        self.exit_pos = np.array([self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2], dtype=int)

        self._generate_maze()

        self.puddles = set()
        self.puddle_respawn_timer = self.PUDDLE_RESPAWN_INTERVAL
        self._spawn_puddles()

        self.droplet_radius = self._calculate_radius()
        self.droplet_visual_radius = self.droplet_radius
        self.ui_volume_bar_width = self.droplet_volume / self.MAX_VOLUME
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held and shift_held are ignored per the brief
        
        self.steps += 1
        reward = 0.0
        
        old_volume = self.droplet_volume
        
        # 1. Update Game Logic
        self._update_player(movement)
        
        puddle_collected = self._check_puddle_collection()
        if puddle_collected:
            # Sound: *bloop*
            self._spawn_particles(self.droplet_pos, self.COLOR_PUDDLE)

        self._update_puddles_timer()
        self._update_walls()
        
        # 2. Calculate Reward
        reward += self._calculate_reward(old_volume, puddle_collected)
        self.score += reward
        
        # 3. Check Termination
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self._is_win_condition_met():
                # Sound: *success chime*
                reward += 100.0
                self.score += 100.0
                self._spawn_particles(self.droplet_pos, self.COLOR_EXIT, 50)
        
        # 4. Update visual elements smoothly
        self.droplet_radius = self._calculate_radius()
        self.droplet_visual_radius += (self.droplet_radius - self.droplet_visual_radius) * 0.2
        target_bar_width = self.droplet_volume / self.MAX_VOLUME
        self.ui_volume_bar_width += (target_bar_width - self.ui_volume_bar_width) * 0.2

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _calculate_radius(self):
        return self.CELL_SIZE * 0.45 * math.sqrt(self.droplet_volume / self.INITIAL_VOLUME)

    def _generate_maze(self):
        self.walls = set()
        for x in range(self.GRID_WIDTH):
            self.walls.add((x, 0))
            self.walls.add((x, self.GRID_HEIGHT - 1))
        for y in range(self.GRID_HEIGHT):
            self.walls.add((0, y))
            self.walls.add((self.GRID_WIDTH - 1, y))

        # Randomized DFS for maze generation
        stack = [(1, 1)]
        visited = set([(1, 1)])
        path_cells = set([(1,1)])

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.GRID_WIDTH -1 and 0 < ny < self.GRID_HEIGHT -1 and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.np_random.choice([tuple(n) for n in neighbors])
                wx, wy = (cx + nx) // 2, (cy + ny) // 2
                path_cells.add((wx, wy))
                path_cells.add((nx, ny))
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        
        for x in range(1, self.GRID_WIDTH - 1):
            for y in range(1, self.GRID_HEIGHT - 1):
                if (x, y) not in path_cells:
                    self.walls.add((x, y))

        # Ensure exit is clear
        if tuple(self.exit_pos) in self.walls: self.walls.remove(tuple(self.exit_pos))
        if (self.exit_pos[0]-1, self.exit_pos[1]) in self.walls: self.walls.remove((self.exit_pos[0]-1, self.exit_pos[1]))
        if (self.exit_pos[0], self.exit_pos[1]-1) in self.walls: self.walls.remove((self.exit_pos[0], self.exit_pos[1]-1))

        # Designate some walls as oscillating
        self.oscillating_walls = []
        potential_osc_walls = []
        for (wx, wy) in self.walls:
            if 1 < wx < self.GRID_WIDTH - 2 and 1 < wy < self.GRID_HEIGHT - 2:
                # Check for horizontal segment
                if (wx - 1, wy) in self.walls and (wx + 1, wy) in self.walls:
                    potential_osc_walls.append(((wx, wy), 'vertical'))
                # Check for vertical segment
                if (wx, wy - 1) in self.walls and (wx, wy + 1) in self.walls:
                    potential_osc_walls.append(((wx, wy), 'horizontal'))
        
        if potential_osc_walls:
            num_osc_walls = min(len(potential_osc_walls), 5)
            indices = self.np_random.choice(len(potential_osc_walls), size=num_osc_walls, replace=False)
            self.oscillating_walls = [potential_osc_walls[i] for i in indices]

    def _update_player(self, movement):
        # Movement cost: higher for larger droplets
        move_cost = 0.05 * (self.droplet_volume / self.INITIAL_VOLUME)
        self.droplet_volume = max(0, self.droplet_volume - move_cost)

        if movement == 0: # no-op
            return

        prev_pos = self.droplet_pos.copy()
        
        if movement == 1: # Up
            self.droplet_pos[1] -= 1
        elif movement == 2: # Down
            self.droplet_pos[1] += 1
        elif movement == 3: # Left
            self.droplet_pos[0] -= 1
        elif movement == 4: # Right
            self.droplet_pos[0] += 1

        # Check for wall collision
        if self._is_wall_at(tuple(self.droplet_pos)):
            # Sound: *thud*
            self.droplet_volume = max(0, self.droplet_volume - 15.0)
            self.droplet_pos = prev_pos # Revert move
            self._spawn_particles(self.droplet_pos, self.COLOR_WALL, 5, speed=1)

    def _is_wall_at(self, pos):
        # Check static walls
        if pos in self.walls:
            return True
        # Check oscillating walls
        for (base_pos, axis), offset in self._get_current_wall_offsets().items():
            current_pos = (base_pos[0] + offset[0], base_pos[1] + offset[1])
            if pos == current_pos:
                return True
        return False

    def _get_current_wall_offsets(self):
        offsets = {}
        for base_pos, axis in self.oscillating_walls:
            oscillation = math.sin(2 * math.pi * self.steps / self.WALL_OSCILLATION_PERIOD)
            offset_val = round(oscillation) # Snaps to -1, 0, 1
            if axis == 'horizontal':
                offsets[(base_pos, axis)] = (offset_val, 0)
            else: # vertical
                offsets[(base_pos, axis)] = (0, offset_val)
        return offsets

    def _check_puddle_collection(self):
        if tuple(self.droplet_pos) in self.puddles:
            self.puddles.remove(tuple(self.droplet_pos))
            self.droplet_volume = min(self.MAX_VOLUME, self.droplet_volume + 20.0)
            return True
        return False

    def _update_puddles_timer(self):
        self.puddle_respawn_timer -= 1
        if self.puddle_respawn_timer <= 0:
            self.puddle_respawn_timer = self.PUDDLE_RESPAWN_INTERVAL
            self._spawn_puddles()

    def _spawn_puddles(self):
        self.puddles.clear()
        num_puddles = 3
        for _ in range(num_puddles):
            while True:
                pos = (self.np_random.integers(1, self.GRID_WIDTH - 1), self.np_random.integers(1, self.GRID_HEIGHT - 1))
                if not self._is_wall_at(pos) and pos not in self.puddles and pos != tuple(self.droplet_pos) and pos != tuple(self.exit_pos):
                    self.puddles.add(pos)
                    break

    def _update_walls(self):
        # The positions are calculated on-the-fly in _is_wall_at and _render_walls
        pass

    def _calculate_reward(self, old_volume, puddle_collected):
        reward = 0.0
        if puddle_collected:
            reward += 10.0
        
        if self.droplet_volume < old_volume and not puddle_collected:
            reward -= 0.1 # Penalty for losing volume due to movement or collision

        return reward

    def _is_win_condition_met(self):
        return tuple(self.droplet_pos) == tuple(self.exit_pos) and \
               self.droplet_volume >= self.INITIAL_VOLUME * self.WIN_THRESHOLD

    def _check_termination(self):
        if self._is_win_condition_met():
            return True
        if self.droplet_volume <= 0:
            return True
        if tuple(self.droplet_pos) == tuple(self.exit_pos): # Reached exit but not enough water
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
            "volume": self.droplet_volume,
            "position": self.droplet_pos,
        }

    def _render_game(self):
        self._render_walls()
        self._render_puddles()
        self._render_exit()
        self._render_particles()
        self._render_droplet()

    def _render_walls(self):
        # Static walls
        for (x, y) in self.walls:
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Oscillating walls
        for (base_pos, axis), offset in self._get_current_wall_offsets().items():
            pos_x = (base_pos[0] + offset[0]) * self.CELL_SIZE
            pos_y = (base_pos[1] + offset[1]) * self.CELL_SIZE
            rect = pygame.Rect(pos_x, pos_y, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_WALL, rect)

    def _render_puddles(self):
        for (x, y) in self.puddles:
            cx = int((x + 0.5) * self.CELL_SIZE)
            cy = int((y + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.3)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_PUDDLE)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_PUDDLE)

    def _render_exit(self):
        x, y = self.exit_pos
        rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, rect)

    def _render_droplet(self):
        cx = int((self.droplet_pos[0] + 0.5) * self.CELL_SIZE)
        cy = int((self.droplet_pos[1] + 0.5) * self.CELL_SIZE)
        
        # Wobble effect
        wobble = 1 + 0.05 * math.sin(self.steps * 0.5)
        radius = int(self.droplet_visual_radius * wobble)
        
        # Glow
        glow_radius = int(radius * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_DROPLET_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (cx - glow_radius, cy - glow_radius))

        # Main droplet
        if radius > 0:
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_DROPLET)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_DROPLET)

    def _spawn_particles(self, grid_pos, color, count=15, speed=2):
        px = (grid_pos[0] + 0.5) * self.CELL_SIZE
        py = (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = self.np_random.uniform(0.5, 1.0) * speed
            vx = math.cos(angle) * vel
            vy = math.sin(angle) * vel
            life = self.np_random.integers(15, 31)
            self.particles.append([px, py, vx, vy, life, color])

    def _render_particles(self):
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1
            radius = int(p[4] / 6)
            if radius > 0:
                pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), radius)
        self.particles = [p for p in self.particles if p[4] > 0]

    def _render_ui(self):
        # Volume Bar
        bar_x, bar_y, bar_w, bar_h = 10, 10, 200, 20
        fill_w = bar_w * self.ui_volume_bar_width
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, fill_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)
        
        volume_text = f"Volume: {int(self.droplet_volume)}%"
        text_surf = self.font_small.render(volume_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (bar_x + 5, bar_y + 3))

        # Score and Steps
        score_text = f"Score: {self.score:.1f}"
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        score_surf = self.font.render(score_text, True, self.COLOR_UI_TEXT)
        steps_surf = self.font.render(steps_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))
        self.screen.blit(steps_surf, (self.WIDTH - steps_surf.get_width() - 10, 35))

        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            if self._is_win_condition_met():
                msg = "SUCCESS!"
                color = self.COLOR_EXIT
            else:
                msg = "GAME OVER"
                color = self.COLOR_WALL
            
            end_text_surf = pygame.font.SysFont("monospace", 60, bold=True).render(msg, True, color)
            text_rect = end_text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text_surf, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It has been modified to use the correct render mode for display
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.quit() # Quit the dummy instance
    pygame.init() # Re-init with a display

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Water Droplet Maze")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Re-render the initial state to the new screen
    frame = np.transpose(obs, (1, 0, 2))
    surf = pygame.surfarray.make_surface(frame)
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward}")
            print("Press 'R' to reset.")

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Control manual play speed
        
    env.close()