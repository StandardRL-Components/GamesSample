
# Generated: 2025-08-27T18:04:09.622075
# Source Brief: brief_01720.md
# Brief Index: 1720

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Reach the green exit before the timer runs out. Avoid the red lasers!"
    )

    game_description = (
        "A fast-paced, top-down arcade game. Navigate a procedurally generated "
        "laser maze to reach the exit. Lasers rotate and speed up over time, "
        "creating a tense challenge of pathfinding and timing."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    CELL_SIZE = 20
    
    # Colors (Neon Aesthetic)
    COLOR_BG = (5, 10, 20)
    COLOR_WALL = (50, 80, 200)
    COLOR_WALL_GLOW = (30, 50, 150)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (200, 200, 50, 100)
    COLOR_EXIT = (0, 255, 100)
    COLOR_EXIT_GLOW = (50, 200, 100, 150)
    COLOR_LASER = (255, 20, 20)
    COLOR_LASER_GLOW = (200, 20, 20)
    COLOR_UI_TEXT = (220, 220, 255)
    
    # Game Parameters
    INITIAL_TIME_SECONDS = 30
    FPS = 30
    MAX_STEPS = 1000
    NUM_LASERS = 10

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
        self.font_ui = pygame.font.Font(None, 32)
        
        self.maze = []
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.lasers = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.initial_time_steps = self.INITIAL_TIME_SECONDS * self.FPS
        self.time_remaining = self.initial_time_steps
        self.laser_speed_modifier = 1.0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.initial_time_steps
        self.laser_speed_modifier = 1.0
        
        self._generate_maze()
        self._place_lasers()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, subsequent steps do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Survival reward
        terminated = False
        
        # --- Game Logic Update ---
        self._update_time()
        self._update_difficulty()
        self._update_lasers()
        self._handle_player_movement(action)

        # --- Check Game State ---
        if self._check_exit_condition():
            # SFX: Victory chime
            win_reward = 10.0 + 50.0 * (self.time_remaining / self.initial_time_steps)
            reward += win_reward
            self.score += reward - 0.1 # Adjust for initial survival reward
            terminated = True
        elif self._check_laser_collision():
            # SFX: Player hit/zap
            reward = -10.0
            self.score += reward
            terminated = True
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            # SFX: Game over sound
            terminated = True
        
        if not terminated:
            self.score += reward

        self.steps += 1
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_time(self):
        self.time_remaining = max(0, self.time_remaining - 1)

    def _update_difficulty(self):
        if self.steps > 0:
            # Change laser patterns
            if self.steps % 100 == 0:
                for laser in self.lasers:
                    if self.np_random.random() < 0.2: # 20% chance to reverse
                        laser['speed'] *= -1
            # Increase laser speed
            if self.steps % 200 == 0:
                self.laser_speed_modifier += 0.05

    def _update_lasers(self):
        for laser in self.lasers:
            laser['angle'] = (laser['angle'] + laser['speed'] * self.laser_speed_modifier) % (2 * math.pi)

    def _handle_player_movement(self, action):
        movement = action[0]
        px, py = self.player_pos
        gx, gy = int(px / self.CELL_SIZE), int(py / self.CELL_SIZE)

        # SFX: Player move (subtle tick)
        if movement == 1 and not self.maze[gy][gx]['walls'][0]: # Up
            py -= self.CELL_SIZE
        elif movement == 2 and not self.maze[gy][gx]['walls'][2]: # Down
            py += self.CELL_SIZE
        elif movement == 3 and not self.maze[gy][gx]['walls'][3]: # Left
            px -= self.CELL_SIZE
        elif movement == 4 and not self.maze[gy][gx]['walls'][1]: # Right
            px += self.CELL_SIZE
        
        self.player_pos = (px, py)

    def _check_exit_condition(self):
        return self.player_pos == self.exit_pos

    def _check_laser_collision(self):
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.CELL_SIZE, self.CELL_SIZE)
        for laser in self.lasers:
            center_x, center_y = laser['pos']
            angle = laser['angle']
            length = laser['length']
            
            # Check both directions from center for two-sided lasers
            for direction in [-1, 1]:
                end_x = center_x + direction * length * math.cos(angle)
                end_y = center_y + direction * length * math.sin(angle)
                if player_rect.clipline((center_x, center_y), (end_x, end_y)):
                    return True
        return False

    def _generate_maze(self):
        w, h = self.GRID_WIDTH, self.GRID_HEIGHT
        self.maze = [[{'visited': False, 'walls': [True, True, True, True]} for _ in range(w)] for _ in range(h)]
        
        stack = []
        start_cell = (0, 0)
        current_cell = start_cell
        self.maze[current_cell[1]][current_cell[0]]['visited'] = True
        path_for_exit = [current_cell]
        stack.append(current_cell)

        while stack:
            current_cell = stack.pop()
            cx, cy = current_cell
            
            neighbors = []
            # Check Up, Right, Down, Left
            if cy > 0 and not self.maze[cy - 1][cx]['visited']: neighbors.append((cx, cy - 1))
            if cx < w - 1 and not self.maze[cy][cx + 1]['visited']: neighbors.append((cx + 1, cy))
            if cy < h - 1 and not self.maze[cy + 1][cx]['visited']: neighbors.append((cx, cy + 1))
            if cx > 0 and not self.maze[cy][cx - 1]['visited']: neighbors.append((cx, cy - 1))
            
            unvisited_neighbors = []
            for nx, ny in [(cx, cy-1), (cx+1, cy), (cx, cy+1), (cx-1, cy)]:
                if 0 <= nx < w and 0 <= ny < h and not self.maze[ny][nx]['visited']:
                    unvisited_neighbors.append((nx, ny))

            if unvisited_neighbors:
                stack.append(current_cell)
                next_cell = self.np_random.choice([i for i in range(len(unvisited_neighbors))])
                next_cell = unvisited_neighbors[next_cell]
                nx, ny = next_cell

                # Remove walls
                if nx == cx + 1: # Right
                    self.maze[cy][cx]['walls'][1] = False
                    self.maze[ny][nx]['walls'][3] = False
                elif nx == cx - 1: # Left
                    self.maze[cy][cx]['walls'][3] = False
                    self.maze[ny][nx]['walls'][1] = False
                elif ny == cy + 1: # Down
                    self.maze[cy][cx]['walls'][2] = False
                    self.maze[ny][nx]['walls'][0] = False
                elif ny == cy - 1: # Up
                    self.maze[cy][cx]['walls'][0] = False
                    self.maze[ny][nx]['walls'][2] = False
                
                self.maze[ny][nx]['visited'] = True
                path_for_exit.append(next_cell)
                stack.append(next_cell)

        self.player_pos = (start_cell[0] * self.CELL_SIZE, start_cell[1] * self.CELL_SIZE)
        exit_gx, exit_gy = path_for_exit[-1]
        self.exit_pos = (exit_gx * self.CELL_SIZE, exit_gy * self.CELL_SIZE)

    def _place_lasers(self):
        self.lasers = []
        dead_ends = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if sum(self.maze[y][x]['walls']) == 3:
                    # Don't place on start/end cells
                    if (x, y) != (self.player_pos[0]/self.CELL_SIZE, self.player_pos[1]/self.CELL_SIZE) and \
                       (x, y) != (self.exit_pos[0]/self.CELL_SIZE, self.exit_pos[1]/self.CELL_SIZE):
                        dead_ends.append((x, y))
        
        self.np_random.shuffle(dead_ends)
        
        for i in range(min(self.NUM_LASERS, len(dead_ends))):
            gx, gy = dead_ends[i]
            self.lasers.append({
                'pos': (gx * self.CELL_SIZE + self.CELL_SIZE / 2, gy * self.CELL_SIZE + self.CELL_SIZE / 2),
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'speed': self.np_random.uniform(0.01, 0.03) * self.np_random.choice([-1, 1]),
                'length': self.np_random.integers(2, 5) * self.CELL_SIZE
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw exit with glow
        exit_rect = pygame.Rect(self.exit_pos[0], self.exit_pos[1], self.CELL_SIZE, self.CELL_SIZE)
        glow_rect = exit_rect.inflate(self.CELL_SIZE * 0.5, self.CELL_SIZE * 0.5)
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, self.COLOR_EXIT_GLOW, shape_surf.get_rect(), border_radius=4)
        self.screen.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=4)
        
        # Draw maze walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                walls = self.maze[y][x]['walls']
                px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
                if walls[0]: # Top
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.CELL_SIZE, py), 2)
                if walls[1]: # Right
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.CELL_SIZE, py), (px + self.CELL_SIZE, py + self.CELL_SIZE), 2)
                if walls[2]: # Bottom
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.CELL_SIZE), (px + self.CELL_SIZE, py + self.CELL_SIZE), 2)
                if walls[3]: # Left
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.CELL_SIZE), 2)

        # Draw lasers with glow
        for laser in self.lasers:
            center_x, center_y = int(laser['pos'][0]), int(laser['pos'][1])
            angle = laser['angle']
            length = laser['length']
            
            for direction in [-1, 1]:
                end_x = center_x + direction * length * math.cos(angle)
                end_y = center_y + direction * length * math.sin(angle)
                
                # Glow effect
                pygame.draw.aaline(self.screen, self.COLOR_LASER_GLOW, (center_x, center_y), (end_x, end_y), 5)
                pygame.draw.aaline(self.screen, self.COLOR_LASER_GLOW, (center_x, center_y), (end_x, end_y), 3)
                # Core beam
                pygame.draw.line(self.screen, self.COLOR_LASER, (center_x, center_y), (end_x, end_y), 2)
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 4, self.COLOR_LASER)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 4, self.COLOR_LASER)

        # Draw player with glow
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.CELL_SIZE, self.CELL_SIZE)
        glow_rect = player_rect.inflate(self.CELL_SIZE * 0.4, self.CELL_SIZE * 0.4)
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, self.COLOR_PLAYER_GLOW, shape_surf.get_rect(), border_radius=3)
        self.screen.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
    def _render_ui(self):
        time_text = f"TIME: {self.time_remaining / self.FPS:.1f}"
        score_text = f"SCORE: {int(self.score)}"
        
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 15, 10))
        self.screen.blit(score_surf, (15, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining / self.FPS,
        }

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Requires pygame to be installed with display support
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Laser Maze")
    
    terminated = False
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        movement_action = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        # Construct the full action for the MultiDiscrete space
        action = [movement_action, 0, 0] # Space and Shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
    env.close()