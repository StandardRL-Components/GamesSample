
# Generated: 2025-08-27T15:39:23.935409
# Source Brief: brief_01040.md
# Brief Index: 1040

        
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
        "Controls: Use arrow keys to move your avatar tile by tile. Avoid the red mines!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated minefield maze, collecting gems for points while avoiding explosive mines."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.MAZE_WIDTH = self.WIDTH // self.GRID_SIZE
        self.MAZE_HEIGHT = self.HEIGHT // self.GRID_SIZE
        self.MAX_STAGES = 3
        self.NUM_GEMS_PER_STAGE = 15
        self.INITIAL_MINES = 5
        self.MAX_TIME_PER_STAGE = 60 * 10 # 60 seconds at 10 steps/sec
        self.WALL_DENSITY_INCREASE = 0.1 # This is conceptual, implemented via maze generation complexity

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_WALL = (60, 70, 80)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_GEM = (255, 220, 0)
        self.COLOR_GEM_GLOW = (255, 255, 150)
        self.COLOR_MINE = (255, 50, 50)
        self.COLOR_MINE_GLOW = (255, 150, 150)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_TIMER_WARN = (255, 100, 100)

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # --- State Variables ---
        self.player_pos = None
        self.gems = None
        self.mines = None
        self.maze = None
        self.particles = None
        self.steps = None
        self.total_score = None
        self.stage_score = None
        self.time_remaining = None
        self.current_stage = None
        self.game_over = None
        self.game_won = None

        self.reset()
        self.validate_implementation()

    def _generate_maze(self):
        w, h = self.MAZE_WIDTH, self.MAZE_HEIGHT
        # Create a grid where each cell has all four walls up
        maze = [[{'visited': False, 'walls': [True, True, True, True]} for _ in range(h)] for _ in range(w)] # T, R, B, L

        def get_neighbors(x, y):
            neighbors = []
            if x > 0 and not maze[x-1][y]['visited']: neighbors.append((x-1, y, 3, 1)) # Left
            if x < w-1 and not maze[x+1][y]['visited']: neighbors.append((x+1, y, 1, 3)) # Right
            if y > 0 and not maze[x][y-1]['visited']: neighbors.append((x, y-1, 0, 2)) # Top
            if y < h-1 and not maze[x][y+1]['visited']: neighbors.append((x, y+1, 2, 0)) # Bottom
            return neighbors

        stack = []
        # Start from a random cell
        start_x, start_y = self.np_random.integers(0, w), self.np_random.integers(0, h)
        maze[start_x][start_y]['visited'] = True
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = get_neighbors(cx, cy)

            if neighbors:
                nx, ny, wall_to_break, opposite_wall = self.np_random.choice(neighbors, axis=0)
                maze[cx][cy]['walls'][wall_to_break] = False
                maze[nx][ny]['walls'][opposite_wall] = False
                maze[nx][ny]['visited'] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _setup_stage(self):
        self.stage_score = 0
        self.time_remaining = self.MAX_TIME_PER_STAGE
        self.game_over = False
        self.game_won = False
        self.particles = []

        self.maze = self._generate_maze()

        valid_positions = []
        for x in range(self.MAZE_WIDTH):
            for y in range(self.MAZE_HEIGHT):
                valid_positions.append((x, y))

        self.np_random.shuffle(valid_positions)

        self.player_pos = valid_positions.pop()
        
        num_gems = self.NUM_GEMS_PER_STAGE
        self.gems = [valid_positions.pop() for _ in range(num_gems)]

        num_mines = self.INITIAL_MINES + (self.current_stage - 1)
        self.mines = [valid_positions.pop() for _ in range(num_mines)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.total_score = 0
        self.current_stage = 1
        self._setup_stage()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        reward = -0.1  # Cost of existing
        self.steps += 1
        self.time_remaining -= 1

        # --- Player Movement ---
        px, py = self.player_pos
        next_px, next_py = px, py
        moved = False

        if movement == 1 and py > 0: # Up
            if not self.maze[px][py]['walls'][0]:
                next_py -= 1
                moved = True
            else: reward -= 0.2
        elif movement == 2 and py < self.MAZE_HEIGHT - 1: # Down
            if not self.maze[px][py]['walls'][2]:
                next_py += 1
                moved = True
            else: reward -= 0.2
        elif movement == 3 and px > 0: # Left
            if not self.maze[px][py]['walls'][3]:
                next_px -= 1
                moved = True
            else: reward -= 0.2
        elif movement == 4 and px < self.MAZE_WIDTH - 1: # Right
            if not self.maze[px][py]['walls'][1]:
                next_px += 1
                moved = True
            else: reward -= 0.2
        
        if moved:
            self.player_pos = (next_px, next_py)

        # --- Collision Detection & State Updates ---
        # Gem collection
        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            self.stage_score += 10
            reward += 10
            # sfx: gem_collect.wav

        # Mine collision
        if self.player_pos in self.mines:
            self.stage_score -= 100
            reward -= 100
            self.game_over = True
            self._create_explosion(self.player_pos)
            # sfx: explosion.wav

        # Stage clear
        if not self.gems:
            self.stage_score += 50
            reward += 50
            self.total_score += self.stage_score
            if self.current_stage < self.MAX_STAGES:
                self.current_stage += 1
                self._setup_stage()
                # sfx: stage_clear.wav
            else:
                self.game_over = True
                self.game_won = True
                # sfx: game_win.wav

        # Time out
        if self.time_remaining <= 0:
            self.game_over = True
            # sfx: time_out.wav
        
        self._update_particles()

        terminated = self.game_over
        if terminated:
            self.total_score += self.stage_score

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_explosion(self, pos):
        cx = (pos[0] + 0.5) * self.GRID_SIZE
        cy = (pos[1] + 0.5) * self.GRID_SIZE
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(2, 5)
            color = random.choice([self.COLOR_MINE, self.COLOR_MINE_GLOW, (255, 180, 50)])
            self.particles.append({
                'x': cx, 'y': cy,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': life, 'max_life': life, 'radius': radius, 'color': color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.total_score + self.stage_score,
            "stage": self.current_stage,
            "gems_left": len(self.gems),
            "time_left": self.time_remaining,
            "steps": self.steps,
        }

    def _render_game(self):
        # Draw maze walls
        for x in range(self.MAZE_WIDTH):
            for y in range(self.MAZE_HEIGHT):
                cell = self.maze[x][y]
                px, py = x * self.GRID_SIZE, y * self.GRID_SIZE
                if cell['walls'][0]: pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.GRID_SIZE, py), 2)
                if cell['walls'][1]: pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.GRID_SIZE, py), (px + self.GRID_SIZE, py + self.GRID_SIZE), 2)
                if cell['walls'][2]: pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.GRID_SIZE, py + self.GRID_SIZE), (px, py + self.GRID_SIZE), 2)
                if cell['walls'][3]: pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.GRID_SIZE), (px, py), 2)

        # Draw gems
        for gx, gy in self.gems:
            rect = pygame.Rect(gx * self.GRID_SIZE + 4, gy * self.GRID_SIZE + 4, self.GRID_SIZE - 8, self.GRID_SIZE - 8)
            pygame.draw.rect(self.screen, self.COLOR_GEM, rect, border_radius=2)
            pygame.draw.rect(self.screen, self.COLOR_GEM_GLOW, rect, width=1, border_radius=2)

        # Draw mines
        for mx, my in self.mines:
            points = [
                (mx * self.GRID_SIZE + self.GRID_SIZE // 2, my * self.GRID_SIZE + 3),
                (mx * self.GRID_SIZE + 3, my * self.GRID_SIZE + self.GRID_SIZE - 3),
                (mx * self.GRID_SIZE + self.GRID_SIZE - 3, my * self.GRID_SIZE + self.GRID_SIZE - 3)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_MINE_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_MINE)

        # Draw player
        px, py = self.player_pos
        center_x = int(px * self.GRID_SIZE + self.GRID_SIZE / 2)
        center_y = int(py * self.GRID_SIZE + self.GRID_SIZE / 2)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.GRID_SIZE // 2 - 2, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.GRID_SIZE // 2 - 2, self.COLOR_PLAYER_GLOW)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (p['x'] - p['radius'], p['y'] - p['radius']))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.total_score + self.stage_score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_ui.render(f"STAGE: {self.current_stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(centerx=self.WIDTH // 2)
        stage_rect.top = 10
        self.screen.blit(stage_text, stage_rect)

        # Timer
        time_color = self.COLOR_TEXT if self.time_remaining > 100 else self.COLOR_TIMER_WARN
        time_str = f"TIME: {self.time_remaining // 10:02d}"
        time_text = self.font_ui.render(time_str, True, time_color)
        time_rect = time_text.get_rect(right=self.WIDTH - 10)
        time_rect.top = 10
        self.screen.blit(time_text, time_rect)

        # Gems Remaining
        gems_text = self.font_ui.render(f"GEMS: {len(self.gems)}", True, self.COLOR_TEXT)
        self.screen.blit(gems_text, (10, self.HEIGHT - 30))
        
        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            
            end_text = self.font_game_over.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

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

# Example usage to test the environment visually
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up a display window
    pygame.display.set_caption("Minefield Maze")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        if terminated:
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
                total_reward = 0
        else:
            # Only step if an action is taken, because auto_advance is False
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}, Info: {info}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Limit FPS for human playability

    env.close()