
# Generated: 2025-08-27T20:06:37.614871
# Source Brief: brief_02352.md
# Brief Index: 2352

        
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
        "Controls: Use arrow keys to move. Avoid red traps and reach the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a deadly maze to find the exit before the timer runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Maze Dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAZE_WIDTH, self.MAZE_HEIGHT = 31, 19 # Must be odd numbers
        self.TILE_SIZE = 20
        self.OFFSET_X = (self.SCREEN_WIDTH - self.MAZE_WIDTH * self.TILE_SIZE) // 2
        self.OFFSET_Y = (self.SCREEN_HEIGHT - self.MAZE_HEIGHT * self.TILE_SIZE) // 2

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (50, 50, 70)
        self.COLOR_FLOOR = self.COLOR_BG
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_EXIT = (0, 255, 150)
        self.COLOR_TRAP = (255, 50, 100)
        self.COLOR_TEXT = (220, 220, 240)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Game State Attributes
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.traps = None
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.particles = []
        self.np_random = None
        
        self.reset()
        self.validate_implementation()

    def _generate_maze(self):
        # Create a grid full of walls
        grid = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.uint8)
        
        # Recursive backtracking function
        def _carve(x, y):
            directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
            self.np_random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= ny < self.MAZE_HEIGHT and 0 <= nx < self.MAZE_WIDTH and grid[ny, nx] == 1:
                    grid[ny, nx] = 0
                    grid[y + dy // 2, x + dx // 2] = 0
                    _carve(nx, ny)

        # Start carving from a random odd position
        start_x, start_y = (1, 1)
        grid[start_y, start_x] = 0
        _carve(start_x, start_y)
        
        # Set player start and exit
        player_start_pos = (1, 1)
        exit_pos = (self.MAZE_WIDTH - 2, self.MAZE_HEIGHT - 2)
        grid[exit_pos[1], exit_pos[0]] = 0 # Ensure exit is open

        # Place traps
        num_traps = self.np_random.integers(15, 25)
        traps = set()
        while len(traps) < num_traps:
            tx = self.np_random.integers(1, self.MAZE_WIDTH - 1)
            ty = self.np_random.integers(1, self.MAZE_HEIGHT - 1)
            if grid[ty, tx] == 0 and (tx, ty) != player_start_pos and (tx, ty) != exit_pos:
                traps.add((tx, ty))
        
        return grid, list(traps), player_start_pos, exit_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.np_random = np.random.default_rng(seed)
        self.maze, self.traps, self.player_pos, self.exit_pos = self._generate_maze()
        
        self.steps = 0
        self.score = 0
        self.time_left = 60
        self.game_over = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        self.steps += 1
        self.time_left -= 1
        
        reward = -0.1
        terminated = False

        # --- Player Movement ---
        px, py = self.player_pos
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        # Check for wall collision
        if self.maze[py, px] == 0:
            self.player_pos = (px, py)

        # --- Check Game Events ---
        if self.player_pos in self.traps:
            # Sound: Explosion or zap
            reward = -50
            self.game_over = True
            terminated = True
            self._create_particles(self.player_pos, self.COLOR_TRAP, 50)
        
        elif self.player_pos == self.exit_pos:
            # Sound: Success chime
            reward = 100
            self.game_over = True
            terminated = True
            self._create_particles(self.player_pos, self.COLOR_EXIT, 50)

        elif self.time_left <= 0:
            # Sound: Timeout buzzer
            reward = -10 # Small penalty for timeout
            self.game_over = True
            terminated = True

        self.score += reward
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _create_particles(self, pos, color, count):
        cx = self.OFFSET_X + pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2
        cy = self.OFFSET_Y + pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': [cx, cy], 'vel': vel, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw maze walls
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(self.OFFSET_X + x * self.TILE_SIZE,
                                       self.OFFSET_Y + y * self.TILE_SIZE,
                                       self.TILE_SIZE, self.TILE_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # Draw traps
        trap_radius_base = self.TILE_SIZE * 0.25
        pulsation = math.sin(self.steps * 0.2) * self.TILE_SIZE * 0.1
        trap_radius = int(trap_radius_base + pulsation)
        for x, y in self.traps:
            cx = int(self.OFFSET_X + x * self.TILE_SIZE + self.TILE_SIZE / 2)
            cy = int(self.OFFSET_Y + y * self.TILE_SIZE + self.TILE_SIZE / 2)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, trap_radius, self.COLOR_TRAP)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, trap_radius, self.COLOR_TRAP)

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(self.OFFSET_X + ex * self.TILE_SIZE,
                                self.OFFSET_Y + ey * self.TILE_SIZE,
                                self.TILE_SIZE, self.TILE_SIZE)
        
        # Exit glow effect
        glow_size = int(self.TILE_SIZE * (1.5 + math.sin(self.steps * 0.1) * 0.4))
        glow_color = (*self.COLOR_EXIT, 20)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(glow_surf, (exit_rect.centerx - glow_size // 2, exit_rect.centery - glow_size // 2))

        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=3)
        
        # Draw player
        px, py = self.player_pos
        player_rect = pygame.Rect(self.OFFSET_X + px * self.TILE_SIZE + 2,
                                  self.OFFSET_Y + py * self.TILE_SIZE + 2,
                                  self.TILE_SIZE - 4, self.TILE_SIZE - 4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['life'] / p['max_life'] * 3)
            if radius > 0:
                # Simple rect for performance, could be circles
                pygame.draw.rect(self.screen, color, (*pos, radius, radius))

    def _render_ui(self):
        time_text = self.font_large.render(f"{self.time_left}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(time_text, time_rect)

        timer_label = self.font_small.render("TIME", True, self.COLOR_TEXT)
        timer_label_rect = timer_label.get_rect(midbottom=time_rect.midtop)
        self.screen.blit(timer_label, timer_label_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "player_pos": self.player_pos,
        }

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

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                print("--- Game Reset ---")

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Since this is turn-based, we only take one action per frame
            # This manual play loop is different from an agent's loop
            # We'll map the first key press we find to an action
            
            move = 0 # no-op
            if keys[pygame.K_UP]:
                move = 1
            elif keys[pygame.K_DOWN]:
                move = 2
            elif keys[pygame.K_LEFT]:
                move = 3
            elif keys[pygame.K_RIGHT]:
                move = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [move, space, shift]
            
            # In a turn-based game, we only step if a move is made
            if move != 0:
                obs, reward, terminated, _, info = env.step(action)
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.1f}, Score: {info['score']:.1f}, Terminated: {terminated}")
        
        # Drawing
        draw_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for manual play

    env.close()