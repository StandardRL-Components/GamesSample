
# Generated: 2025-08-27T23:55:31.479314
# Source Brief: brief_03624.md
# Brief Index: 3624

        
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
        "Controls: Arrow keys to move your character. Collect all the coins and reach the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze, collect coins, and reach the exit within 20 moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Game parameters
    MAZE_DIM_W, MAZE_DIM_H = 21, 13  # Must be odd numbers
    MAX_MOVES = 20
    NUM_COINS = 8
    
    # Screen and rendering
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PADDING = 20
    CELL_SIZE = min((SCREEN_WIDTH - 2 * PADDING) // MAZE_DIM_W, (SCREEN_HEIGHT - 2 * PADDING) // MAZE_DIM_H)
    MAZE_PX_WIDTH = CELL_SIZE * MAZE_DIM_W
    MAZE_PX_HEIGHT = CELL_SIZE * MAZE_DIM_H
    OFFSET_X = (SCREEN_WIDTH - MAZE_PX_WIDTH) // 2
    OFFSET_Y = (SCREEN_HEIGHT - MAZE_PX_HEIGHT) // 2

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_WALL = (60, 70, 80)
    COLOR_PATH = (30, 45, 60)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_COIN = (255, 215, 0)
    COLOR_EXIT = (0, 200, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    # Rewards
    REWARD_MOVE = -0.2
    REWARD_COIN = 1.0
    REWARD_EXIT = 10.0

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
        
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        self.rng = None
        
        # Game state variables are initialized in reset()
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.coins = None
        self.remaining_moves = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            random.seed(seed) # Also seed python's random for maze generation
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.remaining_moves = self.MAX_MOVES
        self.particles.clear()
        
        self.maze = self._generate_maze()
        self.player_pos = [1, 1]
        self.exit_pos = [self.MAZE_DIM_W - 2, self.MAZE_DIM_H - 2]
        
        self._place_coins()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        if movement != 0:  # 0 is no-op
            self.remaining_moves -= 1
            reward += self.REWARD_MOVE
            
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            next_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            
            if self.maze[next_pos[1], next_pos[0]] == 0:  # 0 is a path
                self.player_pos = next_pos
                # sfx: move.wav
            else:
                # sfx: bump_wall.wav
                pass # Hit a wall, move is spent but position doesn't change

        # Check for coin collection
        if self.player_pos in self.coins:
            self.coins.remove(self.player_pos)
            self.score += 1
            reward += self.REWARD_COIN
            self._spawn_particles(self.player_pos)
            # sfx: coin_collect.wav

        # Check termination conditions
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += self.REWARD_EXIT
            terminated = True
            self.game_over = True
            self.win = True
            # sfx: win.wav
        elif self.remaining_moves <= 0:
            terminated = True
            self.game_over = True
            # sfx: lose.wav
            
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _grid_to_pixel(self, grid_pos):
        x = self.OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _place_coins(self):
        self.coins = []
        possible_locs = np.argwhere(self.maze == 0).tolist()
        possible_locs = [[c, r] for r, c in possible_locs] # Convert to [x, y]

        if self.player_pos in possible_locs:
            possible_locs.remove(self.player_pos)
        if self.exit_pos in possible_locs:
            possible_locs.remove(self.exit_pos)
        
        num_to_place = min(self.NUM_COINS, len(possible_locs))
        
        if self.rng:
            indices = self.rng.choice(len(possible_locs), size=num_to_place, replace=False)
            self.coins = [possible_locs[i] for i in indices]
        else:
            self.coins = random.sample(possible_locs, num_to_place)

    def _generate_maze(self):
        w, h = self.MAZE_DIM_W, self.MAZE_DIM_H
        maze = np.ones((h, w), dtype=np.uint8)  # 1 = wall

        def carve(x, y):
            dirs = [(0, -2), (0, 2), (2, 0), (-2, 0)]
            random.shuffle(dirs)
            
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 < ny < h - 1 and 0 < nx < w - 1 and maze[ny, nx] == 1:
                    maze[ny, nx] = 0  # Carve path
                    maze[y + dy // 2, x + dx // 2] = 0  # Carve wall in between
                    carve(nx, ny)

        start_x, start_y = (1, 1)
        maze[start_y, start_x] = 0
        carve(start_x, start_y)
        return maze

    def _render_game(self):
        # Draw maze
        for r in range(self.MAZE_DIM_H):
            for c in range(self.MAZE_DIM_W):
                rect = pygame.Rect(
                    self.OFFSET_X + c * self.CELL_SIZE,
                    self.OFFSET_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                color = self.COLOR_WALL if self.maze[r, c] == 1 else self.COLOR_PATH
                pygame.draw.rect(self.screen, color, rect)
        
        # Draw exit
        exit_px, exit_py = self._grid_to_pixel(self.exit_pos)
        exit_rect = pygame.Rect(exit_px - self.CELL_SIZE // 2, exit_py - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)

        # Draw coins
        coin_radius = int(self.CELL_SIZE * 0.3)
        for coin_pos in self.coins:
            px, py = self._grid_to_pixel(coin_pos)
            pygame.gfxdraw.filled_circle(self.screen, px, py, coin_radius, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, px, py, coin_radius, self.COLOR_COIN)

        # Draw player
        player_px, player_py = self._grid_to_pixel(self.player_pos)
        player_radius = int(self.CELL_SIZE * 0.35)
        
        # Pulsing glow effect
        glow_alpha = (math.sin(self.steps * 0.2) * 0.5 + 0.5) * 60 + 20
        glow_radius = int(player_radius * (1.5 + math.sin(self.steps * 0.2) * 0.2))
        glow_color = (*self.COLOR_PLAYER, glow_alpha)
        
        # Create a temporary surface for the glow
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (player_px - glow_radius, player_py - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Player circle
        pygame.gfxdraw.filled_circle(self.screen, player_px, player_py, player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_px, player_py, player_radius, self.COLOR_PLAYER)

    def _render_text(self, text, font, color, pos, shadow=True, center=False):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        if shadow:
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        self._render_text(f"Moves: {self.remaining_moves}", self.font_ui, self.COLOR_TEXT, (15, 10))
        self._render_text(f"Score: {self.score}", self.font_ui, self.COLOR_TEXT, (self.SCREEN_WIDTH - 115, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_EXIT if self.win else (200, 50, 50)
            self._render_text(message, self.font_game_over, color, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), center=True)

    def _spawn_particles(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(10, 25)
            radius = random.uniform(2, 5)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'radius': radius, 'max_life': life})

    def _update_and_render_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
                
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*self.COLOR_COIN, alpha)
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                radius = int(p['radius'] * (p['life'] / p['max_life']))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        self.particles = active_particles

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
        assert self.remaining_moves > 0
        assert self.player_pos == [1,1]
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test game logic assertions
        self.reset()
        assert self.remaining_moves == self.MAX_MOVES
        self.step([1, 0, 0]) # Move up
        assert self.remaining_moves == self.MAX_MOVES - 1
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Runner")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not terminated:
        action = 0 # No-op by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
                elif event.key == pygame.K_RIGHT:
                    action = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    action = 0
                elif event.key == pygame.K_q:
                    terminated = True

        # Only step if an action was taken
        if action != 0:
            obs, reward, term, trunc, info = env.step([action, 0, 0])
            terminated = term
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {env.remaining_moves}, Terminated: {term}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate
        
    print("Game Over!")
    pygame.quit()