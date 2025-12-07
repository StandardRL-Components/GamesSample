
# Generated: 2025-08-27T18:39:08.019200
# Source Brief: brief_01904.md
# Brief Index: 1904

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Collect 10 coins and reach the green exit before you run out of moves."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "A minimalist puzzle game. Navigate a procedurally generated maze, "
        "collect coins, and find the exit within a limited number of moves. "
        "Each move counts, so plan your path carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.maze_width = 21
        self.maze_height = 15
        self.cell_size = 20
        self.max_moves = 50
        self.coin_requirement = 10
        self.num_coins_to_spawn = 15
        self.max_steps = 500

        self.maze_pixel_width = self.maze_width * self.cell_size
        self.maze_pixel_height = self.maze_height * self.cell_size
        self.maze_offset_x = (self.screen_width - self.maze_pixel_width) // 2
        self.maze_offset_y = (self.screen_height - self.maze_pixel_height) // 2

        # Colors
        self.color_bg = (20, 30, 40)
        self.color_wall = (80, 90, 100)
        self.color_player = (255, 220, 0)
        self.color_player_glow = (255, 220, 0, 50)
        self.color_coin = (255, 190, 0)
        self.color_exit = (0, 255, 120)
        self.color_text = (220, 220, 230)
        
        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_title = pygame.font.SysFont("Verdana", 28, bold=True)

        # Initialize state variables (will be properly set in reset)
        self.maze = []
        self.player_pos = [0, 0]
        self.exit_pos = [0, 0]
        self.coins = []
        self.particles = []
        self.moves_left = 0
        self.coins_collected = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_maze()
        self._place_game_elements()
        
        self.steps = 0
        self.score = 0
        self.coins_collected = 0
        self.moves_left = self.max_moves
        self.game_over = False
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        moved = False
        if movement != 0:
            self.moves_left -= 1
            
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
            
            if self._can_move(self.player_pos[0], self.player_pos[1], dx, dy):
                self.player_pos[0] += dx
                self.player_pos[1] += dy
                moved = True
                # SFX: Player step sound

        # Check for coin collection
        for i, coin_pos in enumerate(self.coins):
            if self.player_pos == coin_pos:
                self.coins.pop(i)
                self.coins_collected += 1
                self.score += 1
                reward += 1
                self._create_coin_particles(coin_pos)
                # SFX: Coin collect sound
                break

        # Check for termination conditions
        if self.player_pos == self.exit_pos and self.coins_collected >= self.coin_requirement:
            self.game_over = True
            terminated = True
            reward += 50
            self.score += 50
            # SFX: Win jingle
        elif self.moves_left <= 0:
            self.game_over = True
            terminated = True
            # SFX: Lose/out of moves sound
        
        self.steps += 1
        if self.steps >= self.max_steps:
            terminated = True
            self.game_over = True
        
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.color_bg)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "coins_collected": self.coins_collected,
        }

    def _generate_maze(self):
        self.maze = [[{'N': True, 'S': True, 'E': True, 'W': True, 'visited': False} for _ in range(self.maze_height)] for _ in range(self.maze_width)]
        
        stack = []
        start_x, start_y = self.np_random.integers(0, self.maze_width), self.np_random.integers(0, self.maze_height)
        stack.append((start_x, start_y))
        self.maze[start_x][start_y]['visited'] = True

        while stack:
            x, y = stack[-1]
            neighbors = []
            if y > 0 and not self.maze[x][y-1]['visited']: neighbors.append('N')
            if y < self.maze_height - 1 and not self.maze[x][y+1]['visited']: neighbors.append('S')
            if x < self.maze_width - 1 and not self.maze[x+1][y]['visited']: neighbors.append('E')
            if x > 0 and not self.maze[x-1][y]['visited']: neighbors.append('W')

            if neighbors:
                direction = self.np_random.choice(neighbors)
                if direction == 'N':
                    self.maze[x][y]['N'] = False
                    self.maze[x][y-1]['S'] = False
                    stack.append((x, y-1))
                    self.maze[x][y-1]['visited'] = True
                elif direction == 'S':
                    self.maze[x][y]['S'] = False
                    self.maze[x][y+1]['N'] = False
                    stack.append((x, y+1))
                    self.maze[x][y+1]['visited'] = True
                elif direction == 'E':
                    self.maze[x][y]['E'] = False
                    self.maze[x+1][y]['W'] = False
                    stack.append((x+1, y))
                    self.maze[x+1][y]['visited'] = True
                elif direction == 'W':
                    self.maze[x][y]['W'] = False
                    self.maze[x-1][y]['E'] = False
                    stack.append((x-1, y))
                    self.maze[x-1][y]['visited'] = True
            else:
                stack.pop()
    
    def _place_game_elements(self):
        self.player_pos = [0, 0]
        self.exit_pos = [self.maze_width - 1, self.maze_height - 1]
        
        possible_coin_locs = []
        for x in range(self.maze_width):
            for y in range(self.maze_height):
                if [x, y] != self.player_pos and [x, y] != self.exit_pos:
                    possible_coin_locs.append([x, y])
        
        coin_indices = self.np_random.choice(len(possible_coin_locs), self.num_coins_to_spawn, replace=False)
        self.coins = [possible_coin_locs[i] for i in coin_indices]

    def _can_move(self, x, y, dx, dy):
        if dx == 1 and not self.maze[x][y]['E']: return True
        if dx == -1 and not self.maze[x][y]['W']: return True
        if dy == 1 and not self.maze[x][y]['S']: return True
        if dy == -1 and not self.maze[x][y]['N']: return True
        return False

    def _render_game(self):
        # Draw Exit
        exit_rect = pygame.Rect(
            self.maze_offset_x + self.exit_pos[0] * self.cell_size,
            self.maze_offset_y + self.exit_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.color_exit, exit_rect)
        
        # Draw Coins
        coin_radius = self.cell_size // 4
        for x, y in self.coins:
            cx = self.maze_offset_x + int((x + 0.5) * self.cell_size)
            cy = self.maze_offset_y + int((y + 0.5) * self.cell_size)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, coin_radius, self.color_coin)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, coin_radius, self.color_coin)

        # Draw Particles
        for p in self.particles:
            p_radius = int(p['life'] / p['max_life'] * (self.cell_size / 6))
            if p_radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), p_radius, p['color'])

        # Draw Maze Walls
        for x in range(self.maze_width):
            for y in range(self.maze_height):
                px, py = self.maze_offset_x + x * self.cell_size, self.maze_offset_y + y * self.cell_size
                if self.maze[x][y]['N']:
                    pygame.draw.line(self.screen, self.color_wall, (px, py), (px + self.cell_size, py), 2)
                if self.maze[x][y]['S']:
                    pygame.draw.line(self.screen, self.color_wall, (px, py + self.cell_size), (px + self.cell_size, py + self.cell_size), 2)
                if self.maze[x][y]['W']:
                    pygame.draw.line(self.screen, self.color_wall, (px, py), (px, py + self.cell_size), 2)
                if self.maze[x][y]['E']:
                    pygame.draw.line(self.screen, self.color_wall, (px + self.cell_size, py), (px + self.cell_size, py + self.cell_size), 2)

        # Draw Player
        player_radius = self.cell_size // 3
        px = self.maze_offset_x + int((self.player_pos[0] + 0.5) * self.cell_size)
        py = self.maze_offset_y + int((self.player_pos[1] + 0.5) * self.cell_size)
        
        # Glow effect
        glow_surf = pygame.Surface((player_radius * 4, player_radius * 4), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, player_radius * 2, player_radius * 2, int(player_radius * 1.5), self.color_player_glow)
        self.screen.blit(glow_surf, (px - player_radius * 2, py - player_radius * 2))

        pygame.gfxdraw.filled_circle(self.screen, px, py, player_radius, self.color_player)
        pygame.gfxdraw.aacircle(self.screen, px, py, player_radius, self.color_player)

    def _render_ui(self):
        ui_margin = 20
        # Moves Left
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.color_text)
        self.screen.blit(moves_text, (ui_margin, ui_margin))
        
        # Coins Collected
        coins_text = self.font_ui.render(f"Coins: {self.coins_collected} / {self.coin_requirement}", True, self.color_text)
        self.screen.blit(coins_text, (ui_margin, ui_margin + 30))

        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.color_text)
        self.screen.blit(score_text, (ui_margin, ui_margin + 60))

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_pos == self.exit_pos and self.coins_collected >= self.coin_requirement:
                end_text = self.font_title.render("LEVEL COMPLETE!", True, self.color_exit)
            else:
                end_text = self.font_title.render("GAME OVER", True, (255, 80, 80))
            
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)

    def _create_coin_particles(self, grid_pos):
        px = self.maze_offset_x + (grid_pos[0] + 0.5) * self.cell_size
        py = self.maze_offset_y + (grid_pos[1] + 0.5) * self.cell_size
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            life = self.np_random.integers(15, 30)
            color_val = self.np_random.integers(180, 256)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': (color_val, color_val * 0.8, 0)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.98
            p['vel'][1] *= 0.98
            p['life'] -= 1

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Maze Runner")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    running = True
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                # For turn-based games, we step on key press
                if any([keys[pygame.K_UP], keys[pygame.K_DOWN], keys[pygame.K_LEFT], keys[pygame.K_RIGHT]]):
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated:
                        print(f"Game Over! Final Score: {info['score']}")

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play

    pygame.quit()