
# Generated: 2025-08-28T03:45:33.944793
# Source Brief: brief_02109.md
# Brief Index: 2109

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Use arrow keys to swap gems. Match 3 or more to score points and create combos."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match gems to reach a target score in this fast-paced, isometric puzzle game. "
        "Create cascades and combos to maximize your score before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_GEM_TYPES = 6
        self.MAX_MOVES = 20
        self.WIN_SCORE = 1000
        self.MAX_STEPS = 1000

        # Rewards
        self.REWARD_PER_GEM = 1.0
        self.REWARD_COMBO_BONUS = 1.0
        self.REWARD_INVALID_MOVE = -0.1
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -10.0

        # Visuals
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 80)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_INVALID = (255, 50, 50, 100)
        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 255, 80),  # Yellow
            (255, 80, 255),  # Magenta
            (80, 255, 255),  # Cyan
        ]
        self.TILE_WIDTH = 56
        self.TILE_HEIGHT = self.TILE_WIDTH // 2
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.grid = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.selected_pos = None
        self.particles = []
        self.show_invalid_flash = False
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.steps = 0
        self.game_over = False
        self.selected_pos = None
        self.particles = []
        self.show_invalid_flash = False
        
        self._generate_solvable_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.show_invalid_flash = False

        movement = action[0]  # 0-4: none/up/down/left/right
        
        if movement > 0:
            if self.selected_pos is None:
                self.selected_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
            else:
                reward = self._handle_swap(movement)
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += self.REWARD_WIN
            else:
                reward += self.REWARD_LOSE

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_swap(self, direction):
        r1, c1 = self.selected_pos
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction - 1]
        r2, c2 = r1 + dr, c1 + dc

        self.selected_pos = None 

        if not (0 <= r2 < self.GRID_SIZE and 0 <= c2 < self.GRID_SIZE):
            self.show_invalid_flash = True
            return self.REWARD_INVALID_MOVE

        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        matches = self._find_all_matches()
        if not matches:
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            self.show_invalid_flash = True
            # play_sound('invalid.wav')
            return self.REWARD_INVALID_MOVE
        
        self.moves_left -= 1
        reward = 0
        combo = 0
        
        while matches:
            num_matched = len(matches)
            self.score += num_matched * 10 * (combo + 1)
            reward += num_matched * self.REWARD_PER_GEM
            if combo > 0:
                reward += self.REWARD_COMBO_BONUS * combo
            
            for r, c in matches:
                self._create_particles(r, c, self.grid[r, c])
            
            # play_sound('match.wav')
            for r, c in matches:
                self.grid[r, c] = -1

            self._apply_gravity()
            self._refill_board()
            
            combo += 1
            matches = self._find_all_matches()
        
        return reward

    def _check_termination(self):
        return self.score >= self.WIN_SCORE or self.moves_left <= 0 or self.steps >= self.MAX_STEPS

    def _generate_solvable_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_all_matches() and self._find_possible_moves():
                break

    def _find_all_matches(self, grid=None):
        if grid is None:
            grid = self.grid
        
        matched_gems = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = grid[r, c]
                if gem_type == -1: continue

                if c < self.GRID_SIZE - 2 and grid[r, c+1] == gem_type and grid[r, c+2] == gem_type:
                    for i in range(3):
                        if c + i < self.GRID_SIZE and grid[r, c+i] == gem_type: matched_gems.add((r, c+i))
                
                if r < self.GRID_SIZE - 2 and grid[r+1, c] == gem_type and grid[r+2, c] == gem_type:
                    for i in range(3):
                        if r + i < self.GRID_SIZE and grid[r+i, c] == gem_type: matched_gems.add((r+i, c))
        return matched_gems

    def _find_possible_moves(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                temp_grid = self.grid.copy()
                if c < self.GRID_SIZE - 1:
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if self._find_all_matches(temp_grid): return True
                
                temp_grid = self.grid.copy()
                if r < self.GRID_SIZE - 1:
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if self._find_all_matches(temp_grid): return True
        return False

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if empty_row != r:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = -1
                    empty_row -= 1
    
    def _refill_board(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _to_iso(self, r, c):
        return int(self.ORIGIN_X + (c - r) * self.TILE_WIDTH / 2), int(self.ORIGIN_Y + (c + r) * self.TILE_HEIGHT / 2)

    def _create_particles(self, r, c, gem_type):
        center_x, center_y = self._to_iso(r, c)
        color = self.GEM_COLORS[gem_type]
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'color': color})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
                radius = max(0, int(p['life'] / 5))
                pygame.draw.circle(self.screen, p['color'], [int(p['pos'][0]), int(p['pos'][1])], radius)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, self._to_iso(r, 0), self._to_iso(r, self.GRID_SIZE), 1)
        for c in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, self._to_iso(0, c), self._to_iso(self.GRID_SIZE, c), 1)

        gem_radius = int(self.TILE_HEIGHT * 0.8)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    iso_x, iso_y = self._to_iso(r, c)
                    color = self.GEM_COLORS[gem_type]
                    pygame.gfxdraw.filled_circle(self.screen, iso_x, iso_y, gem_radius, color)
                    highlight_color = tuple(min(255, val + 60) for val in color)
                    pygame.gfxdraw.aacircle(self.screen, iso_x, iso_y, gem_radius, highlight_color)
                    
        if self.selected_pos:
            r, c = self.selected_pos
            points = [self._to_iso(r, c), self._to_iso(r + 1, c), self._to_iso(r + 1, c + 1), self._to_iso(r, c + 1)]
            pygame.draw.polygon(self.screen, (255, 255, 0), points, 3)

        self._update_and_draw_particles()
        
        if self.show_invalid_flash:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill(self.COLOR_INVALID)
            self.screen.blit(s, (0, 0))

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        text_rect = moves_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(moves_text, text_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            end_text = self.font_main.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, end_rect)
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, final_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import time
    
    env = GameEnv(render_mode="rgb_array")
    
    MANUAL_PLAY = True

    if MANUAL_PLAY:
        real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Gem Matcher")
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            action = np.array([0, 0, 0])
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    elif event.key == pygame.K_r:
                        obs, info = env.reset()
                        print(f"Game reset. Score: {info['score']}, Moves: {info['moves_left']}")
            
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}, Terminated: {terminated}")
            
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30)
            
        env.close()
    else:
        obs, info = env.reset()
        total_reward = 0
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if (i % 20 == 0):
                print(f"Step {i}: Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")
            if terminated:
                print(f"Episode finished after {i+1} steps. Final Info: {info}, Total Reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
        env.close()